"""
AstrBot 图像生成插件主模块

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .core.config_manager import (
    LLM_TOOL_IMAGE_GENERATION,
    LLM_TOOL_PRESET_EDIT,
    LLM_TOOL_PRESET_QUERY,
    LLM_TOOL_TASK_MANAGEMENT,
    ConfigManager,
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_TASK_ID,
    RESULT_INFO_USAGE,
)
from .core.generator import ImageGenerator
from .core.image_processor import ImageProcessor
from .core.llm_result_handler import LLMResultHandler
from .core.llm_tool import (
    ImageGenerationTool,
    ImageTaskTool,
    PresetEditTool,
    PresetQueryTool,
    adjust_tool_parameters,
)
from .core.public_api import ImageGenerationPublicAPI
from .core.reference_collector import collect_command_reference_images
from .core.constants import UNSPECIFIED_OPTION
from .core.logging_utils import (
    log_prefix,
    mask_sensitive,
    safe_log_text,
)
from .core.safety_auditor import SafetyAuditor
from .core.task_manager import GenerationTaskRecord, TaskManager
from .core.types import GenerationRequest, ImageCapability, ImageData
from .core.usage_manager import UsageManager
from .core.utils import validate_aspect_ratio, validate_resolution


LOG = log_prefix("Plugin")


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class ImageGenerationPlugin(Star):
    """图像生成插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context

        # 数据目录配置：持久数据放插件数据目录，图片临时文件放 AstrBot 官方临时目录
        self.data_dir = StarTools.get_data_dir()
        self.astrbot_temp_dir = Path(get_astrbot_temp_path())
        self.image_temp_dir = self.astrbot_temp_dir / "astrbot_plugin_image_generation"
        self.image_temp_dir.mkdir(parents=True, exist_ok=True)

        # 初始化配置管理器
        self.config_manager = ConfigManager(config)

        # 初始化使用数据管理器
        self.usage_manager = UsageManager(
            str(self.data_dir), self.config_manager.usage_settings
        )

        # 初始化图片处理器
        self.image_processor = ImageProcessor(
            str(self.image_temp_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            str(self.data_dir),
            allowed_local_base_dirs=[str(self.astrbot_temp_dir)],
        )

        # 初始化任务管理器
        self.task_manager = TaskManager()

        # 初始化 LLM 工具结果处理器
        self.llm_result_handler = LLMResultHandler(
            context=self.context,
            config_manager=self.config_manager,
            task_manager=self.task_manager,
            create_background_task=self.create_background_task,
        )

        # 初始化安全审核器
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # 初始化供其他插件调用的公共 API
        self.public_api = ImageGenerationPublicAPI(self)

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self.request_semaphore: asyncio.Semaphore | None = None

    # ---------------------- 生命周期 ----------------------

    async def initialize(self):
        """插件加载时调用"""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.request_semaphore = asyncio.Semaphore(
                self.config_manager.max_concurrent_tasks
            )
            logger.info(
                f"{LOG} 初始化生图生成器: "
                f"供应商={safe_log_text(self.config_manager.adapter_config.name)}，"
                f"模型={safe_log_text(self.config_manager.adapter_config.model)}，"
                f"最大并发生图请求={self.config_manager.max_concurrent_tasks}"
            )
        else:
            logger.error(f"{LOG} 适配器配置加载失败，插件未初始化")

        # 注册 LLM 工具
        self._register_llm_tools()

        # 配置定时任务
        self._setup_tasks()

        # 执行启动任务（在后台异步执行）
        self.task_manager.create_task(self.task_manager.run_startup_tasks())

        logger.info(
            f"{LOG} 插件加载完成，模型: {safe_log_text(self.config_manager.adapter_config.model if self.config_manager.adapter_config else '未知')}"
        )

    async def terminate(self):
        """插件卸载时调用"""
        try:
            if self.generator:
                await self.generator.close()
            await self.task_manager.cancel_all()
            logger.info(f"{LOG} 插件已卸载")
        except Exception as exc:
            logger.error(f"{LOG} 卸载清理出错: {exc}", exc_info=True)

    # ---------------------- 内部工具 ----------------------

    def _setup_tasks(self) -> None:
        """配置并启动定时任务。"""
        # Jimeng2API 自动领积分任务
        self._setup_jimeng_token_task()

    def _register_llm_tools(self) -> None:
        """Register enabled LLM tools."""
        tools = []
        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_IMAGE_GENERATION):
            if self.generator:
                image_tool = ImageGenerationTool(plugin=self)
                self._adjust_tool_parameters(image_tool)
                tools.append(image_tool)
            else:
                logger.warning(f"{LOG} 生图工具已启用，但生成器未初始化")

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_PRESET_QUERY):
            tools.append(PresetQueryTool(plugin=self))

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_TASK_MANAGEMENT):
            tools.append(ImageTaskTool(plugin=self))

        if self.config_manager.is_llm_tool_enabled(LLM_TOOL_PRESET_EDIT):
            tools.append(PresetEditTool(plugin=self))

        if tools:
            self.context.add_llm_tools(*tools)
            logger.info(
                f"{LOG} 已注册 LLM 工具: " + ", ".join(tool.name for tool in tools)
            )

    def _setup_jimeng_token_task(self) -> None:
        """配置即梦自动领积分任务。

        该任务会：
        1. 在插件启动时执行一次（通过启动任务）
        2. 每天日期变更时自动执行（通过每日任务）

        注意：只要配置中包含即梦渠道，就会启用该任务，
        无论当前使用的是哪个渠道。
        """
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter
        from .core.types import AdapterType

        # 检查配置中是否包含即梦渠道（而非检查当前适配器）
        jimeng_config = self.config_manager.get_provider_config(AdapterType.JIMENG2API)
        if not jimeng_config:
            return

        # 创建专门用于任务的即梦适配器实例
        jimeng_adapter = Jimeng2APIAdapter(jimeng_config)

        # 1. 注册为启动任务，插件启动时执行一次
        self.task_manager.register_startup_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
        )

        # 2. 注册为每日任务，日期变更时执行
        self.task_manager.start_daily_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
            check_interval_seconds=300,  # 每5分钟检查一次日期变更
            run_immediately=False,  # 启动任务已处理，无需重复执行
        )
        logger.info(f"{LOG} 已配置即梦2API自动领积分任务（启动时+每日）")

    def _adjust_tool_parameters(self, tool: ImageGenerationTool) -> None:
        """根据适配器能力动态调整工具参数。"""
        if not self.generator or not self.generator.adapter:
            return
        capabilities = self.generator.adapter.get_capabilities()
        adjust_tool_parameters(tool, capabilities)
        props = tool.parameters.get("properties", {})
        if not self.config_manager.personas:
            props.pop("persona", None)
        elif isinstance(props.get("persona"), dict):
            props["persona"].pop("enum", None)
            if persona_names := "、".join(self.config_manager.personas):
                props["persona"]["description"] = (
                    str(props["persona"].get("description", "")).rstrip("。")
                    + f"。可用人设: {persona_names}；多个名称可用空格分隔。"
                )

    def create_background_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro, name=name)

    def create_generation_task(
        self,
        *,
        task_id: str,
        source: str,
        prompt: str,
        images_data: list[ImageData] | None,
        unified_msg_origin: str,
        aspect_ratio: str,
        resolution: str,
        image_count: int,
        is_usage_limit_admin: bool,
        preset: str | None = None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        source_event: AstrMessageEvent | None = None,
        auto_send: bool = True,
    ) -> GenerationTaskRecord:
        """Create and track an image generation task in the unified task manager."""
        if preset is None:
            preset, preset_label = self._format_template_summary(
                presets or [],
                personas or [],
            )
        image_count = self.normalize_image_count(image_count)
        record = self.task_manager.create_generation_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                image_count=image_count,
                task_id=task_id,
                is_usage_limit_admin=is_usage_limit_admin,
                deliver_via_ai=source == "LLM工具",
                auto_send=auto_send,
            ),
            task_id=task_id,
            source=source,
            unified_msg_origin=unified_msg_origin,
            prompt=prompt,
            reference_image_count=len(images_data or []),
            requested_count=image_count,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            preset=preset,
            preset_label=preset_label,
        )
        if source == "LLM工具":
            self.llm_result_handler.attach_task_wakeup(
                record,
                source_event=source_event,
            )
        return record

    def is_usage_limit_admin(self, event: AstrMessageEvent) -> bool:
        """Return whether an event sender is an AstrBot admin for usage limits."""
        try:
            return bool(event.is_admin())
        except Exception as exc:
            logger.debug(f"{LOG} 获取管理员状态失败: {exc}")
            return False

    def normalize_image_count(self, value: Any) -> int:
        """Normalize requested image count using configured bounds."""
        try:
            count = int(value)
        except (TypeError, ValueError):
            count = self.config_manager.default_image_count
        return max(1, min(count, self.config_manager.max_image_count))

    def _parse_command_image_count(self, prompt: str) -> tuple[int, str]:
        """Parse optional image count from command prompt suffix."""
        raw_prompt = prompt.strip()
        default_count = self.config_manager.default_image_count
        if not raw_prompt:
            return default_count, ""

        tokens = raw_prompt.split()
        if tokens[-1].isdecimal():
            return self.normalize_image_count(tokens[-1]), " ".join(tokens[:-1]).strip()

        return default_count, raw_prompt

    def _find_named_entry(self, entries: dict[str, Any], token: str) -> str | None:
        """Find an entry by exact or case-insensitive name."""
        if token in entries:
            return token
        lowered_token = token.lower()
        for name in entries:
            if name.lower() == lowered_token:
                return name
        return None

    def _parse_preset_prompt(
        self,
        preset_content: Any,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[str, str, str]:
        """Parse a preset prompt and optional generation overrides."""
        preset_prompt = str(preset_content or "").strip()
        if not preset_prompt.startswith("{"):
            return preset_prompt, aspect_ratio, resolution

        try:
            preset_data = json.loads(preset_prompt)
        except json.JSONDecodeError:
            return preset_prompt, aspect_ratio, resolution

        if not isinstance(preset_data, dict):
            return preset_prompt, aspect_ratio, resolution

        preset_prompt = str(preset_data.get("prompt", "") or "").strip()
        aspect_ratio = str(preset_data.get("aspect_ratio") or aspect_ratio)
        resolution = str(preset_data.get("resolution") or resolution)
        return preset_prompt, aspect_ratio, resolution

    def _parse_command_prompt_templates(
        self,
        prompt: str,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[str, str, str, list[str], list[str], list[tuple[str, str]]]:
        """Apply leading space-separated preset/persona names to a command prompt."""
        raw_prompt = prompt.strip()
        if not raw_prompt:
            return "", aspect_ratio, resolution, [], [], []

        tokens = raw_prompt.split()
        prompt_parts: list[str] = []
        matched_presets: list[str] = []
        matched_personas: list[str] = []
        persona_images: list[tuple[str, str]] = []
        extra_content = ""

        for index, token in enumerate(tokens):
            matched_preset = self._find_named_entry(self.config_manager.presets, token)
            if matched_preset:
                preset_prompt, aspect_ratio, resolution = self._parse_preset_prompt(
                    self.config_manager.presets[matched_preset],
                    aspect_ratio,
                    resolution,
                )
                if preset_prompt:
                    prompt_parts.append(preset_prompt)
                matched_presets.append(matched_preset)
                continue

            matched_persona = self._find_named_entry(
                self.config_manager.personas, token
            )
            if matched_persona:
                persona = self.config_manager.personas[matched_persona]
                persona_prompt = persona.prompt.strip()
                if persona_prompt:
                    prompt_parts.append(persona_prompt)
                if persona.image:
                    persona_images.append((matched_persona, persona.image))
                matched_personas.append(matched_persona)
                continue

            extra_content = " ".join(tokens[index:]).strip()
            break

        if not matched_presets and not matched_personas:
            return raw_prompt, aspect_ratio, resolution, [], [], []

        if extra_content:
            prompt_parts.append(extra_content)

        return (
            " ".join(part for part in prompt_parts if part).strip(),
            aspect_ratio,
            resolution,
            matched_presets,
            matched_personas,
            persona_images,
        )

    def _format_template_summary(
        self,
        matched_presets: list[str],
        matched_personas: list[str],
    ) -> tuple[str | None, str]:
        """Format matched preset/persona names for task metadata."""
        if matched_presets and matched_personas:
            return (
                "；".join(
                    (
                        f"预设: {'、'.join(matched_presets)}",
                        f"人设: {'、'.join(matched_personas)}",
                    )
                ),
                "预设/人设",
            )
        if matched_presets:
            return "、".join(matched_presets), "预设"
        if matched_personas:
            return "、".join(matched_personas), "人设"
        return None, "预设"

    def _format_start_template_values(
        self,
        *,
        preset: str | None,
        presets: list[str] | None,
        personas: list[str] | None,
    ) -> dict[str, str]:
        """Build preset/persona placeholder values for the start-task template."""
        preset_names = "、".join(presets or [])
        persona_names = "、".join(personas or [])
        return {
            "preset": preset_names or (preset or ""),
            "presets": preset_names,
            "persona": persona_names,
            "personas": persona_names,
            "preset_block": f"[预设: {preset_names}]" if preset_names else "",
            "persona_block": f"[人设: {persona_names}]" if persona_names else "",
        }

    def format_start_task_message(
        self,
        *,
        prompt: str,
        reference_image_count: int,
        image_count: int,
        preset: str | None,
        preset_label: str = "预设",
        presets: list[str] | None = None,
        personas: list[str] | None = None,
        aspect_ratio: str,
        resolution: str,
        task_id: str,
    ) -> str:
        """Render start-task message from configured template."""
        template = self.config_manager.start_task_message_template
        if not template.strip():
            return ""

        model = ""
        if self.config_manager.adapter_config:
            model = (
                f"{self.config_manager.adapter_config.name}/"
                f"{self.config_manager.adapter_config.model}"
            )

        values = _SafeFormatDict(
            reference_image_count=str(reference_image_count),
            image_count=str(image_count),
            count=str(image_count),
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
            model=model,
            mode="图生图" if reference_image_count else "文生图",
            preset_label=preset_label,
            image_count_block=f"[数量: {image_count}张]" if image_count > 1 else "",
            count_block=f"[数量: {image_count}张]" if image_count > 1 else "",
            reference_images_block=(
                f"[{reference_image_count}张参考图]" if reference_image_count else ""
            ),
            **self._format_start_template_values(
                preset=preset,
                presets=presets,
                personas=personas,
            ),
        )

        try:
            return template.format_map(values)
        except Exception as exc:
            logger.warning(f"{LOG} 开始任务提示模板格式化失败: {exc}")
            return (
                "已开始生图任务{reference_images_block}{preset_block}"
                "{persona_block}{image_count_block} [任务ID: {task_id}]"
            ).format_map(values)

    def format_task_detail(self, record: GenerationTaskRecord) -> str:
        """Format one task record for command output."""
        lines = [f"🧾 任务 {record.task_id}: {record.status_label}"]
        lines.append(f"来源: {record.source}")
        lines.append(f"提示词: {record.prompt_summary or '无'}")
        lines.append(f"数量: {record.result_count}/{record.requested_count}张")
        lines.append(f"参考图: {record.reference_image_count}张")
        lines.append(f"宽高比: {record.aspect_ratio}，分辨率: {record.resolution}")
        if record.max_retry_attempts:
            progress = f"第 {record.current_index}/{record.requested_count} 张"
            lines.append(
                f"重试: {progress}，{record.retry_attempt}/{record.max_retry_attempts}"
            )
        if record.preset:
            lines.append(f"{record.preset_label}: {record.preset}")

        if record.started_at:
            duration = record.duration_seconds
            if duration is not None:
                lines.append(f"耗时: {duration:.2f}s")
        else:
            lines.append(f"排队: {record.queued_seconds:.2f}s")

        if record.message:
            lines.append(f"说明: {record.message}")
        if record.items:
            lines.append("子请求:")
            for item in sorted(
                record.items.values(), key=lambda task_item: task_item.index
            ):
                if item.status == "succeeded":
                    item_line = f"  {item.index}. 成功 {item.result_count}张"
                elif item.status == "failed":
                    item_line = f"  {item.index}. 失败"
                else:
                    item_line = f"  {item.index}. 运行中"
                if item.max_retry_attempts:
                    item_line += (
                        f"，重试 {item.retry_attempts}/{item.max_retry_attempts}"
                    )
                lines.append(item_line)
        return "\n".join(lines)

    def format_task_list(self, records: list[GenerationTaskRecord]) -> str:
        """Format a compact task list for command output."""
        if not records:
            return "📭 当前没有正在进行的生图任务"

        lines = ["📋 正在进行的生图任务:"]
        for index, record in enumerate(records, 1):
            parts = [
                f"{record.task_id}",
                record.status_label,
                record.source,
                f"数量{record.result_count}/{record.requested_count}张",
                f"参考图{record.reference_image_count}张",
            ]
            lines.append(f"{index}. " + " | ".join(parts))
        lines.append(
            "\n用法: \n/生图任务 <编号或任务ID> 查看详情\n/生图取消 <编号或任务ID> 取消任务"
        )
        return "\n".join(lines)

    def format_image_command_help(self) -> str:
        """Format help text for the image generation command."""
        adapter_config = self.config_manager.adapter_config
        current_model = (
            f"{adapter_config.name}/{adapter_config.model}"
            if adapter_config
            else "未配置"
        )
        lines = [
            "🎨 生图帮助",
            f"当前模型: {current_model}",
            "",
            "指令列表:",
            "/生图 [预设/人设] [提示词] [数量]",
            "/生图模型 - 查看或切换模型",
            "/生图任务 [编号或任务ID]- 查看正在进行的任务",
            "/生图取消 <编号或任务ID> - 取消指定任务",
            "/预设 [添加/删除] - 查看或管理预设/人设",
        ]
        return "\n".join(lines)

    def resolve_task_reference(
        self,
        unified_msg_origin: str,
        task_ref: str,
        *,
        include_finished: bool = False,
    ) -> GenerationTaskRecord | None:
        """Resolve a task id or active list number into a task for one session."""
        task_ref = task_ref.strip()
        if not task_ref:
            return None

        active_records = self.task_manager.list_generation_tasks(
            unified_msg_origin=unified_msg_origin,
            include_finished=False,
            limit=10,
        )
        if task_ref.isdigit():
            index = int(task_ref) - 1
            if 0 <= index < len(active_records):
                return active_records[index]

        for record in active_records:
            if record.task_id == task_ref:
                return record

        if include_finished:
            record = self.task_manager.get_generation_task(task_ref)
            if record and record.unified_msg_origin == unified_msg_origin:
                return record
        return None

    def resolve_active_task_reference(
        self, unified_msg_origin: str, task_ref: str
    ) -> GenerationTaskRecord | None:
        """Resolve a task id or list number into an active task for one session."""
        return self.resolve_task_reference(
            unified_msg_origin,
            task_ref,
            include_finished=False,
        )

    # ---------------------- 核心生图逻辑 ----------------------

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[ImageData] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        image_count: int = 1,
        task_id: str | None = None,
        is_usage_limit_admin: bool = False,
        deliver_via_ai: bool = False,
        auto_send: bool = True,
    ) -> None:
        """异步生成图片并发送。"""
        if not self.generator or not self.generator.adapter:
            if task_id:
                self.task_manager.mark_generation_task_failed(
                    task_id, "生图生成器未初始化"
                )
                logger.warning(
                    f"{log_prefix('Task', task_id)} 生成器未初始化，任务提前结束"
                )
            return

        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        capabilities = self.generator.adapter.get_capabilities()

        # 检查并清理不支持的参数
        task_log = log_prefix("Task", task_id)
        image_count = self.normalize_image_count(image_count)
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"{task_log} 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
            )
            images_data = None

        if (
            not (capabilities & ImageCapability.ASPECT_RATIO)
            and aspect_ratio != UNSPECIFIED_OPTION
        ):
            logger.debug(
                f"{task_log} 当前适配器不支持指定比例，已忽略参数: {safe_log_text(aspect_ratio)}"
            )
            aspect_ratio = UNSPECIFIED_OPTION

        if (
            not (capabilities & ImageCapability.RESOLUTION)
            and resolution != UNSPECIFIED_OPTION
        ):
            logger.debug(
                f"{task_log} 当前适配器不支持指定分辨率，已忽略参数: {safe_log_text(resolution)}"
            )
            resolution = UNSPECIFIED_OPTION

        final_ar = validate_aspect_ratio(aspect_ratio) or None
        if final_ar == UNSPECIFIED_OPTION:
            final_ar = None
        final_res = validate_resolution(resolution)
        if final_res == UNSPECIFIED_OPTION:
            final_res = None

        images: list[ImageData] = []
        if images_data:
            for image in images_data:
                if isinstance(image, ImageData):
                    images.append(image)
                else:
                    data, mime = image
                    images.append(ImageData(data=data, mime_type=mime))
        self.task_manager.update_generation_task_references(
            task_id,
            reference_image_count=len(images),
        )

        logger.debug(
            f"{task_log} 生图请求已规范化: 数量={image_count}张，参考图={len(images)}张，"
            f"宽高比={safe_log_text(final_ar or UNSPECIFIED_OPTION)}，"
            f"分辨率={safe_log_text(final_res or UNSPECIFIED_OPTION)}"
        )

        await self._do_generate_and_send(
            prompt,
            unified_msg_origin,
            images,
            final_ar,
            final_res,
            image_count,
            task_id,
            is_usage_limit_admin,
            deliver_via_ai,
            auto_send,
        )

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        image_count: int,
        task_id: str,
        is_usage_limit_admin: bool,
        deliver_via_ai: bool = False,
        auto_send: bool = True,
    ) -> None:
        """执行生成逻辑并发送结果。"""
        start_time = time.time()
        task_log = log_prefix("Task", task_id)
        self.task_manager.mark_generation_task_running(task_id)
        if not self.generator:
            logger.warning(f"{task_log} 生成器未初始化，跳过生成请求")
            self.task_manager.mark_generation_task_failed(task_id, "生图生成器未初始化")
            return
        logger.debug(
            f"{task_log} 调用生图适配器: 数量={image_count}张，参考图={len(images)}张，"
            f"宽高比={safe_log_text(aspect_ratio or UNSPECIFIED_OPTION)}，"
            f"分辨率={safe_log_text(resolution or UNSPECIFIED_OPTION)}"
        )

        converted_images = await self.generator.convert_reference_images(images)

        generated_file_paths, errors = await self._generate_image_requests_concurrently(
            task_id=task_id,
            task_log=task_log,
            prompt=prompt,
            images=converted_images,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_count=image_count,
        )

        end_time = time.time()
        duration = end_time - start_time

        if not generated_file_paths:
            error = "; ".join(errors) or "模型未返回图片"
            self.task_manager.mark_generation_task_failed(task_id, error)
            if deliver_via_ai or not auto_send or not unified_msg_origin:
                return
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 生成失败: {error}"),
            )
            return

        logger.debug(
            f"{task_log} 生成完成，耗时: {duration:.2f}s, 图片数量: {len(generated_file_paths)}/{image_count}"
        )

        # 生图后图片审核
        image_allowed, image_reason = await self.safety_auditor.audit_generated_images(
            prompt=prompt,
            image_paths=generated_file_paths,
            unified_msg_origin=unified_msg_origin,
        )
        if not image_allowed:
            # 生成已消耗模型调用成本，即使审核失败也计入实际生成额度。
            if unified_msg_origin:
                self.usage_manager.record_usage(
                    unified_msg_origin,
                    is_admin=is_usage_limit_admin,
                    count=len(generated_file_paths),
                )
            self.task_manager.mark_generation_task_failed(
                task_id,
                f"图片内容审核未通过: {image_reason}",
            )
            if deliver_via_ai or not auto_send or not unified_msg_origin:
                return
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 图片内容审核未通过: {image_reason}"),
            )
            return

        result_message = "图片已生成，等待 AI 处理" if deliver_via_ai else "图片已发送"
        if errors:
            result_message = f"{result_message}；部分失败: {'; '.join(errors)}"

        self.task_manager.mark_generation_task_succeeded(
            task_id,
            result_count=len(generated_file_paths),
            result_paths=generated_file_paths,
            message=result_message,
        )

        # 记录实际成功生成的图片数量
        if unified_msg_origin:
            self.usage_manager.record_usage(
                unified_msg_origin,
                is_admin=is_usage_limit_admin,
                count=len(generated_file_paths),
            )

        if deliver_via_ai or not auto_send or not unified_msg_origin:
            return

        info_parts = []
        if self.config_manager.should_show_result_info(RESULT_INFO_DURATION):
            info_parts.append(f"📊 耗时: {duration:.2f}s")

        if (
            self.config_manager.should_show_result_info(RESULT_INFO_MODEL)
            and self.config_manager.adapter_config
        ):
            info_parts.append(
                f"🤖 模型: {self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            )

        if self.config_manager.should_show_result_info(RESULT_INFO_COUNT):
            info_parts.append(f"🖼️ 数量: {len(generated_file_paths)}张")

        if self.config_manager.should_show_result_info(RESULT_INFO_TASK_ID):
            info_parts.append(f"🧾 任务ID: {task_id}")

        if (
            self.config_manager.should_show_result_info(RESULT_INFO_USAGE)
            and self.usage_manager.is_daily_limit_enabled()
        ):
            count = self.usage_manager.get_usage_count(unified_msg_origin)
            daily_limit = (
                "∞"
                if self.usage_manager.is_limit_exempt(
                    unified_msg_origin,
                    is_admin=is_usage_limit_admin,
                )
                else str(self.usage_manager.get_daily_limit())
            )
            info_parts.append(f"📅 今日用量: {count}/{daily_limit}")

        if info_parts:
            info_message = "\n".join(info_parts)
        else:
            info_message = ""

        await self._send_generated_images(
            unified_msg_origin,
            generated_file_paths,
            info_message=info_message,
        )

    async def _send_generated_images(
        self,
        unified_msg_origin: str,
        image_paths: list[str],
        *,
        info_message: str = "",
    ) -> None:
        """按配置将生成图片分批发送，避免单条消息图片过多。"""
        max_per_message = max(1, self.config_manager.max_images_per_message)
        total = len(image_paths)
        for start in range(0, total, max_per_message):
            batch_paths = image_paths[start : start + max_per_message]
            chain = MessageChain()
            for file_path in batch_paths:
                chain.file_image(file_path)

            is_last_batch = start + max_per_message >= total
            if is_last_batch and info_message:
                chain.message("\n" + info_message)

            await self.context.send_message(unified_msg_origin, chain)

    async def _generate_image_requests_concurrently(
        self,
        *,
        task_id: str,
        task_log: str,
        prompt: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        image_count: int,
    ) -> tuple[list[str], list[str]]:
        """Generate all requested images concurrently under request-level limits."""
        generated_file_paths: list[str] = []
        errors: list[str] = []
        pending_tasks: dict[asyncio.Task, int] = {}
        next_index = 1
        max_pending_requests = min(
            image_count,
            max(1, self.config_manager.max_concurrent_tasks),
        )

        async def schedule_next_request() -> None:
            nonlocal next_index
            if next_index > image_count:
                return
            current_index = next_index
            next_index += 1
            self.task_manager.update_generation_task_progress(
                task_id,
                current_index=current_index,
                result_count=len(generated_file_paths),
                message=(
                    f"正在生成第 {current_index}/{image_count} 张，"
                    f"已完成 {len(generated_file_paths)}/{image_count} 张"
                ),
            )
            task = asyncio.create_task(
                self._generate_one_image_request(
                    GenerationRequest(
                        prompt=prompt,
                        images=images,
                        aspect_ratio=aspect_ratio,
                        resolution=resolution,
                        task_id=task_id,
                        batch_index=current_index,
                        batch_count=image_count,
                        retry_status_callback=lambda retry_attempt, max_retry_attempts, current_index=current_index: (
                            self.task_manager.update_generation_task_retry_status(
                                task_id,
                                current_index=current_index,
                                retry_attempt=retry_attempt,
                                max_retry_attempts=max_retry_attempts,
                            )
                        ),
                    )
                ),
                name=f"image_generation_request:{task_id}:{current_index}",
            )
            pending_tasks[task] = current_index

        while len(pending_tasks) < max_pending_requests:
            await schedule_next_request()

        try:
            while pending_tasks:
                done_tasks, _ = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for done_task in done_tasks:
                    current_index = pending_tasks.pop(done_task)
                    try:
                        result = done_task.result()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        result = None
                        error_message = f"第 {current_index} 张生成失败: {exc}"
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error=str(exc),
                        )
                        logger.warning(
                            f"{task_log} {safe_log_text(error_message, 200)}"
                        )

                    if result is None:
                        continue

                    if result.error:
                        error_message = f"第 {current_index} 张生成失败: {result.error}"
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error=result.error,
                        )
                        logger.warning(
                            f"{task_log} {safe_log_text(error_message, 200)}"
                        )
                    elif not result.images:
                        error_message = f"第 {current_index} 张生成失败: 模型未返回图片"
                        errors.append(error_message)
                        self.task_manager.update_generation_task_item_result(
                            task_id,
                            index=current_index,
                            status="failed",
                            error="模型未返回图片",
                        )
                        logger.warning(f"{task_log} {error_message}")
                    else:
                        saved_count = 0
                        for img_bytes in result.images:
                            file_path = self.image_processor.save_generated_image(
                                task_id, img_bytes
                            )
                            if file_path:
                                generated_file_paths.append(file_path)
                                saved_count += 1
                            else:
                                error_message = (
                                    f"第 {current_index} 张生成失败: 未能保存图片"
                                )
                                errors.append(error_message)
                                self.task_manager.update_generation_task_item_result(
                                    task_id,
                                    index=current_index,
                                    status="failed",
                                    result_count=saved_count,
                                    error="未能保存图片",
                                )
                                logger.warning(f"{task_log} {error_message}")
                                break
                        else:
                            self.task_manager.update_generation_task_item_result(
                                task_id,
                                index=current_index,
                                status="succeeded",
                                result_count=saved_count,
                            )

                    self.task_manager.update_generation_task_progress(
                        task_id,
                        current_index=min(next_index, image_count),
                        result_count=len(generated_file_paths),
                        message=f"已生成 {len(generated_file_paths)}/{image_count} 张",
                    )

                    if next_index <= image_count:
                        await schedule_next_request()
        finally:
            for pending_task in pending_tasks:
                pending_task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)

        return generated_file_paths, errors

    async def _generate_one_image_request(
        self,
        request: GenerationRequest,
    ):
        """Run one adapter generation request under the request-level semaphore."""
        if not self.generator:
            return None
        if self.request_semaphore is None:
            return await self.generator.generate_preconverted(
                request,
                images=request.images,
            )
        async with self.request_semaphore:
            return await self.generator.generate_preconverted(
                request,
                images=request.images,
            )

    # ---------------------- 指令处理 ----------------------

    @filter.command("生图任务")
    async def image_task_command(self, event: AstrMessageEvent, task_id: str = ""):
        """查看生图任务列表或指定任务详情。"""
        user_id = event.unified_msg_origin
        task_id = (task_id or "").strip()

        if task_id:
            record = self.resolve_task_reference(
                user_id,
                task_id,
                include_finished=True,
            )
            if not record:
                yield event.plain_result(f"❌ 任务不存在或已被清理: {task_id}")
                return
            if record.unified_msg_origin != user_id and not self.is_usage_limit_admin(
                event
            ):
                yield event.plain_result("❌ 不能查看其他会话的生图任务")
                return
            yield event.plain_result(self.format_task_detail(record))
            return

        records = self.task_manager.list_generation_tasks(
            unified_msg_origin=user_id,
            include_finished=False,
            limit=10,
        )
        yield event.plain_result(self.format_task_list(records))

    @filter.command("生图取消")
    async def cancel_image_task_command(
        self, event: AstrMessageEvent, task_id: str = ""
    ):
        """取消指定生图任务。"""
        task_id = (task_id or "").strip()
        if not task_id:
            active_records = self.task_manager.list_generation_tasks(
                unified_msg_origin=event.unified_msg_origin,
                include_finished=False,
                limit=5,
            )
            if active_records:
                yield event.plain_result(
                    "❌ 请提供要取消的任务ID\n" + self.format_task_list(active_records)
                )
            else:
                yield event.plain_result("📭 当前没有可取消的生图任务")
            return

        record = self.resolve_active_task_reference(event.unified_msg_origin, task_id)
        if not record:
            yield event.plain_result(f"❌ 正在进行的任务不存在: {task_id}")
            return

        _, message = self.task_manager.cancel_generation_task(
            record.task_id,
            unified_msg_origin=event.unified_msg_origin,
        )
        logger.debug(
            f"{log_prefix('Task', record.task_id)} 用户请求取消任务: "
            f"用户={mask_sensitive(event.unified_msg_origin)}，结果={safe_log_text(message)}"
        )
        yield event.plain_result(message)

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """处理生图指令。"""
        user_id = event.unified_msg_origin
        is_usage_limit_admin = self.is_usage_limit_admin(event)

        user_input = (event.message_str or "").strip()
        masked_uid = mask_sensitive(user_id)
        logger.debug(
            f"{LOG} 收到生图指令: 用户={masked_uid}，输入={safe_log_text(user_input)}"
        )

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        raw_prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        if not raw_prompt:
            yield event.plain_result(self.format_image_command_help())
            return

        if not self.generator or not self.generator.adapter:
            logger.warning(f"{LOG} 生图指令失败: 生成器未初始化，用户={masked_uid}")
            yield event.plain_result("❌ 生图生成器未初始化")
            return

        image_count, prompt = self._parse_command_image_count(raw_prompt)

        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution
        (
            prompt,
            aspect_ratio,
            resolution,
            matched_presets,
            matched_personas,
            persona_images,
        ) = self._parse_command_prompt_templates(prompt, aspect_ratio, resolution)
        preset_or_persona, preset_label = self._format_template_summary(
            matched_presets,
            matched_personas,
        )

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        if (
            not self.config_manager.adapter_config
            or not self.config_manager.adapter_config.api_keys
        ):
            logger.warning(f"{LOG} 生图指令失败: 未配置 API Key，用户={masked_uid}")
            yield event.plain_result("❌ 未配置 API Key，无法生成图片")
            return

        check_result = self.usage_manager.check_rate_limit(
            user_id,
            is_admin=is_usage_limit_admin,
            requested_count=image_count,
            update_timestamp=False,
        )
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return

        prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            logger.warning(
                f"{LOG} 提示词审核未通过: 用户={masked_uid}, 原因={safe_log_text(prompt_reason, 160)}"
            )
            yield event.plain_result(f"❌ 提示词审核未通过: {prompt_reason}")
            return

        check_result = self.usage_manager.check_rate_limit(
            user_id,
            is_admin=is_usage_limit_admin,
            requested_count=image_count,
        )
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]
        images_data: list[ImageData] | None = None
        if self.generator.adapter.get_capabilities() & ImageCapability.IMAGE_TO_IMAGE:
            try:
                images_data = await collect_command_reference_images(
                    self.image_processor,
                    event,
                    persona_images,
                    task_id=task_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"{log_prefix('Task', task_id)} 参考图准备失败: {safe_log_text(exc, 200)}",
                    exc_info=True,
                )
                images_data = []

        reference_image_count = len(images_data or [])

        msg = self.format_start_task_message(
            prompt=prompt,
            reference_image_count=reference_image_count,
            image_count=image_count,
            preset=preset_or_persona,
            preset_label=preset_label,
            presets=matched_presets,
            personas=matched_personas,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
        )
        if msg:
            yield event.plain_result(msg)

        self.create_generation_task(
            task_id=task_id,
            source="指令",
            prompt=prompt,
            images_data=images_data,
            unified_msg_origin=event.unified_msg_origin,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            image_count=image_count,
            is_usage_limit_admin=is_usage_limit_admin,
            preset=preset_or_persona,
            preset_label=preset_label,
            presets=matched_presets,
            personas=matched_personas,
        )

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """切换生图模型。"""
        if not self.config_manager.adapter_config:
            yield event.plain_result("❌ 适配器未初始化")
            return

        models = self.config_manager.adapter_config.available_models or []

        if not model_index:
            lines = ["📋 可用模型列表:"]
            current_model_full = f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "供应商名称/模型名称"

                # 更新配置并重新加载
                self.config_manager.save_model_setting(raw_model)
                self.config_manager.reload()

                if self.generator:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )

                yield event.plain_result(f"✅ 模型已切换: {raw_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生图预设。"""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.debug(
            f"{LOG} 收到预设指令: 用户={masked_uid}，输入={safe_log_text(message_str)}"
        )

        parts = message_str.split(maxsplit=1)
        cmd_text = parts[1].strip() if len(parts) > 1 else ""

        if not cmd_text:
            if not self.config_manager.presets and not self.config_manager.personas:
                yield event.plain_result("📋 当前没有预设或人设")
                return
            preset_list = []
            if self.config_manager.presets:
                preset_list.append("📋 预设列表:")
            for idx, (name, prompt) in enumerate(
                self.config_manager.presets.items(), 1
            ):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")

            if self.config_manager.personas:
                if preset_list:
                    preset_list.append("")
                preset_list.append("👤 人设列表:")
                for idx, (name, persona) in enumerate(
                    self.config_manager.personas.items(), 1
                ):
                    display = (
                        persona.prompt[:20] + "..."
                        if len(persona.prompt) > 20
                        else persona.prompt
                    )
                    image_mark = "有参考图" if persona.image else "无参考图"
                    preset_list.append(f"{idx}. {name}: {display} [{image_mark}]")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parts = cmd_text[3:].split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.config_manager.save_preset(name.strip(), prompt.strip())
                yield event.plain_result(f"✅ 预设已添加: {name.strip()}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")
        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")
