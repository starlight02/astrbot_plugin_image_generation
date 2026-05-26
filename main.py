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
    ConfigManager,
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_USAGE,
)
from .core.generator import ImageGenerator
from .core.image_processor import ImageProcessor
from .core.llm_tool import (
    ImageGenerationTool,
    PresetEditTool,
    PresetQueryTool,
    adjust_tool_parameters,
)
from .core.constants import UNSPECIFIED_OPTION
from .core.logging_utils import log_prefix, mask_sensitive, safe_log_text
from .core.safety_auditor import SafetyAuditor
from .core.task_manager import TaskManager
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
        self.image_temp_dir = (
            Path(get_astrbot_temp_path()) / "astrbot_plugin_image_generation"
        )
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
        )

        # 初始化任务管理器
        self.task_manager = TaskManager()

        # 初始化安全审核器
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self.semaphore: asyncio.Semaphore | None = None

    # ---------------------- 生命周期 ----------------------

    async def initialize(self):
        """插件加载时调用"""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.semaphore = asyncio.Semaphore(self.config_manager.max_concurrent_tasks)
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
            props["persona"]["enum"] = list(self.config_manager.personas)

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro)

    def is_usage_limit_admin(self, event: AstrMessageEvent) -> bool:
        """Return whether an event sender is an AstrBot admin for usage limits."""
        try:
            return bool(event.is_admin())
        except Exception as exc:
            logger.debug(f"{LOG} 获取管理员状态失败: {exc}")
            return False

    def _find_named_entry(self, entries: dict[str, Any], token: str) -> str | None:
        """Find an entry by exact or case-insensitive name."""
        if token in entries:
            return token
        lowered_token = token.lower()
        for name in entries:
            if name.lower() == lowered_token:
                return name
        return None

    def format_start_task_message(
        self,
        *,
        prompt: str,
        reference_image_count: int,
        preset: str | None,
        preset_label: str = "预设",
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
            prompt=prompt,
            preset=preset or "",
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
            model=model,
            mode="图生图" if reference_image_count else "文生图",
            reference_images_block=(
                f"[{reference_image_count}张参考图]" if reference_image_count else ""
            ),
            preset_block=f"[{preset_label}: {preset}]" if preset else "",
        )

        try:
            return template.format_map(values)
        except Exception as exc:
            logger.warning(f"{LOG} 开始任务提示模板格式化失败: {exc}")
            return "已开始生图任务{reference_images_block}{preset_block}".format_map(
                values
            )

    # ---------------------- 核心生图逻辑 ----------------------

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
        is_usage_limit_admin: bool = False,
    ) -> None:
        """异步生成图片并发送。"""
        if not self.generator or not self.generator.adapter:
            return

        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        capabilities = self.generator.adapter.get_capabilities()

        # 检查并清理不支持的参数
        task_log = log_prefix("Task", task_id)
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"{task_log} 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
            )
            images_data = None

        if (
            not (capabilities & ImageCapability.ASPECT_RATIO)
            and aspect_ratio != UNSPECIFIED_OPTION
        ):
            logger.info(
                f"{task_log} 当前适配器不支持指定比例，已忽略参数: {safe_log_text(aspect_ratio)}"
            )
            aspect_ratio = UNSPECIFIED_OPTION

        if (
            not (capabilities & ImageCapability.RESOLUTION)
            and resolution != UNSPECIFIED_OPTION
        ):
            logger.info(
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
            for data, mime in images_data:
                images.append(ImageData(data=data, mime_type=mime))

        # 使用信号量控制并发
        if self.semaphore is None:
            await self._do_generate_and_send(
                prompt,
                unified_msg_origin,
                images,
                final_ar,
                final_res,
                task_id,
                is_usage_limit_admin,
            )
            return

        async with self.semaphore:
            await self._do_generate_and_send(
                prompt,
                unified_msg_origin,
                images,
                final_ar,
                final_res,
                task_id,
                is_usage_limit_admin,
            )

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
        is_usage_limit_admin: bool,
    ) -> None:
        """执行生成逻辑并发送结果。"""
        start_time = time.time()
        task_log = log_prefix("Task", task_id)
        if not self.generator:
            logger.warning(f"{task_log} 生成器未初始化，跳过生成请求")
            return
        result = await self.generator.generate(
            GenerationRequest(
                prompt=prompt,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )
        end_time = time.time()
        duration = end_time - start_time

        if result.error:
            logger.error(
                f"{task_log} 生成失败，耗时: {duration:.2f}s, 错误: {safe_log_text(result.error, 200)}"
            )
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 生成失败: {result.error}"),
            )
            return

        logger.info(
            f"{task_log} 生成成功，耗时: {duration:.2f}s, 图片数量: {len(result.images) if result.images else 0}"
        )

        if not result.images:
            return

        generated_file_paths: list[str] = []
        for img_bytes in result.images:
            file_path = self.image_processor.save_generated_image(task_id, img_bytes)
            if file_path:
                generated_file_paths.append(file_path)

        if not generated_file_paths:
            logger.warning(f"{task_log} 未能保存任何生成图片")
            return

        # 生图后图片审核
        image_allowed, image_reason = await self.safety_auditor.audit_generated_images(
            prompt=prompt,
            image_paths=generated_file_paths,
            unified_msg_origin=unified_msg_origin,
        )
        if not image_allowed:
            logger.warning(
                f"{task_log} 图片审核未通过: {safe_log_text(image_reason, 200)}"
            )
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 图片内容审核未通过: {image_reason}"),
            )
            return

        # 记录使用次数
        self.usage_manager.record_usage(
            unified_msg_origin,
            is_admin=is_usage_limit_admin,
        )

        chain = MessageChain()
        for file_path in generated_file_paths:
            chain.file_image(file_path)

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
            chain.message("\n" + "\n".join(info_parts))

        await self.context.send_message(unified_msg_origin, chain)

    # ---------------------- 指令处理 ----------------------

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """处理生图指令。"""
        user_id = event.unified_msg_origin
        is_usage_limit_admin = self.is_usage_limit_admin(event)

        # 检查频率限制和每日限制
        check_result = self.usage_manager.check_rate_limit(
            user_id,
            is_admin=is_usage_limit_admin,
        )
        if isinstance(check_result, str):
            if check_result:
                yield event.plain_result(check_result)
            return

        masked_uid = mask_sensitive(user_id)

        user_input = (event.message_str or "").strip()
        logger.info(
            f"{LOG} 收到生图指令 - 用户: {masked_uid}, 输入摘要: {safe_log_text(user_input)}"
        )

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution

        # 检查是否命中预设
        matched_preset = None
        matched_persona = None
        persona_image = ""
        extra_content = ""
        if prompt:
            parts = prompt.split(maxsplit=1)
            first_token = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            matched_preset = self._find_named_entry(
                self.config_manager.presets, first_token
            )
            if matched_preset:
                extra_content = rest
            else:
                matched_persona = self._find_named_entry(
                    self.config_manager.personas, first_token
                )
                if matched_persona:
                    extra_content = rest

        if matched_preset:
            logger.info(f"{LOG} 命中预设: {safe_log_text(matched_preset)}")
            preset_content = self.config_manager.presets[matched_preset]
            try:
                # 预设支持 JSON 格式配置高级参数
                if isinstance(
                    preset_content, str
                ) and preset_content.strip().startswith("{"):
                    preset_data = json.loads(preset_content)
                    if isinstance(preset_data, dict):
                        prompt = preset_data.get("prompt", "")
                        aspect_ratio = preset_data.get("aspect_ratio", aspect_ratio)
                        resolution = preset_data.get("resolution", resolution)
                    else:
                        prompt = preset_content
                else:
                    prompt = preset_content
            except json.JSONDecodeError:
                prompt = preset_content

            if extra_content:
                prompt = f"{prompt} {extra_content}"

        if matched_persona:
            logger.info(f"{LOG} 命中人设: {safe_log_text(matched_persona)}")
            persona = self.config_manager.personas[matched_persona]
            prompt = persona.prompt
            persona_image = persona.image
            if extra_content:
                prompt = f"{prompt} {extra_content}".strip()

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            yield event.plain_result(f"❌ 提示词审核未通过: {prompt_reason}")
            return

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]
        task_log = log_prefix("Task", task_id)

        # 获取参考图
        images_data = None
        if (
            self.generator
            and self.generator.adapter
            and (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            )
        ):
            images_data = []
            if persona_image:
                if persona_image_data := await self.image_processor.download_image(
                    persona_image
                ):
                    images_data.append(persona_image_data)
                else:
                    logger.warning(
                        f"{task_log} 人设参考图获取失败: {safe_log_text(matched_persona)}"
                    )
            images_data.extend(
                await self.image_processor.fetch_images_from_event(event)
            )

        msg = self.format_start_task_message(
            prompt=prompt,
            reference_image_count=len(images_data or []),
            preset=matched_preset or matched_persona,
            preset_label="人设" if matched_persona else "预设",
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
        )
        if msg:
            yield event.plain_result(msg)

        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
                is_usage_limit_admin=is_usage_limit_admin,
            )
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
        logger.info(
            f"{LOG} 收到预设指令 - 用户: {masked_uid}, 内容摘要: {safe_log_text(message_str)}"
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
