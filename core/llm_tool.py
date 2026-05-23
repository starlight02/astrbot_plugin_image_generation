"""
LLM 可调用的图像生成工具模块
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .logging_utils import log_prefix, mask_sensitive, safe_log_text, safe_log_url
from .types import ImageCapability


ASPECT_RATIO_OPTIONS = [
    "自动",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]
RESOLUTION_OPTIONS = ["1K", "2K", "4K"]
LOG = log_prefix("LLMTool")


def _extract_event(context: ContextWrapper[AstrAgentContext] | dict[str, Any]) -> Any:
    """Extract AstrBot message event from different tool context wrappers."""
    wrapped_context = getattr(context, "context", None)
    if event := getattr(wrapped_context, "event", None):
        return event
    if event := getattr(context, "event", None):
        return event
    if isinstance(context, dict):
        return context.get("event")
    return None


def _normalize_string_items(raw: Any) -> list[str]:
    """Normalize one or many string-like values from tool arguments."""
    if raw is None:
        return []
    if isinstance(raw, str):
        item = raw.strip()
        return [item] if item else []
    if isinstance(raw, dict):
        for key in ("url", "path", "file", "name"):
            if items := _normalize_string_items(raw.get(key)):
                return items
        return []
    if isinstance(raw, Iterable):
        items: list[str] = []
        for value in raw:
            items.extend(_normalize_string_items(value))
        return items
    item = str(raw).strip()
    return [item] if item else []


def _normalize_preset_query_category(category: str) -> str:
    """Normalize preset query category aliases."""
    normalized = category.strip().lower()
    if normalized in {
        "人设",
        "persona",
        "personas",
        "list_persona",
        "list_personas",
        "get_persona",
        "persona_list",
    }:
        return "persona"
    return "preset"


def _normalize_preset_edit_action(action: str) -> str:
    """Normalize preset edit action aliases."""
    normalized = action.strip().lower()
    if normalized in {"添加", "新增", "保存", "save", "create", "add", "add_preset"}:
        return "create_preset"
    if normalized in {
        "删除",
        "移除",
        "remove",
        "del",
        "delete",
        "delete_preset",
    }:
        return "delete_preset"
    return normalized


def _format_preset_detail(name: str, content: Any) -> str:
    """Format one preset's full content for query results."""
    content_text = str(content or "").strip()
    lines = [f"📋 预设详情: {name}"]
    if content_text.startswith("{"):
        try:
            preset_data = json.loads(content_text)
        except json.JSONDecodeError:
            lines.append("格式: 高级 JSON（解析失败，将按原文展示）")
            lines.append(f"内容: {content_text}")
            return "\n".join(lines)

        if isinstance(preset_data, dict):
            lines.append("格式: 高级 JSON")
            if prompt := str(preset_data.get("prompt", "") or "").strip():
                lines.append(f"提示词: {prompt}")
            if aspect_ratio := str(preset_data.get("aspect_ratio", "") or "").strip():
                lines.append(f"宽高比: {aspect_ratio}")
            if resolution := str(preset_data.get("resolution", "") or "").strip():
                lines.append(f"分辨率: {resolution}")
            if description := str(preset_data.get("description", "") or "").strip():
                lines.append(f"描述: {description}")
            lines.append(f"原始内容: {content_text}")
            return "\n".join(lines)

    lines.append("格式: 简单提示词")
    lines.append(f"内容: {content_text}")
    return "\n".join(lines)


def _format_persona_detail(name: str, persona: Any) -> str:
    """Format one persona's full content for query results."""
    lines = [f"👤 人设详情: {name}"]
    lines.append(f"提示词: {persona.prompt}")
    lines.append(f"参考图: {persona.image or '无'}")
    return "\n".join(lines)


def _validate_preset_content(content: str) -> str | None:
    """Validate preset content when it is written by an LLM tool."""
    if not content.startswith("{"):
        return None

    try:
        preset_data = json.loads(content)
    except json.JSONDecodeError as exc:
        return f"高级 JSON 预设格式错误: {exc}"

    if not isinstance(preset_data, dict):
        return "高级 JSON 预设必须是对象"
    if not str(preset_data.get("prompt", "") or "").strip():
        return "高级 JSON 预设必须包含非空 prompt 字段"

    aspect_ratio = str(preset_data.get("aspect_ratio", "") or "").strip()
    if aspect_ratio and aspect_ratio not in ASPECT_RATIO_OPTIONS:
        return f"高级 JSON 预设的 aspect_ratio 不支持: {aspect_ratio}"

    resolution = str(preset_data.get("resolution", "") or "").strip()
    if resolution and resolution not in RESOLUTION_OPTIONS:
        return f"高级 JSON 预设的 resolution 不支持: {resolution}"

    return None


def _parse_preset(
    plugin: Any,
    preset_name: str,
    prompt: str,
    aspect_ratio: Any,
    resolution: Any,
) -> tuple[str, str, str, str | None, str | None]:
    """Apply preset prompt and optional generation overrides."""
    if not preset_name:
        return str(prompt).strip(), str(aspect_ratio), str(resolution), None, None

    matched_preset = plugin._find_named_entry(
        plugin.config_manager.presets, preset_name
    )
    if not matched_preset:
        return (
            "",
            str(aspect_ratio),
            str(resolution),
            None,
            f"❌ 预设不存在: {preset_name}",
        )

    preset_content = plugin.config_manager.presets[matched_preset]
    preset_prompt = str(preset_content or "").strip()
    if preset_prompt.startswith("{"):
        try:
            preset_data = json.loads(preset_prompt)
            if isinstance(preset_data, dict):
                preset_prompt = str(preset_data.get("prompt", "")).strip()
                aspect_ratio = str(preset_data.get("aspect_ratio") or aspect_ratio)
                resolution = str(preset_data.get("resolution") or resolution)
        except json.JSONDecodeError:
            pass

    final_prompt = f"{preset_prompt} {prompt}".strip()
    return final_prompt, str(aspect_ratio), str(resolution), matched_preset, None


def _parse_persona(
    plugin: Any,
    persona_name: str,
    prompt: str,
) -> tuple[str, str, str | None, str | None]:
    """Apply persona prompt and reference image."""
    if not persona_name:
        return str(prompt).strip(), "", None, None

    matched_persona = plugin._find_named_entry(
        plugin.config_manager.personas,
        persona_name,
    )
    if not matched_persona:
        return "", "", None, f"❌ 人设不存在: {persona_name}"

    persona = plugin.config_manager.personas[matched_persona]
    final_prompt = f"{persona.prompt} {prompt}".strip()
    return final_prompt, persona.image, matched_persona, None


def _resolve_avatar_user_id(event: Any, ref: str) -> str | None:
    """Resolve an avatar reference into a platform user id."""
    normalized = ref.strip().lower()
    if not normalized:
        return None
    if normalized == "self" and hasattr(event, "get_self_id"):
        return str(event.get_self_id())
    if normalized == "sender" and hasattr(event, "get_sender_id"):
        return str(event.get_sender_id() or event.unified_msg_origin)

    cleaned = normalized.removeprefix("qq:").removeprefix("@").strip()
    if cleaned.isdigit():
        return cleaned
    return None


async def _download_reference_images(
    plugin: Any,
    references: Any,
    *,
    reference_label: str,
    task_id: str | None = None,
) -> list[tuple[bytes, str]]:
    """Download explicit reference images from URLs or local file paths."""
    images_data: list[tuple[bytes, str]] = []
    task_log = log_prefix("LLMTool", task_id) if task_id else LOG
    for reference in _normalize_string_items(references):
        if image_data := await plugin.image_processor.download_image(reference):
            images_data.append(image_data)
        else:
            logger.warning(
                f"{task_log} {reference_label}参考图获取失败: {safe_log_url(reference)}"
            )
    return images_data


async def _collect_reference_images(
    plugin: Any,
    event: Any,
    *,
    capabilities: ImageCapability,
    reference_images: Any = None,
    avatar_references: Any = None,
    persona_image: str = "",
    persona_name: str | None = None,
    task_id: str | None = None,
) -> list[tuple[bytes, str]]:
    """Collect explicit, persona, avatar, and message-context reference images."""
    task_log = log_prefix("LLMTool", task_id) if task_id else LOG
    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if reference_images or avatar_references or persona_image:
            logger.warning(f"{task_log} 当前适配器不支持参考图，已忽略工具参考图参数")
        return []

    images_data: list[tuple[bytes, str]] = []
    avatar_user_ids: set[str] = set()

    if persona_image:
        if persona_image_data := await plugin.image_processor.download_image(
            persona_image
        ):
            images_data.append(persona_image_data)
        else:
            logger.warning(
                f"{task_log} 人设参考图获取失败: {safe_log_text(persona_name)}"
            )

    images_data.extend(
        await _download_reference_images(
            plugin,
            reference_images,
            reference_label="显式",
            task_id=task_id,
        )
    )

    for ref in _normalize_string_items(avatar_references):
        user_id = _resolve_avatar_user_id(event, ref)
        if not user_id or user_id in avatar_user_ids:
            continue
        avatar_user_ids.add(user_id)
        if avatar_data := await plugin.image_processor.get_avatar(user_id):
            images_data.append((avatar_data, "image/jpeg"))
            logger.info(f"{task_log} 已添加 {mask_sensitive(user_id)} 的头像作为参考图")

    images_data.extend(
        await plugin.image_processor.fetch_images_from_event(
            event,
            avatar_user_ids=avatar_user_ids,
        )
    )
    return images_data


async def _start_generation_task(
    plugin: Any,
    event: Any,
    *,
    prompt: str,
    aspect_ratio: str,
    resolution: str,
    reference_images: Any = None,
    avatar_references: Any = None,
    persona_image: str = "",
    preset_or_persona: str | None = None,
    persona_name: str | None = None,
    preset_label: str = "预设",
) -> ToolExecResult:
    """Validate request, collect references, and schedule image generation."""
    if not plugin.generator or not plugin.generator.adapter:
        return "❌ 生图生成器未初始化"

    is_usage_limit_admin = plugin.is_usage_limit_admin(event)
    check_result = plugin.usage_manager.check_rate_limit(
        event.unified_msg_origin,
        is_admin=is_usage_limit_admin,
    )
    if isinstance(check_result, str):
        if check_result:
            masked_uid = mask_sensitive(event.unified_msg_origin)
            logger.warning(
                f"{LOG} 工具调用触发限制: {check_result} (用户: {masked_uid})"
            )
        return check_result

    if (
        not plugin.config_manager.adapter_config
        or not plugin.config_manager.adapter_config.api_keys
    ):
        masked_uid = mask_sensitive(event.unified_msg_origin)
        logger.warning(f"{LOG} 工具调用失败: 未配置 API Key (用户: {masked_uid})")
        return "❌ 未配置 API Key，无法生成图片"

    prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
        prompt,
        event.unified_msg_origin,
    )
    if not prompt_allowed:
        return f"❌ 提示词审核未通过: {prompt_reason}"

    task_id = hashlib.md5(
        f"{time.time()}{event.unified_msg_origin}".encode()
    ).hexdigest()[:8]

    try:
        images_data = await _collect_reference_images(
            plugin,
            event,
            capabilities=plugin.generator.adapter.get_capabilities(),
            reference_images=reference_images,
            avatar_references=avatar_references,
            persona_image=persona_image,
            persona_name=persona_name,
            task_id=task_id,
        )
    except Exception as exc:
        logger.error(
            f"{log_prefix('LLMTool', task_id)} 处理参考图失败: {exc}",
            exc_info=True,
        )
        images_data = []

    plugin.create_background_task(
        plugin._generate_and_send_image_async(
            prompt=prompt,
            images_data=images_data or None,
            unified_msg_origin=event.unified_msg_origin,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
            is_usage_limit_admin=is_usage_limit_admin,
        )
    )

    return plugin.format_start_task_message(
        prompt=prompt,
        reference_image_count=len(images_data),
        preset=preset_or_persona,
        preset_label=preset_label,
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        task_id=task_id,
    )


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的统一生图工具。"""

    name: str = "generate_image"
    description: str = (
        "使用生图模型生成或修改图片；支持普通生图、预设、自拍、人像、头像和人设照片。"
        "当用户要求自拍/头像/人像/某个人设或角色出镜时，应优先填写 persona。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词。设置 preset 或 persona 时可填写额外提示词，也可留空。",
                },
                "preset": {
                    "type": "string",
                    "description": "可选。使用已配置的预设名称，会将预设提示词作为基础提示词，并可继承预设中的宽高比和分辨率。",
                },
                "persona": {
                    "type": "string",
                    "description": "可选。使用已配置的人设名称。当用户要求自拍、头像、人像、人设照片或某角色出镜时优先填写；会拼接人设描述，并在支持图生图时加入人设参考图。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。如果不确定，请使用'自动'；填写 persona 且未指定时默认按自拍/人像使用 9:16。",
                    "enum": ASPECT_RATIO_OPTIONS,
                    "default": "自动",
                },
                "resolution": {
                    "type": "string",
                    "description": "图片质量/分辨率。默认使用 '1K'。",
                    "enum": RESOLUTION_OPTIONS,
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "可选。需要使用头像作为参考图时填写，'self' 表示机器人，'sender' 表示发送者，也可填写 QQ 号/用户 ID。",
                    "items": {"type": "string"},
                },
                "reference_images": {
                    "type": "array",
                    "description": "可选。参考图列表，支持 Linux/Windows 绝对路径、file:// 文件 URL 或 http(s) 网络图片 URL。仅支持图生图的模型会使用。",
                    "items": {"type": "string"},
                },
            },
            "required": [],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行通用生图工具调用。"""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        prompt = str(kwargs.get("prompt", "") or "").strip()
        raw_aspect_ratio = kwargs.get("aspect_ratio")
        aspect_ratio = raw_aspect_ratio or plugin.config_manager.default_aspect_ratio
        resolution = (
            kwargs.get("resolution") or plugin.config_manager.default_resolution
        )

        prompt, aspect_ratio, resolution, matched_preset, error = _parse_preset(
            plugin,
            str(kwargs.get("preset", "") or "").strip(),
            prompt,
            aspect_ratio,
            resolution,
        )
        if error:
            return error
        prompt, persona_image, matched_persona, error = _parse_persona(
            plugin,
            str(kwargs.get("persona", "") or "").strip(),
            prompt,
        )
        if error:
            return error
        if matched_persona and not raw_aspect_ratio:
            aspect_ratio = "9:16"
        if not prompt:
            return "❌ 请提供图片生成的提示词、预设或人设"

        preset_or_persona = matched_preset
        preset_label = "预设"
        if matched_preset and matched_persona:
            preset_or_persona = f"{matched_preset} / {matched_persona}"
            preset_label = "预设/人设"
        elif matched_persona:
            preset_or_persona = matched_persona
            preset_label = "人设"

        event = _extract_event(context)
        if not event:
            logger.warning(f"{LOG} 工具调用上下文缺少事件。上下文类型: {type(context)}")
            return "❌ 无法获取当前消息上下文"

        return await _start_generation_task(
            plugin,
            event,
            prompt=prompt,
            aspect_ratio=str(aspect_ratio),
            resolution=str(resolution),
            reference_images=kwargs.get("reference_images"),
            avatar_references=kwargs.get("avatar_references"),
            persona_image=persona_image,
            preset_or_persona=preset_or_persona,
            persona_name=matched_persona,
            preset_label=preset_label,
        )


@pydantic_dataclass
class PresetQueryTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的预设/人设查询工具。"""

    name: str = "query_image_presets"
    description: str = (
        "查询生图预设或人设。默认只返回名称列表；当需要了解某个预设或人设的具体内容时，"
        "填写 name 查看详情。查询到的预设名称可用于 generate_image 的 preset 参数，人设名称可用于 persona 参数。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "查询类别。preset 查询生图预设，persona 查询人设。默认 preset。",
                    "enum": ["preset", "persona"],
                    "default": "preset",
                },
                "name": {
                    "type": "string",
                    "description": "可选。填写后查看指定预设或人设的具体内容；留空时只返回名称列表。",
                },
            },
            "required": [],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行预设/人设查询工具调用。"""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = _extract_event(context)
        if (
            event
            and not plugin.usage_manager.is_limit_exempt(
                event.unified_msg_origin,
                is_admin=plugin.is_usage_limit_admin(event),
            )
            and plugin.usage_manager.is_session_blocked(event.unified_msg_origin)
        ):
            return plugin.config_manager.usage_settings.blacklist_block_message

        category = _normalize_preset_query_category(
            str(kwargs.get("category", "preset") or "preset")
        )
        name = str(kwargs.get("name", "") or "").strip()

        if category == "persona":
            if name:
                matched_name = plugin._find_named_entry(
                    plugin.config_manager.personas, name
                )
                if not matched_name:
                    return f"❌ 人设不存在: {name}"
                return _format_persona_detail(
                    matched_name, plugin.config_manager.personas[matched_name]
                )

            if not plugin.config_manager.personas:
                return "👤 当前没有人设"
            lines = ["👤 人设列表:"]
            for idx, (persona_name, persona) in enumerate(
                plugin.config_manager.personas.items(),
                1,
            ):
                image_mark = "有参考图" if persona.image else "无参考图"
                lines.append(f"{idx}. {persona_name} [{image_mark}]")
            return "\n".join(lines)

        if name:
            matched_name = plugin._find_named_entry(plugin.config_manager.presets, name)
            if not matched_name:
                return f"❌ 预设不存在: {name}"
            return _format_preset_detail(
                matched_name, plugin.config_manager.presets[matched_name]
            )

        if not plugin.config_manager.presets:
            return "📋 当前没有预设"
        lines = ["📋 预设列表:"]
        for idx, preset_name in enumerate(plugin.config_manager.presets, 1):
            lines.append(f"{idx}. {preset_name}")
        return "\n".join(lines)


@pydantic_dataclass
class PresetEditTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的预设编辑工具。"""

    name: str = "edit_image_presets"
    description: str = (
        "创建或删除生图预设。支持两种预设格式："
        "1) 简单格式：name + prompt，prompt 直接写普通提示词；"
        "2) 高级 JSON 格式：prompt 写 JSON 字符串，如 "
        '{"prompt":"提示词","aspect_ratio":"16:9","resolution":"2K","description":"描述"}。'
        "只编辑预设，不编辑人设。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "要执行的操作。create_preset 创建或覆盖预设，delete_preset 删除预设。",
                    "enum": ["create_preset", "delete_preset"],
                },
                "name": {
                    "type": "string",
                    "description": "预设名称。创建和删除均必填。",
                },
                "prompt": {
                    "type": "string",
                    "description": '创建预设时必填。可填写普通提示词，或填写高级 JSON 字符串：{"prompt":"提示词","aspect_ratio":"16:9","resolution":"2K","description":"描述"}。',
                },
            },
            "required": ["action", "name"],
        }
    )

    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行预设编辑工具调用。"""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        event = _extract_event(context)
        if (
            event
            and not plugin.usage_manager.is_limit_exempt(
                event.unified_msg_origin,
                is_admin=plugin.is_usage_limit_admin(event),
            )
            and plugin.usage_manager.is_session_blocked(event.unified_msg_origin)
        ):
            return plugin.config_manager.usage_settings.blacklist_block_message

        action = _normalize_preset_edit_action(
            str(kwargs.get("action", "create_preset") or "")
        )
        name = str(kwargs.get("name", "") or "").strip()
        if not name:
            return "❌ 请提供预设名称"

        if action == "create_preset":
            prompt = str(kwargs.get("prompt", "") or "").strip()
            if not prompt:
                return "❌ 请提供预设内容"
            if error := _validate_preset_content(prompt):
                return f"❌ {error}"
            plugin.config_manager.save_preset(name, prompt)
            return f"✅ 预设已保存: {name}"

        if action == "delete_preset":
            matched_name = plugin._find_named_entry(plugin.config_manager.presets, name)
            if not matched_name:
                return f"❌ 预设不存在: {name}"
            plugin.config_manager.delete_preset(matched_name)
            return f"✅ 预设已删除: {matched_name}"

        return "❌ 不支持的操作，请使用 create_preset 或 delete_preset"


def adjust_tool_parameters(
    tool: FunctionTool[AstrAgentContext], capabilities: ImageCapability
) -> None:
    """根据适配器能力动态调整工具参数。"""
    props = tool.parameters.get("properties", {})

    if not (capabilities & ImageCapability.ASPECT_RATIO):
        if "aspect_ratio" in props:
            del props["aspect_ratio"]
            logger.debug(f"{LOG} 适配器不支持宽高比，已从工具参数中移除")

    if not (capabilities & ImageCapability.RESOLUTION):
        if "resolution" in props:
            del props["resolution"]
            logger.debug(f"{LOG} 适配器不支持分辨率，已从工具参数中移除")

    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        for key in ("avatar_references", "reference_images"):
            if key in props:
                del props[key]
        logger.debug(f"{LOG} 适配器不支持参考图，已从工具参数中移除参考图相关参数")
