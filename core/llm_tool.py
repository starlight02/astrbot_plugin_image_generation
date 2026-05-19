"""
LLM 可调用的图像生成工具模块
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .types import ImageCapability

if TYPE_CHECKING:
    pass


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的图像生成工具。"""

    name: str = "generate_image"
    description: str = "使用生图模型生成或修改图片"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词(要将用户的意图原样传达给模型)。如果设置了 persona，可只填写额外提示词或留空。",
                },
                "persona": {
                    "type": "string",
                    "description": "可选。使用已配置的人设名称，会将人设描述拼接到提示词前，并在支持图生图时加入人设参考图。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。如果不确定，请使用'自动'。",
                    "enum": [
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
                    ],
                    "default": "自动",
                },
                "resolution": {
                    "type": "string",
                    "description": "图片质量/分辨率。默认使用 '1K'。",
                    "enum": ["1K", "2K", "4K"],
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "当需要使用某人的头像时使用。'self'表示机器人，'sender'表示发送者，也可以直接使用ID做参数。",
                    "items": {"type": "string"},
                },
            },
            "required": [],
        }
    )

    # 使用 Any 避免 Pydantic 循环引用问题
    # 实际类型为 ImageGenerationPlugin，在 TYPE_CHECKING 中定义
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行工具调用。"""
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        # 获取提示词和人设
        prompt = str(kwargs.get("prompt", "") or "").strip()
        persona_name = str(kwargs.get("persona", "") or "").strip()
        matched_persona = None
        persona_image = ""
        if persona_name:
            matched_persona = plugin._find_named_entry(
                plugin.config_manager.personas,
                persona_name,
            )
            if not matched_persona:
                return f"❌ 人设不存在: {persona_name}"
            persona = plugin.config_manager.personas[matched_persona]
            prompt = f"{persona.prompt} {prompt}".strip()
            persona_image = persona.image

        if not prompt:
            return "❌ 请提供图片生成的提示词或人设"

        # 获取事件上下文
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning(
                f"[ImageGen] 工具调用上下文缺少事件。上下文类型: {type(context)}"
            )
            return "❌ 无法获取当前消息上下文"

        # 检查频率限制和每日限制
        check_result = plugin.usage_manager.check_rate_limit(event.unified_msg_origin)
        if isinstance(check_result, str):
            if check_result:
                logger.warning(
                    f"[ImageGen] 工具调用触发限制: {check_result} (用户: {event.unified_msg_origin})"
                )
            return check_result

        if (
            not plugin.config_manager.adapter_config
            or not plugin.config_manager.adapter_config.api_keys
        ):
            logger.warning(
                f"[ImageGen] 工具调用失败: 未配置 API Key (用户: {event.unified_msg_origin})"
            )
            return "❌ 未配置 API Key，无法生成图片"

        prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            return f"❌ 提示词审核未通过: {prompt_reason}"

        # 工具调用同样支持获取上下文参考图（消息/引用/头像）
        images_data = []
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )

        try:
            if capabilities & ImageCapability.IMAGE_TO_IMAGE:
                avatar_user_ids: set[str] = set()
                if persona_image:
                    if persona_image_data := await plugin.image_processor.download_image(
                        persona_image
                    ):
                        images_data.append(persona_image_data)
                    else:
                        logger.warning(
                            f"[ImageGen] 人设参考图获取失败: {matched_persona}"
                        )
                images_data.extend(
                    await plugin.image_processor.fetch_images_from_event(
                        event,
                        avatar_user_ids=avatar_user_ids,
                    )
                )

                # 处理头像引用参数
                avatar_refs = kwargs.get("avatar_references", [])
                if avatar_refs and isinstance(avatar_refs, list):
                    for ref in avatar_refs:
                        if not isinstance(ref, str):
                            continue
                        ref = ref.strip().lower()
                        user_id = None
                        if ref == "self":
                            user_id = str(event.get_self_id())
                        elif ref == "sender":
                            user_id = str(
                                event.get_sender_id() or event.unified_msg_origin
                            )
                        else:
                            # 简单的 QQ 号校验（可选）
                            if ref.isdigit():
                                user_id = ref

                        if user_id and user_id not in avatar_user_ids:
                            avatar_user_ids.add(user_id)
                            avatar_data = await plugin.image_processor.get_avatar(
                                user_id
                            )
                            if avatar_data:
                                images_data.append((avatar_data, "image/jpeg"))
                                logger.info(
                                    f"[ImageGen] 已添加 {user_id} 的头像作为参考图"
                                )
        except Exception as e:
            logger.error(f"[ImageGen] 处理参考图失败: {e}", exc_info=True)
            # 参考图处理失败不影响文生图流程，记录日志继续执行

        # 生成任务 ID
        task_id = hashlib.md5(
            f"{time.time()}{event.unified_msg_origin}".encode()
        ).hexdigest()[:8]
        aspect_ratio = (
            kwargs.get("aspect_ratio") or plugin.config_manager.default_aspect_ratio
        )
        resolution = (
            kwargs.get("resolution") or plugin.config_manager.default_resolution
        )

        # 创建后台任务进行生图
        plugin.create_background_task(
            plugin._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )

        return plugin.format_start_task_message(
            prompt=prompt,
            reference_image_count=len(images_data),
            preset=matched_persona,
            preset_label="人设" if matched_persona else "预设",
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            task_id=task_id,
        )


def adjust_tool_parameters(
    tool: ImageGenerationTool, capabilities: ImageCapability
) -> None:
    """根据适配器能力动态调整工具参数。"""
    props = tool.parameters["properties"]

    if not (capabilities & ImageCapability.ASPECT_RATIO):
        if "aspect_ratio" in props:
            del props["aspect_ratio"]
            logger.debug("[ImageGen] 适配器不支持宽高比，已从工具参数中移除")

    if not (capabilities & ImageCapability.RESOLUTION):
        if "resolution" in props:
            del props["resolution"]
            logger.debug("[ImageGen] 适配器不支持分辨率，已从工具参数中移除")

    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if "avatar_references" in props:
            del props["avatar_references"]
            logger.debug("[ImageGen] 适配器不支持参考图，已从工具参数中移除头像引用")
