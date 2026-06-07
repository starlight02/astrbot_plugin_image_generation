"""User-facing result formatting helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astrbot.api import logger

from .config_manager import (
    RESULT_INFO_COUNT,
    RESULT_INFO_DURATION,
    RESULT_INFO_MODEL,
    RESULT_INFO_TASK_ID,
    RESULT_INFO_USAGE,
)
from .logging_utils import log_prefix

if TYPE_CHECKING:
    from .config_manager import ConfigManager
    from .task_manager import GenerationTaskRecord
    from .usage_manager import UsageManager


LOG = log_prefix("Formatter")


class SafeFormatDict(dict[str, str]):
    """Keep unknown template placeholders unchanged."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def format_start_template_values(
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
    config_manager: ConfigManager,
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
    template = config_manager.start_task_message_template
    if not template.strip():
        return ""

    model = ""
    if config_manager.adapter_config:
        model = (
            f"{config_manager.adapter_config.name}/"
            f"{config_manager.adapter_config.model}"
        )

    values = SafeFormatDict(
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
        **format_start_template_values(
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


def format_task_detail(record: GenerationTaskRecord) -> str:
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
                item_line += f"，重试 {item.retry_attempts}/{item.max_retry_attempts}"
            lines.append(item_line)
    return "\n".join(lines)


def format_task_list(records: list[GenerationTaskRecord]) -> str:
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


def format_image_command_help(config_manager: ConfigManager) -> str:
    """Format help text for the image generation command."""
    adapter_config = config_manager.adapter_config
    current_model = (
        f"{adapter_config.name}/{adapter_config.model}" if adapter_config else "未配置"
    )
    lines = [
        "🎨 生图帮助",
        f"当前模型: {current_model}",
        "",
        "指令列表:",
        "/生图 [预设/人设] [提示词] [数量]",
        "/生图模型 - 查看或切换模型",
        "/生图任务 [编号或任务ID] - 查看正在进行的任务",
        "/生图取消 <编号或任务ID> - 取消指定任务",
        "/预设 [添加/删除] - 查看或管理预设/人设",
    ]
    return "\n".join(lines)


def build_result_info_message(
    config_manager: ConfigManager,
    usage_manager: UsageManager,
    *,
    unified_msg_origin: str,
    is_usage_limit_admin: bool,
    duration: float,
    result_count: int,
    task_id: str,
) -> str:
    """Build the optional metadata appended to generated image messages."""
    info_parts: list[str] = []
    if config_manager.should_show_result_info(RESULT_INFO_DURATION):
        info_parts.append(f"📊 耗时: {duration:.2f}s")

    if (
        config_manager.should_show_result_info(RESULT_INFO_MODEL)
        and config_manager.adapter_config
    ):
        info_parts.append(
            f"🤖 模型: {config_manager.adapter_config.name}/{config_manager.adapter_config.model}"
        )

    if config_manager.should_show_result_info(RESULT_INFO_COUNT):
        info_parts.append(f"🖼️ 数量: {result_count}张")

    if config_manager.should_show_result_info(RESULT_INFO_TASK_ID):
        info_parts.append(f"🧾 任务ID: {task_id}")

    if (
        config_manager.should_show_result_info(RESULT_INFO_USAGE)
        and usage_manager.is_daily_limit_enabled()
    ):
        count = usage_manager.get_usage_count(unified_msg_origin)
        daily_limit = (
            "∞"
            if usage_manager.is_limit_exempt(
                unified_msg_origin,
                is_admin=is_usage_limit_admin,
            )
            else str(usage_manager.get_daily_limit())
        )
        info_parts.append(f"📅 今日用量: {count}/{daily_limit}")

    return "\n".join(info_parts)
