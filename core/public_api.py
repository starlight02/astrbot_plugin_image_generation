"""Public API for inter-plugin image generation calls."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from astrbot.api import logger

from .logging_utils import log_prefix, mask_sensitive, safe_log_text
from .reference_collector import (
    collect_reference_images_from_personas,
    deduplicate_reference_images,
    download_reference_images,
)
from .task_manager import GenerationTaskRecord
from .task_id import new_task_id
from .template_utils import (
    find_named_entry,
    format_template_summary,
    normalize_name_items,
    parse_preset_prompt,
)
from .types import ImageCapability, ImageData


LOG = log_prefix("PublicAPI")
DEFAULT_SOURCE = "公共接口"
DEFAULT_WAIT_POLL_INTERVAL_SECONDS = 0.5


class PublicAPIResultCode(str, Enum):
    """Stable result codes returned by the inter-plugin public API."""

    ACCEPTED = "accepted"
    GENERATOR_NOT_INITIALIZED = "generator_not_initialized"
    API_KEY_MISSING = "api_key_missing"
    TEMPLATE_NOT_FOUND = "template_not_found"
    EMPTY_PROMPT = "empty_prompt"
    RATE_LIMITED = "rate_limited"
    PROMPT_BLOCKED = "prompt_blocked"
    CANCEL_REQUESTED = "cancel_requested"
    CANCEL_FAILED = "cancel_failed"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    SUCCEEDED = "succeeded"
    NO_RESULT = "no_result"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"

    @classmethod
    def from_task_status(cls, status: str) -> "PublicAPIResultCode":
        """Map task status values to public API result codes."""
        try:
            return cls(status)
        except ValueError:
            return cls.FAILED


@dataclass(frozen=True)
class ImageGenerationTaskSnapshot:
    """Stable snapshot of one image generation task for external plugins."""

    task_id: str
    status: str
    active: bool
    source: str
    requested_count: int
    result_count: int
    reference_image_count: int
    aspect_ratio: str
    resolution: str
    result_paths: list[str]
    error: str
    message: str
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    duration_seconds: float | None


@dataclass(frozen=True)
class ImageGenerationSubmitResult:
    """Result returned after submitting an image generation task."""

    ok: bool
    code: str
    message: str
    task_id: str | None = None
    error: str = ""


@dataclass(frozen=True)
class ImageGenerationResult:
    """Result returned after waiting for generated image files."""

    ok: bool
    code: str
    message: str
    task_id: str
    paths: list[str]
    error: str = ""


@dataclass(frozen=True)
class ImageGenerationOperationResult:
    """Result returned for task operations such as cancellation."""

    ok: bool
    code: str
    message: str
    task_id: str | None = None
    error: str = ""


class ImageGenerationPublicAPI:
    """Programmatic image generation API exposed to other AstrBot plugins."""

    def __init__(self, plugin: Any) -> None:
        self._plugin = plugin

    async def submit_generation_task(
        self,
        *,
        prompt: str = "",
        unified_msg_origin: str | None = None,
        source: str = DEFAULT_SOURCE,
        image_count: int | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        reference_image_sources: Any = None,
        reference_image_data: list[ImageData | tuple[bytes, str]] | None = None,
        presets: str | list[str] | None = None,
        personas: str | list[str] | None = None,
        is_admin: bool = False,
    ) -> ImageGenerationSubmitResult:
        """Submit a background image generation task and return its task id."""
        plugin = self._plugin
        if not plugin.generator or not plugin.generator.adapter:
            return self._submit_error(
                PublicAPIResultCode.GENERATOR_NOT_INITIALIZED,
                "生图生成器未初始化",
            )

        if not plugin.has_required_api_key():
            return self._submit_error(
                PublicAPIResultCode.API_KEY_MISSING,
                "未配置 API Key，无法生成图片",
            )

        request_source = str(source or DEFAULT_SOURCE).strip() or DEFAULT_SOURCE
        scope = str(unified_msg_origin or "").strip()
        use_usage_scope = bool(scope)
        safe_is_admin = bool(is_admin) if use_usage_scope else False
        task_id = self._new_task_id()

        requested_count = plugin.normalize_image_count(
            image_count
            if image_count is not None
            else plugin.config_manager.default_image_count
        )
        final_aspect_ratio = aspect_ratio or plugin.config_manager.default_aspect_ratio
        final_resolution = resolution or plugin.config_manager.default_resolution

        (
            final_prompt,
            final_aspect_ratio,
            final_resolution,
            matched_presets,
            matched_personas,
            persona_images,
            parse_error,
        ) = self._build_prompt(
            prompt=prompt,
            presets=presets,
            personas=personas,
            aspect_ratio=str(final_aspect_ratio),
            resolution=str(final_resolution),
        )
        if parse_error:
            return self._submit_error(
                PublicAPIResultCode.TEMPLATE_NOT_FOUND,
                parse_error,
            )
        if not final_prompt:
            return self._submit_error(
                PublicAPIResultCode.EMPTY_PROMPT,
                "请提供图片生成提示词、预设或人设",
            )

        if use_usage_scope:
            check_result = plugin.usage_manager.check_rate_limit(
                scope,
                is_admin=safe_is_admin,
                requested_count=requested_count,
                update_timestamp=False,
            )
            if isinstance(check_result, str):
                logger.warning(
                    f"{LOG} 公共接口触发使用限制: {safe_log_text(check_result)} "
                    f"(用户: {mask_sensitive(scope)})"
                )
                return self._submit_error(
                    PublicAPIResultCode.RATE_LIMITED,
                    check_result,
                )

        prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
            final_prompt,
            scope,
        )
        if not prompt_allowed:
            return self._submit_error(
                PublicAPIResultCode.PROMPT_BLOCKED,
                f"提示词审核未通过: {prompt_reason}",
                error=prompt_reason,
            )

        if use_usage_scope:
            check_result = plugin.usage_manager.check_rate_limit(
                scope,
                is_admin=safe_is_admin,
                requested_count=requested_count,
            )
            if isinstance(check_result, str):
                logger.warning(
                    f"{LOG} 公共接口触发使用限制: {safe_log_text(check_result)} "
                    f"(用户: {mask_sensitive(scope)})"
                )
                return self._submit_error(
                    PublicAPIResultCode.RATE_LIMITED,
                    check_result,
                )

        references = await self._collect_reference_images(
            reference_image_sources=reference_image_sources,
            reference_image_data=reference_image_data,
            persona_images=persona_images,
            task_id=task_id,
            unified_msg_origin=scope,
        )

        preset_summary, preset_label = self._format_template_summary(
            matched_presets,
            matched_personas,
        )
        record = plugin.create_generation_task(
            task_id=task_id,
            source=request_source,
            prompt=final_prompt,
            images_data=references,
            unified_msg_origin=scope,
            aspect_ratio=str(final_aspect_ratio),
            resolution=str(final_resolution),
            image_count=requested_count,
            is_usage_limit_admin=safe_is_admin,
            preset=preset_summary,
            preset_label=preset_label,
            presets=matched_presets,
            personas=matched_personas,
            auto_send=False,
        )
        return ImageGenerationSubmitResult(
            ok=True,
            code=PublicAPIResultCode.ACCEPTED.value,
            message="任务已提交",
            task_id=record.task_id,
        )

    def get_generation_task(
        self,
        task_id: str,
    ) -> ImageGenerationTaskSnapshot | None:
        """Return one task snapshot by task id."""
        record = self._plugin.task_manager.get_generation_task(task_id.strip())
        return self._snapshot(record) if record else None

    def cancel_generation_task(
        self,
        task_id: str,
        *,
        unified_msg_origin: str | None = None,
    ) -> ImageGenerationOperationResult:
        """Cancel one active image generation task."""
        normalized_task_id = task_id.strip()
        scope = str(unified_msg_origin or "").strip() or None
        ok, message = self._plugin.task_manager.cancel_generation_task(
            normalized_task_id,
            unified_msg_origin=scope,
        )
        return ImageGenerationOperationResult(
            ok=ok,
            code=(
                PublicAPIResultCode.CANCEL_REQUESTED.value
                if ok
                else PublicAPIResultCode.CANCEL_FAILED.value
            ),
            message=message,
            task_id=normalized_task_id,
            error="" if ok else message,
        )

    async def wait_generation_result(
        self,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = DEFAULT_WAIT_POLL_INTERVAL_SECONDS,
    ) -> ImageGenerationResult:
        """Wait until a task finishes and return generated image paths."""
        normalized_task_id = task_id.strip()
        record = self._plugin.task_manager.get_generation_task(normalized_task_id)
        if not record:
            return ImageGenerationResult(
                ok=False,
                code=PublicAPIResultCode.NOT_FOUND.value,
                message=f"任务不存在: {normalized_task_id}",
                task_id=normalized_task_id,
                paths=[],
                error=f"任务不存在: {normalized_task_id}",
            )

        deadline = None
        if timeout_seconds is not None:
            deadline = time.monotonic() + max(0.0, float(timeout_seconds))

        interval = max(0.05, float(poll_interval_seconds))
        while record.is_active:
            if deadline is not None and time.monotonic() >= deadline:
                return ImageGenerationResult(
                    ok=False,
                    code=PublicAPIResultCode.TIMEOUT.value,
                    message="等待任务完成超时",
                    task_id=normalized_task_id,
                    paths=[],
                    error="等待任务完成超时",
                )
            await asyncio.sleep(interval)
            record = self._plugin.task_manager.get_generation_task(normalized_task_id)
            if not record:
                return ImageGenerationResult(
                    ok=False,
                    code=PublicAPIResultCode.NOT_FOUND.value,
                    message=f"任务不存在: {normalized_task_id}",
                    task_id=normalized_task_id,
                    paths=[],
                    error=f"任务不存在: {normalized_task_id}",
                )

        if record.status.value == "succeeded" and record.result_paths:
            return ImageGenerationResult(
                ok=True,
                code=PublicAPIResultCode.SUCCEEDED.value,
                message=record.message or "任务已完成",
                task_id=normalized_task_id,
                paths=list(record.result_paths),
            )
        if record.status.value == "succeeded":
            return ImageGenerationResult(
                ok=False,
                code=PublicAPIResultCode.NO_RESULT.value,
                message="任务已完成但没有生成图片路径",
                task_id=normalized_task_id,
                paths=[],
                error="任务已完成但没有生成图片路径",
            )
        return ImageGenerationResult(
            ok=False,
            code=PublicAPIResultCode.from_task_status(record.status.value).value,
            message=record.error or record.message or record.status_label,
            task_id=normalized_task_id,
            paths=list(record.result_paths),
            error=record.error,
        )

    async def generate_image_files(
        self,
        *,
        prompt: str = "",
        unified_msg_origin: str | None = None,
        source: str = DEFAULT_SOURCE,
        image_count: int | None = None,
        aspect_ratio: str | None = None,
        resolution: str | None = None,
        reference_image_sources: Any = None,
        reference_image_data: list[ImageData | tuple[bytes, str]] | None = None,
        presets: str | list[str] | None = None,
        personas: str | list[str] | None = None,
        is_admin: bool = False,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = DEFAULT_WAIT_POLL_INTERVAL_SECONDS,
    ) -> ImageGenerationResult:
        """Submit an image generation task and wait for local image file paths."""
        submit_result = await self.submit_generation_task(
            prompt=prompt,
            unified_msg_origin=unified_msg_origin,
            source=source,
            image_count=image_count,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            reference_image_sources=reference_image_sources,
            reference_image_data=reference_image_data,
            presets=presets,
            personas=personas,
            is_admin=is_admin,
        )
        if not submit_result.ok or not submit_result.task_id:
            return ImageGenerationResult(
                ok=False,
                code=submit_result.code,
                message=submit_result.message,
                task_id=submit_result.task_id or "",
                paths=[],
                error=submit_result.error or submit_result.message,
            )
        return await self.wait_generation_result(
            submit_result.task_id,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def _submit_error(
        self,
        code: PublicAPIResultCode,
        message: str,
        *,
        error: str = "",
    ) -> ImageGenerationSubmitResult:
        return ImageGenerationSubmitResult(
            ok=False,
            code=code.value,
            message=message,
            error=error or message,
        )

    def _snapshot(
        self,
        record: GenerationTaskRecord,
    ) -> ImageGenerationTaskSnapshot:
        return ImageGenerationTaskSnapshot(
            task_id=record.task_id,
            status=record.status.value,
            active=record.is_active,
            source=record.source,
            requested_count=record.requested_count,
            result_count=record.result_count,
            reference_image_count=record.reference_image_count,
            aspect_ratio=record.aspect_ratio,
            resolution=record.resolution,
            result_paths=list(record.result_paths),
            error=record.error,
            message=record.message,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            duration_seconds=record.duration_seconds,
        )

    def _new_task_id(self) -> str:
        return new_task_id()

    def _normalize_name_items(self, raw: Any) -> list[str]:
        return normalize_name_items(raw)

    def _find_named_entry(self, entries: dict[str, Any], token: str) -> str | None:
        return find_named_entry(entries, token)

    def _parse_preset_prompt(
        self,
        preset_content: Any,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[str, str, str]:
        return parse_preset_prompt(preset_content, aspect_ratio, resolution)

    def _build_prompt(
        self,
        *,
        prompt: str,
        presets: str | list[str] | None,
        personas: str | list[str] | None,
        aspect_ratio: str,
        resolution: str,
    ) -> tuple[
        str,
        str,
        str,
        list[str],
        list[str],
        list[tuple[str, str]],
        str | None,
    ]:
        prompt_parts: list[str] = []
        matched_presets: list[str] = []
        matched_personas: list[str] = []
        persona_images: list[tuple[str, str]] = []
        config_manager = self._plugin.config_manager

        for preset_name in self._normalize_name_items(presets):
            matched_preset = self._find_named_entry(
                config_manager.presets,
                preset_name,
            )
            if not matched_preset:
                return (
                    "",
                    aspect_ratio,
                    resolution,
                    [],
                    [],
                    [],
                    f"预设不存在: {preset_name}",
                )
            preset_prompt, aspect_ratio, resolution = self._parse_preset_prompt(
                config_manager.presets[matched_preset],
                aspect_ratio,
                resolution,
            )
            if preset_prompt:
                prompt_parts.append(preset_prompt)
            matched_presets.append(matched_preset)

        for persona_name in self._normalize_name_items(personas):
            matched_persona = self._find_named_entry(
                config_manager.personas,
                persona_name,
            )
            if not matched_persona:
                return (
                    "",
                    aspect_ratio,
                    resolution,
                    [],
                    [],
                    [],
                    f"人设不存在: {persona_name}",
                )
            persona = config_manager.personas[matched_persona]
            persona_prompt = persona.prompt.strip()
            if persona_prompt:
                prompt_parts.append(persona_prompt)
            if persona.image:
                persona_images.append((matched_persona, persona.image))
            matched_personas.append(matched_persona)

        if extra_prompt := str(prompt or "").strip():
            prompt_parts.append(extra_prompt)
        final_prompt = " ".join(part for part in prompt_parts if part).strip()
        return (
            final_prompt,
            aspect_ratio,
            resolution,
            matched_presets,
            matched_personas,
            persona_images,
            None,
        )

    def _format_template_summary(
        self,
        matched_presets: list[str],
        matched_personas: list[str],
    ) -> tuple[str | None, str]:
        return format_template_summary(matched_presets, matched_personas)

    async def _collect_reference_images(
        self,
        *,
        reference_image_sources: Any,
        reference_image_data: list[ImageData] | None,
        persona_images: list[tuple[str, str]],
        task_id: str,
        unified_msg_origin: str | None = None,
    ) -> list[ImageData]:
        plugin = self._plugin
        if not plugin.generator or not plugin.generator.adapter:
            return []
        capabilities = plugin.generator.adapter.get_capabilities()
        has_references = bool(
            persona_images or reference_image_sources or reference_image_data
        )
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
            if has_references:
                logger.warning(
                    f"{log_prefix('PublicAPI', task_id)} 当前适配器不支持参考图，已忽略公共接口参考图参数"
                )
            return []

        images_data: list[ImageData] = []
        workspace_dir = plugin.image_processor.workspace_dir_for_origin(
            unified_msg_origin
        )
        if persona_images:
            images_data.extend(
                await collect_reference_images_from_personas(
                    plugin.image_processor,
                    persona_images,
                    task_id=task_id,
                    log_context="PublicAPI",
                    workspace_dir=workspace_dir,
                )
            )
        images_data.extend(
            await download_reference_images(
                plugin.image_processor,
                reference_image_sources,
                reference_label="公共接口",
                task_id=task_id,
                log_context="PublicAPI",
                workspace_dir=workspace_dir,
            )
        )
        images_data.extend(self._normalize_reference_image_data(reference_image_data))
        return deduplicate_reference_images(
            images_data,
            task_id=task_id,
            log_context="PublicAPI",
        )

    def _normalize_reference_image_data(
        self,
        reference_image_data: list[ImageData] | None,
    ) -> list[ImageData]:
        images_data: list[ImageData] = []
        max_size = self._plugin.config_manager.usage_settings.max_image_size_mb
        for item in reference_image_data or []:
            source_url = None
            if isinstance(item, ImageData):
                data = item.data
                source_url = item.source_url
            else:
                try:
                    data, _mime = item
                except (TypeError, ValueError):
                    logger.warning(f"{LOG} 已忽略格式错误的二进制参考图")
                    continue
            if isinstance(data, bytearray):
                data = bytes(data)
            if not isinstance(data, bytes) or not data:
                logger.warning(f"{LOG} 已忽略空的二进制参考图")
                continue
            if len(data) > max_size * 1024 * 1024:
                logger.warning(f"{LOG} 已忽略超过大小限制的二进制参考图 ({max_size}MB)")
                continue
            image_data = self._plugin.image_processor.validate_image_data(
                data,
                source_url=source_url,
                log_source=source_url or "公共接口二进制参考图",
            )
            if image_data:
                images_data.append(image_data)
        return images_data
