from __future__ import annotations

import abc
import asyncio
import hashlib
import json
import re
from typing import Any

import aiohttp

from astrbot.api import logger

from .constants import DEFAULT_DOWNLOAD_TIMEOUT
from .logging_utils import log_prefix, mask_sensitive, safe_log_text
from .types import AdapterConfig, GenerationRequest, GenerationResult, ImageCapability


API_STATUS_ERROR_PATTERN = re.compile(r"API 错误\s*\((\d{3})\)")
DEBUG_JSON_STRING_LIMIT = 1000
DEBUG_JSON_EDGE_CHARS = 120
DEBUG_JSON_LIST_LIMIT = 50
BASE64_VALUE_RE = re.compile(r"^[A-Za-z0-9+/=\s]+$")


class BaseImageAdapter(abc.ABC):
    """图像生成适配器基类。"""

    requires_api_key = True

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.download_timeout = DEFAULT_DOWNLOAD_TIMEOUT
        self.max_retry_attempts = max(1, config.max_retry_attempts)
        self.debug_request_logging = config.debug_request_logging
        self.non_retryable_status_codes = set(config.non_retryable_status_codes)
        self.non_retryable_error_keywords = [
            keyword.lower()
            for keyword in config.non_retryable_error_keywords
            if keyword
        ]
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""

    def _get_configured_capabilities(self) -> ImageCapability:
        """根据配置项构建适配器能力。"""
        capability_map: dict[str, ImageCapability] = {
            "text_to_image": ImageCapability.TEXT_TO_IMAGE,
            "image_to_image": ImageCapability.IMAGE_TO_IMAGE,
            "aspect_ratio": ImageCapability.ASPECT_RATIO,
            "resolution": ImageCapability.RESOLUTION,
        }

        result = ImageCapability.NONE
        for key, capability_flag in capability_map.items():
            if self.config.capability_options.get(key, False):
                result |= capability_flag
        return result

    async def close(self) -> None:
        """关闭底层的 HTTP 会话。"""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话。"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key。"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_masked_api_key(self) -> str:
        """获取脱敏后的当前 API Key，用于日志输出。"""
        return mask_sensitive(self._get_current_api_key())

    def _get_log_prefix(self, task_id: str | None = None) -> str:
        """获取统一的日志前缀。"""
        adapter_name = self.__class__.__name__.replace("Adapter", "")
        return log_prefix(adapter_name, task_id)

    def _get_timeout(self) -> aiohttp.ClientTimeout:
        """获取统一的请求超时配置。"""
        return aiohttp.ClientTimeout(total=self.timeout)

    def _get_download_timeout(self) -> aiohttp.ClientTimeout:
        """获取统一的下载超时配置。"""
        return aiohttp.ClientTimeout(total=self.download_timeout)

    def _log_debug_json(
        self, label: str, value: Any, task_id: str | None = None
    ) -> None:
        """按需输出 JSON 调试日志，长字符串会摘要避免刷屏。"""
        if not self.debug_request_logging:
            return

        prefix = self._get_log_prefix(task_id)
        safe_value = self._sanitize_debug_json(value)
        json_text = json.dumps(
            safe_value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        logger.debug(f"{prefix} {label} JSON: {json_text}")

    def _sanitize_debug_json(self, value: Any) -> Any:
        """保留 JSON 结构，同时截断图片 Base64 和其他超长字符串。"""
        if isinstance(value, dict):
            return {key: self._sanitize_debug_json(item) for key, item in value.items()}
        if isinstance(value, list):
            items = [
                self._sanitize_debug_json(item)
                for item in value[:DEBUG_JSON_LIST_LIMIT]
            ]
            if len(value) > DEBUG_JSON_LIST_LIMIT:
                omitted_count = len(value) - DEBUG_JSON_LIST_LIMIT
                items.append(f"<list truncated: {omitted_count} items omitted>")
            return items
        if isinstance(value, str):
            return self._sanitize_debug_string(value)
        return value

    def _sanitize_debug_string(self, value: str) -> str:
        """截断调试 JSON 中可能撑爆日志的字符串值。"""
        if len(value) <= DEBUG_JSON_STRING_LIMIT:
            return value

        digest = hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:12]
        if value.startswith("data:"):
            media_type = value.split(",", 1)[0]
            return (
                f"<data-url omitted: {media_type}, len={len(value)}, sha256={digest}>"
            )
        if BASE64_VALUE_RE.fullmatch(value):
            return f"<base64 omitted: len={len(value)}, sha256={digest}>"

        head = value[:DEBUG_JSON_EDGE_CHARS]
        tail = value[-DEBUG_JSON_EDGE_CHARS:]
        return f"{head}<string truncated: len={len(value)}, sha256={digest}>{tail}"

    def _log_debug_json_text(
        self, label: str, value: str, task_id: str | None = None
    ) -> None:
        """当文本响应可解析为 JSON 时按需输出 JSON 调试日志。"""
        if not self.debug_request_logging:
            return

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return
        self._log_debug_json(label, parsed, task_id)

    async def _read_response_json(
        self, response: aiohttp.ClientResponse, task_id: str | None = None
    ) -> Any:
        """读取响应 JSON，并在开关开启时输出安全摘要后的响应体。"""
        data = await response.json()
        self._log_debug_json("响应", data, task_id)
        return data

    def _rotate_api_key(self) -> None:
        """轮换 API Key。"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.debug(
                f"{self._get_log_prefix()} 轮换 API Key -> 索引 {self.current_key_index}"
            )

    def update_model(self, model: str) -> None:
        """更新使用的模型。"""
        self.model = model

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """带重试逻辑的图像生成模板方法。

        子类应重写 `_generate_once()` 方法来实现具体的生成逻辑。
        如需在生成前进行预处理验证，可重写 `_pre_generate()` 方法。
        """
        if self.requires_api_key and not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        prefix = self._get_log_prefix(request.task_id)
        logger.debug(
            f"{prefix} 准备生图请求: 模型={safe_log_text(self.model)}，"
            f"进度={request.batch_index}/{request.batch_count}，"
            f"参考图={len(request.images)}张，最大重试={self.max_retry_attempts}次"
        )

        # 预处理检查（子类可重写）
        pre_result = self._pre_generate(request)
        if pre_result is not None:
            logger.warning(
                f"{prefix} 生图请求预检查未通过: {safe_log_text(pre_result.error)}"
            )
            return pre_result

        last_error = "未配置 API Key"
        for attempt in range(self.max_retry_attempts):
            if request.retry_status_callback:
                request.retry_status_callback(attempt + 1, self.max_retry_attempts)
            if attempt:
                logger.debug(
                    f"{prefix} 重试生图请求 ({attempt + 1}/{self.max_retry_attempts})"
                )

            images, err = await self._generate_once(request)
            if images is not None:
                logger.debug(f"{prefix} 生图请求完成: 图片={len(images)}张")
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失败"
            logger.warning(
                f"{prefix} 生图请求尝试失败 ({attempt + 1}/{self.max_retry_attempts}): "
                f"{safe_log_text(last_error, 200)}"
            )
            if not self._is_retryable_error(last_error):
                logger.warning(
                    f"{prefix} 生图请求错误不可重试，停止重试: {safe_log_text(last_error, 200)}"
                )
                return GenerationResult(images=None, error=last_error)
            if attempt < self.max_retry_attempts - 1:
                self._rotate_api_key()
                # 轮换 Key 时进行指数退避
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        logger.error(f"{prefix} 生图请求全部重试失败: {safe_log_text(last_error, 200)}")
        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    def _is_retryable_error(self, error: str) -> bool:
        """Return whether an adapter error should be retried."""
        if not error:
            return True

        if status_code := self._extract_status_code(error):
            return status_code not in self.non_retryable_status_codes

        normalized_error = error.lower()
        return not any(
            keyword in normalized_error for keyword in self.non_retryable_error_keywords
        )

    def _extract_status_code(self, error: str) -> int | None:
        """Extract an HTTP status code from a normalized adapter error."""
        match = API_STATUS_ERROR_PATTERN.search(error)
        if not match:
            return None
        try:
            return int(match.group(1))
        except ValueError:
            return None

    def _pre_generate(self, _request: GenerationRequest) -> GenerationResult | None:
        """生成前的预处理检查。

        子类可重写此方法进行参数验证。
        返回 None 表示通过检查，返回 GenerationResult 表示提前返回错误。
        """
        return None

    @abc.abstractmethod
    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生成请求。

        子类必须实现此方法。
        返回 (images, error) 元组，成功时 images 非空，失败时 error 非空。
        """
