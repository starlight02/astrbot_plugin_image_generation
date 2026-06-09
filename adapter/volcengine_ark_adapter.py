from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import UNSPECIFIED_OPTION, VOLCENGINE_ARK_DEFAULT_BASE_URL
from ..core.logging_utils import (
    safe_log_error_body,
    safe_log_mapping,
    safe_log_url,
)
from ..core.types import GenerationRequest, ImageCapability, ImageData


class VolcengineArkAdapter(BaseImageAdapter):
    """Volcengine Ark Seedream image generation adapter."""

    DEFAULT_BASE_URL = VOLCENGINE_ARK_DEFAULT_BASE_URL

    SIZE_MAPS: dict[str, dict[str, str]] = {
        "1K": {
            "1:1": "1024x1024",
            "4:3": "1152x864",
            "3:4": "864x1152",
            "16:9": "1280x720",
            "9:16": "720x1280",
            "3:2": "1248x832",
            "2:3": "832x1248",
            "21:9": "1512x648",
            "4:5": "864x1152",
            "5:4": "1152x864",
        },
        "2K": {
            "1:1": "2048x2048",
            "4:3": "2304x1728",
            "3:4": "1728x2304",
            "16:9": "2848x1600",
            "9:16": "1600x2848",
            "3:2": "2496x1664",
            "2:3": "1664x2496",
            "21:9": "3136x1344",
            "4:5": "1728x2304",
            "5:4": "2304x1728",
        },
        "3K": {
            "1:1": "3072x3072",
            "4:3": "3456x2592",
            "3:4": "2592x3456",
            "16:9": "4096x2304",
            "9:16": "2304x4096",
            "3:2": "3744x2496",
            "2:3": "2496x3744",
            "21:9": "4704x2016",
            "4:5": "2592x3456",
            "5:4": "3456x2592",
        },
        "4K": {
            "1:1": "4096x4096",
            "4:3": "4704x3520",
            "3:4": "3520x4704",
            "16:9": "5504x3040",
            "9:16": "3040x5504",
            "3:2": "4992x3328",
            "2:3": "3328x4992",
            "21:9": "6240x2656",
            "4:5": "3520x4704",
            "5:4": "4704x3520",
        },
    }

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次火山方舟图片生成请求。"""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }
        url = self._endpoint_url()

        logger.debug(
            f"{prefix} 请求 URL: {safe_log_url(url)}, Payload 字段: {list(payload.keys())}"
        )
        self._log_debug_json("请求", payload, request.task_id)

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                proxy=self.proxy,
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    self._log_debug_json_text("响应", error_text, request.task_id)
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {safe_log_error_body(error_text)}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await self._read_response_json(resp, request.task_id)
                logger.debug(f"{prefix} 生成成功 (耗时: {duration:.2f}s)")
                return await self._extract_images(data, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {exc}")
            return None, str(exc)

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """构建火山方舟图片生成请求。"""
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "prompt": request.prompt,
            "response_format": "b64_json",
        }

        size = self._resolve_size(request)
        if size:
            payload["size"] = size

        if request.images:
            self._add_images(payload, request.images, request.task_id)

        self._add_extra_options(payload)
        return payload

    def _endpoint_url(self) -> str:
        """Return a usable Ark image generation endpoint URL."""
        base = (self.base_url or self.DEFAULT_BASE_URL).rstrip("/")
        if base.endswith("/api/v3/images/generations"):
            return base
        if base.endswith("/api/v3"):
            return f"{base}/images/generations"
        return f"{base}/api/v3/images/generations"

    def _model_name(self) -> str:
        """获取当前模型名称。"""
        return self.model or "doubao-seedream-5.0-lite"

    def _resolve_size(self, request: GenerationRequest) -> str | None:
        """按配置、分辨率和宽高比解析 Ark size 参数。"""
        if (
            not request.aspect_ratio
            or request.aspect_ratio == UNSPECIFIED_OPTION
            or not request.resolution
            or request.resolution == UNSPECIFIED_OPTION
        ):
            return None

        resolution = self._normalize_resolution(request.resolution)
        aspect_ratio = request.aspect_ratio

        return self.SIZE_MAPS.get(resolution, self.SIZE_MAPS["2K"]).get(
            aspect_ratio, self.SIZE_MAPS[resolution]["1:1"]
        )

    def _normalize_resolution(self, resolution: str | None) -> str:
        """Map plugin resolution values to sizes supported by the selected model."""
        value = resolution or "2K"
        model = self._model_name().lower()

        if "seedream-4.0" in model or "seedream-4-0" in model:
            return value if value in {"1K", "2K", "4K"} else "2K"
        if "seedream-5.0" in model or "seedream-5-0" in model:
            return value if value in {"2K", "3K", "4K"} else "2K"
        return value if value in {"2K", "4K"} else "2K"

    def _add_images(
        self, payload: dict[str, Any], images: list[ImageData], task_id: str | None
    ) -> None:
        """将参考图添加为 Ark 支持的 data URL。"""
        max_images = self._coerce_int(
            self.config.extra.get("max_reference_images"),
            default=14,
            min_value=1,
            max_value=14,
        )
        selected_images = images[:max_images]
        if len(images) > max_images:
            logger.debug(
                f"{self._get_log_prefix(task_id)} 当前配置最多使用 {max_images} 张参考图，已忽略多余图片"
            )

        image_values = [self._to_data_url(image) for image in selected_images]
        payload["image"] = image_values[0] if len(image_values) == 1 else image_values

    def _to_data_url(self, image: ImageData) -> str:
        """Convert image data to a Volcengine Ark-compatible data URL."""
        mime_type = (image.mime_type or "image/png").lower()
        b64_data = base64.b64encode(image.data).decode("ascii")
        return f"data:{mime_type};base64,{b64_data}"

    def _add_extra_options(self, payload: dict[str, Any]) -> None:
        """添加火山方舟可选生图参数。"""
        extra = self.config.extra

        payload["watermark"] = self._coerce_bool(extra.get("watermark"), default=True)

        sequential = str(extra.get("sequential_image_generation") or "disabled").strip()
        if sequential in {"auto", "disabled"}:
            payload["sequential_image_generation"] = sequential
            if sequential == "auto":
                max_images = self._coerce_int(
                    extra.get("sequential_max_images"),
                    default=15,
                    min_value=1,
                    max_value=15,
                )
                payload["sequential_image_generation_options"] = {
                    "max_images": max_images
                }

        optimize_mode = str(extra.get("optimize_prompt_mode") or "").strip()
        if optimize_mode in {"standard", "fast"}:
            payload["optimize_prompt_options"] = {"mode": optimize_mode}

        if self._coerce_bool(extra.get("enable_web_search"), default=False):
            payload["tools"] = [{"type": "web_search"}]

    async def _extract_images(
        self, response: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """从火山方舟响应中提取图片数据。"""
        if response_error := response.get("error"):
            if isinstance(response_error, dict):
                message = response_error.get("message") or response_error.get("code")
                return None, f"API 错误: {message}"
            return None, f"API 错误: {response_error}"

        data_items = response.get("data")
        if not isinstance(data_items, list):
            return None, f"响应中未找到 data 字段: {safe_log_mapping(response)}"

        images: list[bytes] = []
        errors: list[str] = []
        for item in data_items:
            if not isinstance(item, dict):
                continue

            if item_error := item.get("error"):
                errors.append(self._format_item_error(item_error))
                continue

            if b64_json := item.get("b64_json"):
                try:
                    images.append(base64.b64decode(str(b64_json)))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        f"{self._get_log_prefix(task_id)} Base64 解码失败: {exc}"
                    )
                continue

            url = item.get("url")
            if isinstance(url, str) and url:
                if data := await self._download_image(url, task_id):
                    images.append(data)

        if images:
            return images, None
        if errors:
            return None, "; ".join(errors)
        return None, "未找到有效的图片数据"

    def _format_item_error(self, error: Any) -> str:
        """格式化单张图片生成错误。"""
        if not isinstance(error, dict):
            return str(error)
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip()
        if code and message:
            return f"{code}: {message}"
        return message or code or str(error)

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """下载火山方舟返回的临时图片 URL。"""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(
                    f"{prefix} 下载图像失败 ({resp.status}): {safe_log_url(url)}"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {exc}")
        return None

    def _coerce_int(
        self,
        value: Any,
        *,
        default: int,
        min_value: int,
        max_value: int,
    ) -> int:
        """安全转换整数配置。"""
        if value in (None, "") or isinstance(value, bool):
            return default
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return default
        return max(min_value, min(max_value, parsed))

    def _coerce_bool(self, value: Any, *, default: bool) -> bool:
        """安全转换布尔配置。"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "on", "开启", "启用"}:
                return True
            if lowered in {"false", "0", "no", "off", "关闭", "禁用"}:
                return False
        return default
