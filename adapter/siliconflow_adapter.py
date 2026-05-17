from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import SILICONFLOW_DEFAULT_BASE_URL
from ..core.types import GenerationRequest, ImageCapability, ImageData


class SiliconFlowAdapter(BaseImageAdapter):
    """SiliconFlow 图像生成适配器。"""

    DEFAULT_BASE_URL = SILICONFLOW_DEFAULT_BASE_URL

    KOLORS_IMAGE_SIZE_MAP = {
        "1:1": "1024x1024",
        "3:4": "960x1280",
        "4:5": "960x1280",
        "1:2": "720x1440",
        "9:16": "720x1280",
        "2:3": "768x1024",
        "3:2": "1024x768",
        "4:3": "1024x768",
        "5:4": "1024x768",
        "16:9": "1280x720",
        "21:9": "1280x720",
    }
    QWEN_IMAGE_SIZE_MAP = {
        "1:1": "1328x1328",
        "16:9": "1664x928",
        "9:16": "928x1664",
        "4:3": "1472x1140",
        "3:4": "1140x1472",
        "3:2": "1584x1056",
        "2:3": "1056x1584",
        "4:5": "1140x1472",
        "5:4": "1472x1140",
        "21:9": "1664x928",
    }

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        base = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base.rstrip('/')}/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
        }

        logger.debug(f"{prefix} 请求 URL: {url}, Payload 字段: {list(payload.keys())}")

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
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await resp.json()
                logger.info(f"{prefix} 生成成功 (耗时: {duration:.2f}s)")
                return await self._extract_images(data, request.task_id)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {exc}")
            return None, str(exc)

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """构建 SiliconFlow 图片生成请求。"""
        payload: dict[str, Any] = {
            "model": self._model_name(),
            "prompt": request.prompt,
        }

        self._add_extra_options(payload)

        if not self._is_qwen_edit_model():
            image_size = self._resolve_image_size(request)
            if image_size:
                payload["image_size"] = image_size

        if request.images:
            self._add_images(payload, request.images)

        return payload

    def _add_extra_options(self, payload: dict[str, Any]) -> None:
        """添加 SiliconFlow 可选生图参数。"""
        extra = self.config.extra

        if negative_prompt := str(extra.get("negative_prompt") or "").strip():
            payload["negative_prompt"] = negative_prompt

        steps = self._coerce_int(
            extra.get("num_inference_steps"), min_value=1, max_value=100
        )
        if steps is not None:
            payload["num_inference_steps"] = steps

        guidance_scale = self._coerce_float(
            extra.get("guidance_scale"), min_value=0, max_value=20
        )
        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale

    def _resolve_image_size(self, request: GenerationRequest) -> str | None:
        """按模型和宽高比解析 SiliconFlow image_size。"""
        aspect_ratio = request.aspect_ratio or "1:1"
        if aspect_ratio == "自动":
            aspect_ratio = "1:1"

        if self._is_qwen_image_model():
            return self.QWEN_IMAGE_SIZE_MAP.get(aspect_ratio, "1328x1328")
        return self.KOLORS_IMAGE_SIZE_MAP.get(aspect_ratio, "1024x1024")

    def _add_images(self, payload: dict[str, Any], images: list[ImageData]) -> None:
        """将参考图添加为 SiliconFlow 支持的 data URL。"""
        max_images = 3 if self._is_qwen_edit_model() else 1
        if len(images) > max_images:
            logger.info(
                f"{self._get_log_prefix()} 当前模型最多使用 {max_images} 张参考图，已忽略多余图片"
            )

        field_names = ("image", "image2", "image3")
        for field_name, image in zip(field_names, images[:max_images], strict=False):
            mime_type = image.mime_type or "image/png"
            b64_data = base64.b64encode(image.data).decode("ascii")
            payload[field_name] = f"data:{mime_type};base64,{b64_data}"

    async def _extract_images(
        self, response: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """从 SiliconFlow 响应中提取并下载图片。"""
        prefix = self._get_log_prefix(task_id)
        image_items = response.get("images")
        if not isinstance(image_items, list):
            return None, f"响应中未找到 images 字段: {response}"

        images: list[bytes] = []
        for item in image_items:
            if not isinstance(item, dict):
                continue

            if b64_json := item.get("b64_json"):
                try:
                    images.append(base64.b64decode(str(b64_json)))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"{prefix} Base64 解码失败: {exc}")
                continue

            url = item.get("url")
            if not isinstance(url, str) or not url:
                logger.warning(f"{prefix} 无法从响应项中提取图片: {item}")
                continue

            if url.startswith("data:image/"):
                if data := self._decode_data_url(url, task_id):
                    images.append(data)
                continue

            if data := await self._download_image(url, task_id):
                images.append(data)

        if not images:
            return None, "未找到有效的图片数据"
        return images, None

    async def _download_image(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        """下载 SiliconFlow 返回的临时图片 URL。"""
        prefix = self._get_log_prefix(task_id)
        try:
            async with self._get_session().get(
                url, proxy=self.proxy, timeout=self._get_download_timeout()
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(f"{prefix} 下载图像失败 ({resp.status}): {url}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图像异常: {exc}")
        return None

    def _decode_data_url(self, url: str, task_id: str | None = None) -> bytes | None:
        """解码 data URL 图片。"""
        if ";base64," not in url:
            return None
        try:
            _, _, data_part = url.partition(";base64,")
            return base64.b64decode(data_part)
        except Exception as exc:  # noqa: BLE001
            prefix = self._get_log_prefix(task_id)
            logger.warning(f"{prefix} Data URL 解码失败: {exc}")
            return None

    def _is_qwen_image_model(self) -> bool:
        """判断当前模型是否为 Qwen-Image 系列。"""
        return "qwen-image" in self._model_name().lower()

    def _is_qwen_edit_model(self) -> bool:
        """判断当前模型是否为 Qwen 图片编辑模型。"""
        model = self._model_name().lower()
        return "qwen-image-edit" in model

    def _model_name(self) -> str:
        """获取当前模型名称。"""
        return self.model or "Kwai-Kolors/Kolors"

    def _coerce_int(self, value: Any, *, min_value: int, max_value: int) -> int | None:
        """安全转换整数配置。"""
        if value in (None, "") or isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return max(min_value, min(max_value, parsed))

    def _coerce_float(
        self, value: Any, *, min_value: float, max_value: float
    ) -> float | None:
        """安全转换浮点数配置。"""
        if value in (None, "") or isinstance(value, bool):
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(min_value, min(max_value, parsed))
