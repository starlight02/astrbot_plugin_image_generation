from __future__ import annotations

import base64
import time
from typing import Any

import aiohttp
from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, ImageCapability


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI 图像生成适配器 (DALL-E / GPT Image Models)。"""

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    def _is_gpt_image_model(self) -> bool:
        """判断当前是否为 GPT image model (gpt-image-*)。"""
        model_family = self.config.extra.get("model_family", "auto")
        if model_family == "gpt-image":
            return True
        if model_family == "dall-e":
            return False
        # auto: 根据模型名称判断
        return self.model is not None and "gpt-image" in self.model

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        prefix = self._get_log_prefix(request.task_id)

        is_gpt = self._is_gpt_image_model()
        use_edit = bool(request.images) and is_gpt
        if request.images and not is_gpt:
            logger.warning(
                f"{prefix} 提供了参考图但当前模型不支持图生图，仅 GPT Image 系列支持图生图，参考图将被忽略"
            )
        session = self._get_session()
        base = (self.base_url.rstrip("/") if self.base_url else "https://api.openai.com")
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}

        if use_edit:
            url = f"{base}/v1/images/edits"
            form = aiohttp.FormData()
            form.add_field("model", self.model or "gpt-image-1")
            form.add_field("prompt", request.prompt)
            form.add_field("n", "1")
            if size := self._map_aspect_ratio_to_size(request.aspect_ratio, gpt_model=True):
                form.add_field("size", size)
            for img in request.images:
                form.add_field(
                    "image[]", img.data,
                    content_type=img.mime_type,
                    filename="image",
                )
            kwargs: dict = {"data": form}
        else:
            url = f"{base}/v1/images/generations"
            headers["Content-Type"] = "application/json"
            kwargs = {"json": self._build_payload(request)}

        try:
            async with session.post(
                url, headers=headers, proxy=self.proxy, timeout=self._get_timeout(), **kwargs
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
                return await self._extract_images(data)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}")
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        gpt = self._is_gpt_image_model()
        payload: dict[str, Any] = {
            "model": self.model or "dall-e-3",
            "prompt": request.prompt,
            "n": 1,
        }

        if size := self._map_aspect_ratio_to_size(request.aspect_ratio, gpt_model=gpt):
            payload["size"] = size
        # 注意：OpenAI 模型不支持配置分辨率（输出固定为 1K~1.5K），quality 是品质参数，与分辨率无关
        if not gpt:
            # GPT image models 始终返回 b64_json，不支持 response_format 参数
            payload["response_format"] = "b64_json"

        return payload

    def _map_aspect_ratio_to_size(
        self, aspect_ratio: str | None, gpt_model: bool
    ) -> str | None:
        """将宽高比映射为 OpenAI 支持的 size 参数。"""
        if not aspect_ratio or aspect_ratio == "自动":
            if gpt_model:
                return "auto"
            return "1024x1024"

        if gpt_model:
            # GPT image models 仅支持 auto, 1024x1024, 1536x1024 (横), 1024x1536 (竖)
            # 如果用户指定了其他比例，尽量匹配最接近的
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1536x1024",
                "16:9": "1536x1024",
                "4:3": "1536x1024",
                "5:4": "1536x1024",
                "21:9": "1536x1024",
                "2:3": "1024x1536",
                "3:4": "1024x1536",
                "9:16": "1024x1536",
                "4:5": "1024x1536",
            }
        else:
            # DALL-E 3 仅支持 1024x1024, 1024x1792, 1792x1024
            # 如果用户指定了其他比例，尽量匹配最接近的
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1792x1024",
                "16:9": "1792x1024",
                "4:3": "1792x1024",
                "5:4": "1792x1024",
                "21:9": "1792x1024",
                "2:3": "1024x1792",
                "3:4": "1024x1792",
                "9:16": "1024x1792",
                "4:5": "1024x1792",
            }
        return mapping.get(aspect_ratio)

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """从响应中提取图片数据。"""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                # 如果返回的是 URL，需要下载（虽然我们请求的是 b64_json）
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
