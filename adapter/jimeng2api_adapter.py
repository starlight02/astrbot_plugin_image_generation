from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import UNSPECIFIED_OPTION
from ..core.logging_utils import safe_log_error_body, safe_log_mapping, safe_log_url
from ..core.types import GenerationRequest, ImageCapability


class Jimeng2APIAdapter(BaseImageAdapter):
    """Jimeng2API 图像生成适配器。"""

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基类提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        prompt_text = request.prompt
        if prompt_text is None:
            return None, "缺少提示词"
        if not isinstance(prompt_text, str):
            logger.warning(f"{prefix} prompt 非字符串类型: {type(prompt_text)}")
            prompt_text = str(prompt_text)

        base_url = self.base_url or "http://localhost:5100"
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
        }

        try:
            if request.images:
                # 图生图：改为 JSON，images 作为 data URL（服务端声明只接受 URL 或本地文件）
                url = f"{base_url.rstrip('/')}/v1/images/compositions"
                headers["Content-Type"] = "application/json"

                images_as_urls: list[str] = []
                for img in request.images:
                    mime = img.mime_type or "image/jpeg"
                    b64 = base64.b64encode(img.data).decode("ascii")
                    images_as_urls.append(f"data:{mime};base64,{b64}")

                payload: dict[str, object] = {
                    "model": self.model or "jimeng-4.5",
                    "prompt": prompt_text,
                    "images": images_as_urls,
                }
                if request.aspect_ratio and request.aspect_ratio != UNSPECIFIED_OPTION:
                    payload["ratio"] = request.aspect_ratio
                if request.resolution and request.resolution != UNSPECIFIED_OPTION:
                    payload["resolution"] = request.resolution.lower()
                self._log_debug_json("请求", payload, request.task_id)

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
                            f"{prefix} Compositions 错误 ({resp.status}, 耗时: {duration:.2f}s): {safe_log_error_body(error_text)}"
                        )
                        return None, f"API 错误 ({resp.status})"

                    data_json = await self._read_response_json(resp, request.task_id)
                    logger.debug(
                        f"{prefix} Compositions 响应: {safe_log_mapping(data_json)}"
                    )
                    logger.debug(f"{prefix} Compositions 成功 (耗时: {duration:.2f}s)")
                    return await self._extract_images(data_json, request.task_id)
            else:
                # 文生图
                url = f"{base_url.rstrip('/')}/v1/images/generations"
                headers["Content-Type"] = "application/json"

                payload = {
                    "model": self.model or "jimeng-4.5",
                    "prompt": prompt_text,
                    "response_format": "url",  # 默认使用 url，然后下载
                }
                if request.aspect_ratio and request.aspect_ratio != UNSPECIFIED_OPTION:
                    payload["ratio"] = request.aspect_ratio
                if request.resolution and request.resolution != UNSPECIFIED_OPTION:
                    payload["resolution"] = request.resolution.lower()
                self._log_debug_json("请求", payload, request.task_id)

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
                            f"{prefix} Generations 错误 ({resp.status}, 耗时: {duration:.2f}s): {safe_log_error_body(error_text)}"
                        )
                        return None, f"API 错误 ({resp.status})"

                    data_json = await self._read_response_json(resp, request.task_id)
                    logger.debug(
                        f"{prefix} Generations 响应: {safe_log_mapping(data_json)}"
                    )
                    logger.debug(f"{prefix} Generations 成功 (耗时: {duration:.2f}s)")
                    return await self._extract_images(data_json, request.task_id)

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}")
            return None, str(e)

    async def _extract_images(
        self, response: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """从响应中提取图片数据。"""
        prefix = self._get_log_prefix(task_id)
        if response is None:
            return None, "响应为空"
        if "data" not in response:
            return None, f"响应中未找到 data 字段: {safe_log_mapping(response)}"

        data = response.get("data")
        if data is None:
            return None, "data 字段为 None"

        images = []
        for item in data:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                async with self._get_session().get(
                    item["url"], proxy=self.proxy, timeout=self._get_download_timeout()
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())
                    else:
                        logger.error(
                            f"{prefix} 下载图像失败 ({resp.status}): {safe_log_url(item['url'])}"
                        )

        if not images:
            return None, "未找到有效的图片数据"

        return images, None

    async def receive_token(self) -> dict[str, Any]:
        """为所有 API Key 自动领取积分。"""
        results = {}
        if not self.api_keys:
            return {"error": "未配置 API Key"}

        base_url = self.base_url or "http://localhost:5100"
        url = f"{base_url.rstrip('/')}/token/receive"

        for i, key in enumerate(self.api_keys):
            headers = {
                "Authorization": f"Bearer {key}",
            }
            try:
                async with self._get_session().post(
                    url,
                    headers=headers,
                    proxy=self.proxy,
                    timeout=self._get_download_timeout(),
                ) as resp:
                    resp_json = await resp.json()
                    status_code = resp.status
                    results[f"key_{i}"] = {"status": status_code, "data": resp_json}
                    if status_code == 200:
                        logger.info(
                            f"{self._get_log_prefix()} API Key (索引 {i}) 积分领取成功: {safe_log_mapping(resp_json)}"
                        )
                    else:
                        logger.warning(
                            f"{self._get_log_prefix()} API Key (索引 {i}) 积分领取失败 ({status_code}): {safe_log_mapping(resp_json)}"
                        )
            except Exception as e:
                logger.error(
                    f"{self._get_log_prefix()} API Key (索引 {i}) 积分领取请求异常: {e}"
                )
                results[f"key_{i}"] = {"error": str(e)}

        return results
