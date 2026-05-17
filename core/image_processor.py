"""
图片处理模块 - 下载、提取、临时文件保存
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.core.utils.io import download_image_by_url

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


class ImageProcessor:
    """图片处理器 - 负责图片下载、提取和临时文件保存。"""

    def __init__(self, temp_dir: str, max_image_size_mb: int):
        self._temp_dir = temp_dir
        self._max_image_size_mb = max_image_size_mb
        self._ensure_temp_dir()

    def _ensure_temp_dir(self) -> None:
        """确保临时目录存在。"""
        os.makedirs(self._temp_dir, exist_ok=True)

    def update_settings(self, max_image_size_mb: int | None = None) -> None:
        """更新设置。"""
        if max_image_size_mb is not None:
            self._max_image_size_mb = max_image_size_mb

    @property
    def temp_dir(self) -> str:
        """获取临时目录路径。"""
        return self._temp_dir

    async def download_image(self, url: str) -> tuple[bytes, str] | None:
        """下载图片并返回二进制数据和 MIME 类型。"""
        try:
            data: bytes | None = None
            if os.path.exists(url) and os.path.isfile(url):
                with open(url, "rb") as f:
                    data = f.read()
            else:
                # 使用插件临时目录
                file_name = f"ref_{hashlib.md5(url.encode()).hexdigest()[:10]}"
                path = os.path.join(self._temp_dir, file_name)
                path = await download_image_by_url(url, path=path)
                if path:
                    with open(path, "rb") as f:
                        data = f.read()

            if not data:
                return None

            if len(data) > self._max_image_size_mb * 1024 * 1024:
                logger.warning(
                    f"[ImageGen] 图片超过大小限制 ({self._max_image_size_mb}MB)"
                )
                return None

            mime = self._detect_mime_type(data)
            return data, mime
        except Exception as exc:
            logger.error(f"[ImageGen] 获取图片失败 (URL/Path: {url}): {exc}")
        return None

    def _detect_mime_type(self, data: bytes) -> str:
        """检测图片 MIME 类型。"""
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        elif data.startswith(b"GIF"):
            return "image/gif"
        elif data.startswith(b"RIFF") and b"WEBP" in data[:16]:
            return "image/webp"
        return "image/png"

    async def get_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像。"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        try:
            file_name = f"avatar_{user_id}.jpg"
            path = os.path.join(self._temp_dir, file_name)
            path = await download_image_by_url(url, path=path)
            if path:
                with open(path, "rb") as f:
                    return f.read()
        except Exception as e:
            logger.debug(f"[ImageGen] 获取头像失败 (user_id={user_id}): {e}")
        return None

    async def fetch_images_from_event(
        self,
        event: AstrMessageEvent,
        avatar_user_ids: set[str] | None = None,
    ) -> list[tuple[bytes, str]]:
        """从消息事件中提取图片（包括直接发送的图片、引用消息中的图片、被@用户的头像）。"""
        images_data: list[tuple[bytes, str]] = []
        if avatar_user_ids is None:
            avatar_user_ids = set()

        if not event.message_obj or not event.message_obj.message:
            return images_data

        for component in event.message_obj.message:
            try:
                if isinstance(component, Comp.Image):
                    # 处理直接发送的图片
                    url = component.url or component.file
                    if url and (data := await self.download_image(url)):
                        images_data.append(data)
                elif isinstance(component, Comp.Reply):
                    # 处理引用消息中的图片
                    if component.chain:
                        for sub_comp in component.chain:
                            if isinstance(sub_comp, Comp.Image):
                                url = sub_comp.url or sub_comp.file
                                if url and (data := await self.download_image(url)):
                                    images_data.append(data)
                elif isinstance(component, Comp.At):
                    # 处理 @ 用户的头像
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq)
                        if uid in avatar_user_ids:
                            continue
                        avatar_user_ids.add(uid)
                        if avatar_data := await self.get_avatar(uid):
                            images_data.append((avatar_data, "image/jpeg"))
            except Exception as e:
                logger.error(f"[ImageGen] 提取消息组件图片失败: {e}")
                continue
        return images_data

    def save_generated_image(self, task_id: str, img_bytes: bytes) -> str | None:
        """保存生成的图片到临时目录，返回文件路径。"""
        try:
            import time

            file_name = f"gen_{task_id}_{int(time.time())}_{hashlib.md5(img_bytes).hexdigest()[:6]}.png"
            file_path = os.path.join(self._temp_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            return file_path
        except Exception as exc:
            logger.error(f"[ImageGen] 保存图片失败: {exc}")
            return None
