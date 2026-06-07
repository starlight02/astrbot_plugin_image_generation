"""
用户使用数据管理模块
"""

from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

from astrbot.api import logger

from .constants import USAGE_DATA_RETENTION_DAYS
from .logging_utils import log_prefix

if TYPE_CHECKING:
    from .config_manager import UsageSettings


LOG = log_prefix("Usage")


class UsageManager:
    """用户使用数据管理器。"""

    def __init__(self, data_dir: str, settings: UsageSettings):
        self._data_dir = Path(data_dir)
        self._settings = settings
        self._usage_file = self._data_dir / "usage.json"
        self._usage_data: dict[str, dict[str, int]] = {}  # {date: {user_id: count}}
        self._usage_reservations: dict[str, dict[str, int]] = {}
        self._user_request_timestamps: dict[str, float] = {}  # 用于频率限制
        self._load_usage_data()

    def update_settings(self, settings: UsageSettings) -> None:
        """更新设置。"""
        self._settings = settings

    def _load_usage_data(self) -> None:
        """加载用户使用数据。"""
        if self._usage_file.exists():
            try:
                with self._usage_file.open(encoding="utf-8") as f:
                    self._usage_data = json.load(f)

                # 清理旧数据，只保留最近 N 天（由 USAGE_DATA_RETENTION_DAYS 控制）
                today = datetime.date.today()
                keys_to_delete = []
                for date_str in self._usage_data:
                    try:
                        date_obj = datetime.date.fromisoformat(date_str)
                        if (today - date_obj).days > USAGE_DATA_RETENTION_DAYS:
                            keys_to_delete.append(date_str)
                    except ValueError:
                        keys_to_delete.append(date_str)

                if keys_to_delete:
                    for key in keys_to_delete:
                        del self._usage_data[key]
                    self._save_usage_data()
            except Exception as exc:
                logger.error(f"{LOG} 加载使用数据失败: {exc}", exc_info=True)
                self._usage_data = {}

    def _save_usage_data(self) -> None:
        """保存用户使用数据。"""
        try:
            self._usage_file.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_file.open("w", encoding="utf-8") as f:
                json.dump(self._usage_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"{LOG} 保存使用数据失败: {exc}", exc_info=True)

    def _today(self) -> str:
        """Return today's date key for usage accounting."""
        return datetime.date.today().isoformat()

    def _get_reserved_usage(self, date: str, user_id: str) -> int:
        """Return pending reserved usage for one user."""
        return self._usage_reservations.get(date, {}).get(user_id, 0)

    def _reserve_usage(self, date: str, user_id: str, count: int) -> None:
        """Reserve quota before an async generation task starts."""
        count = max(0, count)
        if count <= 0:
            return
        self._usage_reservations.setdefault(date, {})[user_id] = (
            self._get_reserved_usage(date, user_id) + count
        )

    def is_session_blocked(self, user_id: str) -> bool:
        """Check whether the current session UMO is blocked."""
        uid = user_id.strip()
        if not uid:
            return False
        return uid in self._settings.umo_blacklist

    def is_limit_exempt(self, user_id: str, *, is_admin: bool = False) -> bool:
        """Check whether the current request should bypass usage limits."""
        uid = user_id.strip()
        if not uid:
            return False
        if self._settings.admin_bypass_limits and is_admin:
            return True
        return uid in self._settings.umo_whitelist

    def check_rate_limit(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        requested_count: int = 1,
        update_timestamp: bool = True,
    ) -> bool | str:
        """检查用户请求频率限制和每日限制。

        返回:
            - True: 检查通过
            - str: 错误消息

        Args:
            update_timestamp: Whether to reserve the cooldown when checks pass.
        """
        user_id = str(user_id or "").strip()

        # 1. 检查频率限制
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return True

        if self.is_session_blocked(user_id):
            return self._settings.blacklist_block_message

        if self._settings.rate_limit_seconds > 0:
            now = time.time()
            last_ts = self._user_request_timestamps.get(user_id, 0)
            if now - last_ts < self._settings.rate_limit_seconds:
                remaining = int(self._settings.rate_limit_seconds - (now - last_ts))
                return f"❌ 请求过于频繁，请在 {remaining} 秒后再试"
            if update_timestamp:
                self._user_request_timestamps[user_id] = now

        # 2. 检查每日限制
        if self._settings.enable_daily_limit:
            requested_count = max(1, requested_count)
            today = self._today()
            if today not in self._usage_data:
                self._usage_data[today] = {}

            count = self._usage_data[today].get(user_id, 0)
            reserved_count = self._get_reserved_usage(today, user_id)
            if (
                count + reserved_count + requested_count
                > self._settings.daily_limit_count
            ):
                remaining = max(
                    0,
                    self._settings.daily_limit_count - count - reserved_count,
                )
                return (
                    f"❌ 今日剩余生图额度不足，剩余 {remaining} 张，"
                    f"本次请求 {requested_count} 张"
                )
            if update_timestamp:
                self._reserve_usage(today, user_id, requested_count)

        return True

    def record_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        count: int = 1,
    ) -> None:
        """记录用户使用次数。"""
        if not self._settings.enable_daily_limit:
            return
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return

        count = max(1, count)
        today = self._today()
        if today not in self._usage_data:
            self._usage_data[today] = {}
        self._usage_data[today][user_id] = (
            self._usage_data[today].get(user_id, 0) + count
        )
        self._save_usage_data()

    def release_reserved_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        count: int = 1,
    ) -> None:
        """Release previously reserved quota without recording usage."""
        if not self._settings.enable_daily_limit:
            return
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return

        user_id = str(user_id or "").strip()
        if not user_id:
            return

        today = self._today()
        reservations = self._usage_reservations.get(today)
        if not reservations or user_id not in reservations:
            return

        remaining = max(0, reservations[user_id] - max(0, count))
        if remaining:
            reservations[user_id] = remaining
            return

        del reservations[user_id]
        if not reservations:
            del self._usage_reservations[today]

    def settle_usage(
        self,
        user_id: str,
        *,
        is_admin: bool = False,
        reserved_count: int = 0,
        actual_count: int = 0,
    ) -> None:
        """Record actual usage and release the matching reservation."""
        if not self._settings.enable_daily_limit:
            return
        if self.is_limit_exempt(user_id, is_admin=is_admin):
            return

        user_id = str(user_id or "").strip()
        if not user_id:
            return

        actual_count = max(0, actual_count)
        if actual_count:
            today = self._today()
            if today not in self._usage_data:
                self._usage_data[today] = {}
            self._usage_data[today][user_id] = (
                self._usage_data[today].get(user_id, 0) + actual_count
            )
            self._save_usage_data()

        self.release_reserved_usage(
            user_id,
            is_admin=is_admin,
            count=max(0, reserved_count),
        )

    def get_usage_count(self, user_id: str) -> int:
        """获取用户今日使用次数。"""
        today = self._today()
        return self._usage_data.get(today, {}).get(user_id, 0)

    def get_daily_limit(self) -> int:
        """获取每日限制次数。"""
        return self._settings.daily_limit_count

    def is_daily_limit_enabled(self) -> bool:
        """是否启用每日限制。"""
        return self._settings.enable_daily_limit
