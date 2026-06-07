from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from astrbot.api import logger

from .logging_utils import (
    format_optional,
    format_seconds,
    log_prefix,
    mask_sensitive,
    safe_log_text,
)


LOG = log_prefix("TaskManager")
DEFAULT_GENERATION_TASK_HISTORY_LIMIT = 100


class GenerationTaskStatus(str, Enum):
    """Lifecycle states for image generation tasks."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"


ACTIVE_GENERATION_STATUSES = {
    GenerationTaskStatus.QUEUED,
    GenerationTaskStatus.RUNNING,
    GenerationTaskStatus.CANCELLING,
}

GENERATION_TASK_STATUS_LABELS = {
    GenerationTaskStatus.QUEUED: "排队中",
    GenerationTaskStatus.RUNNING: "运行中",
    GenerationTaskStatus.SUCCEEDED: "已完成",
    GenerationTaskStatus.FAILED: "失败",
    GenerationTaskStatus.CANCELLING: "取消中",
    GenerationTaskStatus.CANCELLED: "已取消",
}


@dataclass
class GenerationTaskItem:
    """Per-request generation result metadata."""

    index: int
    status: str = "pending"
    result_count: int = 0
    error: str = ""
    retry_attempts: int = 0
    max_retry_attempts: int = 0


@dataclass
class GenerationTaskRecord:
    """In-memory metadata for one image generation task."""

    task_id: str
    source: str
    unified_msg_origin: str
    prompt_summary: str
    reference_image_count: int
    requested_count: int
    aspect_ratio: str
    resolution: str
    preset: str | None = None
    preset_label: str = "预设"
    status: GenerationTaskStatus = GenerationTaskStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = "任务已提交"
    error: str = ""
    result_count: int = 0
    result_paths: list[str] = field(default_factory=list)
    current_index: int = 0
    retry_attempt: int = 0
    max_retry_attempts: int = 0
    items: dict[int, GenerationTaskItem] = field(default_factory=dict)
    task: asyncio.Task | None = field(default=None, repr=False, compare=False)

    @property
    def is_active(self) -> bool:
        """Return whether the task can still change state."""
        return self.status in ACTIVE_GENERATION_STATUSES

    @property
    def status_label(self) -> str:
        """Return a user-facing status label."""
        return GENERATION_TASK_STATUS_LABELS.get(self.status, self.status.value)

    @property
    def duration_seconds(self) -> float | None:
        """Return active execution duration, excluding queued time."""
        if not self.started_at:
            return None
        end_time = self.finished_at or datetime.now()
        return max(0.0, (end_time - self.started_at).total_seconds())

    @property
    def queued_seconds(self) -> float:
        """Return time spent since creation."""
        end_time = self.started_at or self.finished_at or datetime.now()
        return max(0.0, (end_time - self.created_at).total_seconds())


def _task_name(name: str) -> str:
    """Return a compact task name for logs."""
    return safe_log_text(name, 80)


def _task_elapsed(record: GenerationTaskRecord) -> str:
    """Return a compact task timing summary for logs."""
    queued = record.queued_seconds
    duration = record.duration_seconds
    if duration is None:
        return f"排队={format_seconds(queued)}"
    return f"排队={format_seconds(queued)}，耗时={format_seconds(duration)}"


def _task_creation_summary(record: GenerationTaskRecord) -> str:
    """Return a compact creation summary for generation task logs."""
    return (
        f"来源={safe_log_text(record.source)}，"
        f"用户={mask_sensitive(record.unified_msg_origin)}，"
        f"数量={record.requested_count}张，"
        f"参考图={record.reference_image_count}张，"
        f"{record.preset_label}={format_optional(record.preset)}，"
        f"宽高比={safe_log_text(record.aspect_ratio)}，"
        f"分辨率={safe_log_text(record.resolution)}"
    )


class TaskManager:
    """Unified task manager for background, scheduled, and generation tasks."""

    def __init__(
        self, generation_history_limit: int = DEFAULT_GENERATION_TASK_HISTORY_LIMIT
    ):
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_tasks: dict[str, asyncio.Task] = {}
        self._daily_tasks: dict[str, asyncio.Task] = {}
        self._last_run_dates: dict[str, str] = {}
        self._startup_tasks: list[
            tuple[str, Callable[[], Coroutine[Any, Any, Any]]]
        ] = []
        self._startup_completed: bool = False
        self._generation_tasks: dict[str, GenerationTaskRecord] = {}
        self._generation_history_limit = max(1, generation_history_limit)

    def create_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """Create a generic background task."""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def create_generation_task(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        task_id: str,
        source: str,
        unified_msg_origin: str,
        prompt: str,
        reference_image_count: int,
        requested_count: int,
        aspect_ratio: str,
        resolution: str,
        preset: str | None = None,
        preset_label: str = "预设",
    ) -> GenerationTaskRecord:
        """Create and track an image generation task."""
        if task_id in self._generation_tasks:
            logger.warning(f"{LOG} 生图任务 ID 冲突，覆盖旧记录: {_task_name(task_id)}")

        record = GenerationTaskRecord(
            task_id=task_id,
            source=source,
            unified_msg_origin=unified_msg_origin,
            prompt_summary=safe_log_text(prompt, 80),
            reference_image_count=reference_image_count,
            requested_count=max(1, requested_count),
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            preset=preset,
            preset_label=preset_label,
        )
        self._generation_tasks[task_id] = record
        self._trim_generation_history()

        task = asyncio.create_task(
            self._run_generation_task(task_id, coro),
            name=f"image_generation:{task_id}",
        )
        record.task = task
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        task.add_done_callback(
            functools.partial(self._on_generation_task_done, task_id)
        )
        logger.info(
            f"{log_prefix('Task', task_id)} 已创建生图任务: "
            f"{_task_creation_summary(record)}"
        )
        logger.debug(
            f"{log_prefix('Task', task_id)} 生图任务提示词摘要: "
            f"提示词={safe_log_text(prompt, 80)}"
        )
        return record

    async def _run_generation_task(
        self, task_id: str, coro: Coroutine[Any, Any, Any]
    ) -> None:
        """Run a tracked generation coroutine and close unhandled states."""
        try:
            await coro
            record = self.get_generation_task(task_id)
            if record and record.is_active:
                self.mark_generation_task_succeeded(task_id, message="任务已完成")
        except asyncio.CancelledError:
            self.mark_generation_task_cancelled(task_id, "任务已取消")
            raise
        except Exception as exc:
            self.mark_generation_task_failed(task_id, f"任务执行异常: {exc}")
            logger.error(
                f"{log_prefix('Task', task_id)} 生图任务执行异常: {exc}",
                exc_info=True,
            )

    def _on_generation_task_done(self, task_id: str, _task: asyncio.Task) -> None:
        """Detach asyncio task references when a generation task finishes."""
        if record := self._generation_tasks.get(task_id):
            record.task = None

    def mark_generation_task_running(self, task_id: str) -> None:
        """Mark a generation task as actively running."""
        record = self._generation_tasks.get(task_id)
        if not record or record.status == GenerationTaskStatus.CANCELLING:
            return
        record.status = GenerationTaskStatus.RUNNING
        record.started_at = record.started_at or datetime.now()
        record.message = "任务运行中"
        logger.debug(
            f"{log_prefix('Task', task_id)} 生图任务开始运行: "
            f"排队={format_seconds(record.queued_seconds)}"
        )

    def update_generation_task_references(
        self,
        task_id: str,
        *,
        reference_image_count: int,
    ) -> None:
        """Update prepared reference image metadata for a generation task."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.reference_image_count = max(0, reference_image_count)

    def update_generation_task_retry_status(
        self,
        task_id: str,
        *,
        current_index: int,
        retry_attempt: int,
        max_retry_attempts: int,
    ) -> None:
        """Update the currently running generation retry attempt."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.current_index = max(1, current_index)
        record.retry_attempt = max(0, retry_attempt)
        record.max_retry_attempts = max(0, max_retry_attempts)
        item = record.items.setdefault(
            record.current_index,
            GenerationTaskItem(index=record.current_index, status="running"),
        )
        item.status = "running"
        item.retry_attempts = max(item.retry_attempts, record.retry_attempt)
        item.max_retry_attempts = max(
            item.max_retry_attempts, record.max_retry_attempts
        )

    def update_generation_task_item_result(
        self,
        task_id: str,
        *,
        index: int,
        status: str,
        result_count: int = 0,
        error: str = "",
    ) -> None:
        """Record per-request generation result details."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        safe_index = max(1, index)
        item = record.items.setdefault(
            safe_index,
            GenerationTaskItem(index=safe_index),
        )
        item.status = status
        item.result_count = max(0, result_count)
        item.error = safe_log_text(error, 200) if error else ""

    def update_generation_task_progress(
        self,
        task_id: str,
        *,
        current_index: int,
        result_count: int,
        message: str,
    ) -> None:
        """Update image count progress for one generation task."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.current_index = max(1, current_index)
        record.result_count = max(0, result_count)
        record.message = message

    def mark_generation_task_succeeded(
        self,
        task_id: str,
        *,
        result_count: int = 0,
        result_paths: list[str] | None = None,
        message: str = "任务已完成",
    ) -> None:
        """Mark a generation task as successful."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.status = GenerationTaskStatus.SUCCEEDED
        record.finished_at = record.finished_at or datetime.now()
        record.message = message
        record.error = ""
        record.result_count = result_count or record.result_count
        if result_paths is not None:
            record.result_paths = list(result_paths)
        logger.info(
            f"{log_prefix('Task', task_id)} 生图任务完成: "
            f"来源={safe_log_text(record.source)}，{_task_elapsed(record)}，"
            f"结果={record.result_count}张"
        )

    def mark_generation_task_failed(self, task_id: str, error: str) -> None:
        """Mark a generation task as failed."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.status = GenerationTaskStatus.FAILED
        record.finished_at = record.finished_at or datetime.now()
        record.error = safe_log_text(error, 300)
        record.message = "任务失败"
        logger.warning(
            f"{log_prefix('Task', task_id)} 生图任务失败: "
            f"{_task_elapsed(record)}，错误={record.error}"
        )

    def mark_generation_task_cancelled(
        self, task_id: str, reason: str = "任务已取消"
    ) -> None:
        """Mark a generation task as cancelled."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return
        record.status = GenerationTaskStatus.CANCELLED
        record.finished_at = record.finished_at or datetime.now()
        record.message = reason
        logger.info(
            f"{log_prefix('Task', task_id)} 生图任务已取消: "
            f"{_task_elapsed(record)}，原因={format_optional(reason)}"
        )

    def get_generation_task(self, task_id: str) -> GenerationTaskRecord | None:
        """Return a tracked image generation task by id."""
        return self._generation_tasks.get(task_id)

    def list_generation_tasks(
        self,
        *,
        unified_msg_origin: str | None = None,
        include_finished: bool = True,
        limit: int = 10,
    ) -> list[GenerationTaskRecord]:
        """List tracked generation tasks from newest to oldest."""
        tasks = list(reversed(self._generation_tasks.values()))
        if unified_msg_origin is not None:
            tasks = [t for t in tasks if t.unified_msg_origin == unified_msg_origin]
        if not include_finished:
            tasks = [t for t in tasks if t.is_active]
        return tasks[: max(1, limit)]

    def cancel_generation_task(
        self,
        task_id: str,
        *,
        unified_msg_origin: str | None = None,
    ) -> tuple[bool, str]:
        """Request cancellation for one generation task."""
        record = self._generation_tasks.get(task_id)
        if not record:
            return False, f"❌ 任务不存在: {task_id}"
        if (
            unified_msg_origin is not None
            and record.unified_msg_origin != unified_msg_origin
        ):
            return False, "❌ 不能取消其他会话的生图任务"
        if not record.is_active:
            return False, f"❌ 任务已结束，当前状态: {record.status_label}"

        record.status = GenerationTaskStatus.CANCELLING
        record.message = "正在取消任务"
        logger.debug(f"{log_prefix('Task', task_id)} 收到取消生图任务请求")
        if record.task and not record.task.done():
            record.task.cancel()
            return True, f"✅ 已请求取消任务: {task_id}"

        self.mark_generation_task_cancelled(task_id)
        return True, f"✅ 任务已取消: {task_id}"

    def cleanup_generation_tasks(self, *, unified_msg_origin: str | None = None) -> int:
        """Remove finished generation task records."""
        removed = 0
        for task_id, record in list(self._generation_tasks.items()):
            if record.is_active:
                continue
            if (
                unified_msg_origin is not None
                and record.unified_msg_origin != unified_msg_origin
            ):
                continue
            del self._generation_tasks[task_id]
            removed += 1
        return removed

    def _trim_generation_history(self) -> None:
        """Keep finished task history bounded while preserving active tasks."""
        overflow = len(self._generation_tasks) - self._generation_history_limit
        if overflow <= 0:
            return

        for task_id, record in list(self._generation_tasks.items()):
            if overflow <= 0:
                break
            if record.is_active:
                continue
            del self._generation_tasks[task_id]
            overflow -= 1

    def start_loop_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        run_immediately: bool = True,
    ) -> None:
        """启动一个周期性的定时任务。

        Args:
            name: 任务名称，用于唯一标识和日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
            interval_seconds: 执行间隔（秒）。
            run_immediately: 是否在启动时立即执行一次。
        """
        if name in self._loop_tasks:
            self.stop_loop_task(name)

        log_name = _task_name(name)

        async def _loop():
            if run_immediately:
                try:
                    await coro_func()
                except Exception as e:
                    logger.error(
                        f"{LOG} 定时任务 {log_name} 初始执行失败: {e}",
                        exc_info=True,
                    )

            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await coro_func()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"{LOG} 定时任务 {log_name} 执行出错: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_loop(), name=f"loop_{name}")
        self._loop_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_loop_task_done, name))
        logger.debug(f"{LOG} 定时任务 {log_name} 已启动 (间隔: {interval_seconds}s)")

    def stop_loop_task(self, name: str) -> None:
        """停止指定的定时任务。"""
        if task := self._loop_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            logger.debug(f"{LOG} 定时任务 {_task_name(name)} 已停止")

    def _on_loop_task_done(self, name: str, task: asyncio.Task) -> None:
        """定时任务结束时的回调。"""
        self.background_tasks.discard(task)
        self._loop_tasks.pop(name, None)

    def register_startup_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """注册一个启动时执行的任务。

        Args:
            name: 任务名称，用于日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
        """
        self._startup_tasks.append((name, coro_func))
        logger.debug(f"{LOG} 已注册启动任务: {_task_name(name)}")

    async def run_startup_tasks(self) -> None:
        """执行所有注册的启动任务。

        此方法应在插件初始化完成后调用一次。
        """
        if self._startup_completed:
            logger.warning(f"{LOG} 启动任务已执行过，跳过重复执行")
            return

        if not self._startup_tasks:
            logger.debug(f"{LOG} 没有注册的启动任务")
            self._startup_completed = True
            return

        logger.debug(f"{LOG} 开始执行 {len(self._startup_tasks)} 个启动任务")

        for name, coro_func in self._startup_tasks:
            log_name = _task_name(name)
            try:
                logger.debug(f"{LOG} 执行启动任务: {log_name}")
                await coro_func()
                logger.debug(f"{LOG} 启动任务 {log_name} 执行完成")
            except Exception as e:
                logger.error(
                    f"{LOG} 启动任务 {log_name} 执行失败: {e}",
                    exc_info=True,
                )

        self._startup_completed = True
        logger.debug(f"{LOG} 所有启动任务执行完毕")

    def start_daily_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        check_interval_seconds: float = 60.0,
        run_immediately: bool = False,
    ) -> None:
        """启动一个每日任务，在日期变更时执行。

        Args:
            name: 任务名称，用于唯一标识和日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
            check_interval_seconds: 检查日期变更的间隔（秒），默认 60 秒。
            run_immediately: 是否在启动时立即执行一次（无论日期）。
        """
        if name in self._daily_tasks:
            self.stop_daily_task(name)

        log_name = _task_name(name)

        async def _daily_loop():
            # 初始化上次执行日期
            if run_immediately:
                try:
                    await coro_func()
                    self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")
                    logger.debug(f"{LOG} 每日任务 {log_name} 初始执行完成")
                except Exception as e:
                    logger.error(
                        f"{LOG} 每日任务 {log_name} 初始执行失败: {e}",
                        exc_info=True,
                    )
            else:
                # 记录当前日期，避免启动当天重复执行
                self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")

            while True:
                try:
                    await asyncio.sleep(check_interval_seconds)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    last_run_date = self._last_run_dates.get(name)

                    if current_date != last_run_date:
                        logger.info(
                            f"{LOG} 检测到日期变更 ({last_run_date} -> {current_date})，执行每日任务 {log_name}"
                        )
                        try:
                            await coro_func()
                            self._last_run_dates[name] = current_date
                            logger.info(f"{LOG} 每日任务 {log_name} 执行完成")
                        except Exception as e:
                            logger.error(
                                f"{LOG} 每日任务 {log_name} 执行出错: {e}",
                                exc_info=True,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"{LOG} 每日任务 {log_name} 循环出错: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_daily_loop(), name=f"daily_{name}")
        self._daily_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_daily_task_done, name))
        logger.debug(
            f"{LOG} 每日任务 {log_name} 已启动 (检查间隔: {check_interval_seconds}s)"
        )

    def stop_daily_task(self, name: str) -> None:
        """停止指定的每日任务。"""
        if task := self._daily_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            self._last_run_dates.pop(name, None)
            logger.debug(f"{LOG} 每日任务 {_task_name(name)} 已停止")

    def _on_daily_task_done(self, name: str, task: asyncio.Task) -> None:
        """每日任务结束时的回调。"""
        self.background_tasks.discard(task)
        self._daily_tasks.pop(name, None)
        self._last_run_dates.pop(name, None)

    async def cancel_all(self):
        """取消所有正在运行的任务。"""
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self._loop_tasks.clear()
        self._daily_tasks.clear()
        self._last_run_dates.clear()
        logger.debug(f"{LOG} 所有后台任务已取消")
