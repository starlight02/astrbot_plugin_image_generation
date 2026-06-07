"""Task ID generation helpers."""

from __future__ import annotations

import secrets


def new_task_id() -> str:
    """Return a short random task id for user-facing task tracking."""
    return secrets.token_hex(4)
