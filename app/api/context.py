from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path


current_thread_id: ContextVar[str | None] = ContextVar("current_thread_id", default=None)
current_session_dir: ContextVar[Path | None] = ContextVar("current_session_dir", default=None)


def set_task_context(thread_id: str, session_dir: Path) -> None:
    current_thread_id.set(thread_id)
    current_session_dir.set(session_dir)


def get_thread_id() -> str | None:
    return current_thread_id.get()


def get_session_dir() -> Path | None:
    return current_session_dir.get()
