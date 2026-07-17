from __future__ import annotations

import asyncio
from collections.abc import Mapping as MappingABC
from collections.abc import Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass, field
import math
import re
from time import perf_counter
from types import MappingProxyType
from typing import Any, Literal, Mapping, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from app.agent.actions import TOOL_ACTIONS
from app.api.monitor import EventEmitter
from app.config import OmniMatchSettings


SubAgentStatus = Literal["completed", "failed", "cancelled", "timed_out"]
SubAgentPayloadStatus = Literal["completed", "failed"]
_T = TypeVar("_T")


class _SubAgentDeadlineExceeded(Exception):
    pass


async def _await_before_deadline(
    operation: Callable[[], Awaitable[_T]],
    deadline: float,
) -> _T:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise _SubAgentDeadlineExceeded

    task = asyncio.ensure_future(operation())
    try:
        done, _ = await asyncio.wait({task}, timeout=remaining)
    except BaseException:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        raise
    if task in done:
        return task.result()

    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    raise _SubAgentDeadlineExceeded


def _validate_context_snapshot(value: Any, path: str = "$") -> None:
    if isinstance(value, MappingABC):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"context_snapshot at {path} keys must be strings")
            _validate_context_snapshot(item, f"{path}.{key}")
        return
    if isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _validate_context_snapshot(item, f"{path}[{index}]")
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"context_snapshot at {path} is not JSON-compatible: non-finite float")
    if value is None or isinstance(value, str | int | float | bool):
        return
    raise ValueError(f"context_snapshot at {path} is not JSON-compatible: {type(value).__name__}")


def _freeze_context(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze_context(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_context(item) for item in value)
    return value


def thaw_context_snapshot(value: Any) -> Any:
    """Return a mutable, serialization-friendly copy of a frozen context snapshot."""
    if isinstance(value, MappingABC):
        return {key: thaw_context_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_context_snapshot(item) for item in value]
    return value


@dataclass(frozen=True)
class AgentScope:
    depth: int = 0
    task_id: str | None = None
    allowed_tools: frozenset[str] | None = None
    emit_task_result: bool = True
    context_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_context_snapshot(self.context_snapshot)
        object.__setattr__(
            self, "context_snapshot", _freeze_context(deepcopy(dict(self.context_snapshot)))
        )


class ForkRequest(BaseModel):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    objective: str = Field(min_length=1, max_length=1000)
    allowed_tools: list[str] = Field(min_length=1)
    context_snapshot: dict[str, Any]
    max_steps: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    merge_key: str = Field(min_length=1, max_length=64)

    @field_validator("allowed_tools")
    @classmethod
    def validate_allowed_tools(cls, value: list[str]) -> list[str]:
        unknown = sorted(set(value) - TOOL_ACTIONS)
        if unknown:
            raise ValueError(f"unknown allowed tool: {', '.join(unknown)}")
        return list(dict.fromkeys(value))

    @field_validator("context_snapshot", mode="before")
    @classmethod
    def validate_context_snapshot(cls, value: Any) -> Any:
        _validate_context_snapshot(value)
        return value

    @classmethod
    def parse_many(
        cls,
        arguments: dict[str, Any],
        settings: OmniMatchSettings,
    ) -> list["ForkRequest"]:
        raw_tasks = arguments.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise ValueError("fork arguments.tasks must be a non-empty list")
        if len(raw_tasks) > settings.max_parallel_subagents:
            raise ValueError(
                f"fork task count exceeds max_parallel_subagents={settings.max_parallel_subagents}"
            )

        requests: list[ForkRequest] = []
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                raise ValueError("each fork task must be an object")
            request = cls.model_validate(
                {
                    **raw,
                    "max_steps": raw.get("max_steps", settings.subagent_max_steps),
                    "timeout_seconds": raw.get(
                        "timeout_seconds", settings.subagent_timeout_seconds
                    ),
                }
            )
            if request.max_steps > settings.subagent_max_steps:
                raise ValueError(
                    "fork max_steps exceeds settings.subagent_max_steps="
                    f"{settings.subagent_max_steps}"
                )
            if request.timeout_seconds > settings.subagent_timeout_seconds:
                raise ValueError(
                    "fork timeout_seconds exceeds settings.subagent_timeout_seconds="
                    f"{settings.subagent_timeout_seconds}"
                )
            requests.append(request)

        task_ids = [request.task_id for request in requests]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("fork task_id values must be unique")
        return requests


class SubAgentResult(BaseModel):
    task_id: str
    status: SubAgentStatus
    result: dict[str, Any] | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    step_count: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)


class SubAgentPayload(BaseModel):
    status: SubAgentPayloadStatus = "completed"
    result: dict[str, Any]
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    step_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_status_error(self) -> "SubAgentPayload":
        if self.status == "failed" and not self.error:
            raise ValueError("failed sub-agent payload requires an error")
        if self.status == "completed" and self.error is not None:
            raise ValueError("completed sub-agent payload cannot include an error")
        return self


SubAgentRunner = Callable[[ForkRequest], Awaitable[SubAgentPayload]]


def sanitize_subagent_error(error: object) -> str:
    text = re.sub(
        r"(?i)\b(bearer|basic)(\s+)\S+",
        r"\1\2[REDACTED]",
        str(error),
    )
    return re.sub(
        (
            r"(?i)\b((?:[a-z0-9]+[_-])*(?:api(?:[\s_-]?key)|access(?:[\s_-]?token)|"
            r"secret[_-]?access[_-]?key|client[_-]?secret|password|passwd|pwd|"
            r"authorization|auth(?:[\s_-]?token)?|token|secret))"
            r"(\s*[:=]\s*)(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|\S+)"
        ),
        r"\1\2[REDACTED]",
        text,
    )[:500]


class ForkExecutor:
    def __init__(self, monitor: EventEmitter, max_parallel: int) -> None:
        self.monitor = monitor
        self._semaphore = asyncio.Semaphore(max_parallel)

    async def execute(
        self,
        requests: list[ForkRequest],
        runner: SubAgentRunner,
    ) -> list[SubAgentResult]:
        tasks = [asyncio.create_task(self._run_one(request, runner)) for request in requests]
        try:
            results = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return sorted(results, key=lambda result: result.task_id)

    async def _run_one(
        self,
        request: ForkRequest,
        runner: SubAgentRunner,
    ) -> SubAgentResult:
        deadline = asyncio.get_running_loop().time() + request.timeout_seconds
        started = perf_counter()
        try:
            await _await_before_deadline(
                lambda: self.monitor.emit(
                    "subagent_started",
                    f"Sub-agent {request.task_id} started.",
                    tool="fork",
                    payload={
                        "subagent_id": request.task_id,
                        "objective": request.objective,
                    },
                ),
                deadline,
            )
            try:
                payload = await _await_before_deadline(
                    lambda: self._run_with_semaphore(request, runner),
                    deadline,
                )
                result = SubAgentResult(
                    task_id=request.task_id,
                    status=payload.status,
                    result=payload.result,
                    observations=payload.observations,
                    warnings=payload.warnings,
                    error=(
                        sanitize_subagent_error(payload.error)
                        if payload.error is not None
                        else None
                    ),
                    step_count=payload.step_count,
                    elapsed_ms=int((perf_counter() - started) * 1000),
                )
            except _SubAgentDeadlineExceeded:
                return self._timed_out_result(request, started)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                result = SubAgentResult(
                    task_id=request.task_id,
                    status="failed",
                    error=sanitize_subagent_error(exc),
                    elapsed_ms=int((perf_counter() - started) * 1000),
                )
            try:
                await _await_before_deadline(
                    lambda: self.monitor.emit(
                        "subagent_finished",
                        f"Sub-agent {request.task_id} finished with {result.status}.",
                        tool="fork",
                        payload=result.model_dump(),
                    ),
                    deadline,
                )
            except _SubAgentDeadlineExceeded:
                return result
            return result
        except _SubAgentDeadlineExceeded:
            return self._timed_out_result(request, started)
        except asyncio.CancelledError:
            try:
                await _await_before_deadline(
                    lambda: self.monitor.emit(
                        "subagent_cancelled",
                        f"Sub-agent {request.task_id} cancelled.",
                        tool="fork",
                        payload={"subagent_id": request.task_id},
                    ),
                    deadline,
                )
            except (_SubAgentDeadlineExceeded, Exception):
                pass
            raise

    @staticmethod
    def _timed_out_result(request: ForkRequest, started: float) -> SubAgentResult:
        return SubAgentResult(
            task_id=request.task_id,
            status="timed_out",
            error=f"sub-agent timed out after {request.timeout_seconds}s",
            elapsed_ms=int((perf_counter() - started) * 1000),
        )

    async def _run_with_semaphore(
        self,
        request: ForkRequest,
        runner: SubAgentRunner,
    ) -> SubAgentPayload:
        async with self._semaphore:
            return await runner(request)
