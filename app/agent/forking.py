from __future__ import annotations

from collections.abc import Mapping as MappingABC
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from pydantic import BaseModel, Field, field_validator

from app.agent.actions import TOOL_ACTIONS
from app.config import OmniMatchSettings


SubAgentStatus = Literal["completed", "failed", "cancelled", "timed_out"]


def _freeze_context(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType({key: _freeze_context(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_context(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_context(item) for item in value)
    return value


def thaw_context_snapshot(value: Any) -> Any:
    """Return a mutable, serialization-friendly copy of a frozen context snapshot."""
    if isinstance(value, MappingABC):
        return {key: thaw_context_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_context_snapshot(item) for item in value]
    if isinstance(value, frozenset):
        return [thaw_context_snapshot(item) for item in value]
    return deepcopy(value)


@dataclass(frozen=True)
class AgentScope:
    depth: int = 0
    task_id: str | None = None
    allowed_tools: frozenset[str] | None = None
    emit_task_result: bool = True
    context_snapshot: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context_snapshot", _freeze_context(deepcopy(dict(self.context_snapshot))))


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
                "fork task count exceeds max_parallel_subagents="
                f"{settings.max_parallel_subagents}"
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
    result: dict[str, Any]
    observations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    step_count: int = Field(default=0, ge=0)
