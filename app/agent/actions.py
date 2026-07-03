from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ToolActionName = Literal["plan", "category_insight", "item_search", "shipping", "rank", "pick"]
TerminalActionName = Literal["finish", "clarify", "fail"]
ActionName = ToolActionName | TerminalActionName

TOOL_ACTIONS: set[str] = {"plan", "category_insight", "item_search", "shipping", "rank", "pick"}
TERMINAL_ACTIONS: set[str] = {"finish", "clarify", "fail"}


@dataclass(frozen=True)
class AgentAction:
    name: ActionName
    arguments: dict[str, Any] = field(default_factory=dict)
    thought: str = ""
    message: str = ""

    @classmethod
    def from_provider_data(cls, data: dict[str, Any]) -> "AgentAction":
        raw_name = str(data.get("action") or "").strip()
        raw_arguments = data.get("arguments")
        arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
        thought = str(data.get("thought") or "")
        message = str(data.get("message") or "")

        if raw_name in TOOL_ACTIONS or raw_name in TERMINAL_ACTIONS:
            return cls(
                name=raw_name,  # type: ignore[arg-type]
                arguments=arguments,
                thought=thought,
                message=message,
            )

        return cls(
            name="fail",
            arguments={},
            thought=thought,
            message=f"unknown action from LLM provider: {raw_name or '<missing>'}",
        )

    @property
    def is_terminal(self) -> bool:
        return self.name in TERMINAL_ACTIONS


@dataclass(frozen=True)
class AgentStep:
    action: AgentAction
    observation_count: int
    observations: list[dict[str, Any]]
