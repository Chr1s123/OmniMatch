from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeVar


T = TypeVar("T")
ProviderMode = Literal["real", "placeholder", "fake"]


class ProviderError(RuntimeError):
    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        super().__init__(message)


@dataclass
class ProviderResult(Generic[T]):
    provider: str
    provider_mode: ProviderMode
    latency_ms: int
    data: T
    warnings: list[str] = field(default_factory=list)
    response_summary: str = ""

    def redacted_summary(self) -> str:
        text = re.sub(
            r"Bearer\s+[A-Za-z0-9._-]+",
            "Bearer [REDACTED]",
            self.response_summary,
        )
        return re.sub(r"api_key=([^&\s]+)", "api_key=[REDACTED]", text, flags=re.IGNORECASE)


class LLMProvider(Protocol):
    async def plan_next_action(self, messages: list[dict[str, Any]]) -> ProviderResult[dict[str, Any]]:
        ...


class ProductSearchProvider(Protocol):
    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        ...


class WebSearchProvider(Protocol):
    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        ...


class ShippingProvider(Protocol):
    async def estimate(
        self,
        product: dict[str, Any],
        destination: str | None,
    ) -> ProviderResult[dict[str, Any]]:
        ...
