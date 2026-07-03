# Competition Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert OmniMatch from a mock teaching demo into a competition-grade shopping agent that uses real APIs in development and deterministic placeholder providers for upload/submission runs.

**Architecture:** Add a typed config layer and provider registry first, then move tools behind provider interfaces. Replace the fixed mock AgentLoop with an observation-driven loop, add structured ranking and trace persistence, then expose provider/profile observability through the existing FastAPI/WebSocket/React shell.

**Tech Stack:** Python 3.10, uv, FastAPI, Pydantic, httpx, pytest, pytest-asyncio, React, Vite, TypeScript, plain CSS.

## Global Constraints

- Backend uses `uv` with Python 3.10.
- Frontend uses Vite, React, TypeScript, and plain CSS.
- Real external APIs are the default in `OMNIMATCH_PROFILE=dev`.
- `OMNIMATCH_PROFILE=submission` must run without real secrets using deterministic placeholder providers.
- `OMNIMATCH_PROFILE=test` must not call network APIs.
- `.env` may contain local real keys and must not be committed.
- `.env.example` documents every required setting with empty or placeholder values.
- Tools call provider interfaces; API-specific code lives in `app/providers`.
- Trace logs redact credentials and auth headers.
- This plan describes future implementation work. The current turn only changes documentation.

---

## File Structure

- `app/config.py`: typed settings, profile validation, provider key validation.
- `app/providers/base.py`: provider protocols, result envelopes, provider errors.
- `app/providers/registry.py`: builds providers from settings.
- `app/providers/placeholder.py`: deterministic submission/test providers.
- `app/providers/openai_llm.py`: real LLM adapter selected by config.
- `app/providers/http_product.py`: configurable real product/search API adapter.
- `app/providers/http_web_search.py`: configurable real web search API adapter.
- `app/providers/shipping.py`: real rate-table and placeholder shipping adapters.
- `app/ranking/scorer.py`: deterministic candidate scoring and rejection reasons.
- `app/agent/tool_registry.py`: maps LLM actions to tool functions.
- `app/agent/main_agent.py`: observation-driven AgentLoop.
- `app/eval/cases.py`: evaluation case schema and loader.
- `app/eval/runner.py`: repeatable evaluation runner.
- `app/eval/fixtures/competition_smoke.jsonl`: small regression dataset.
- `frontend/src/App.tsx`: profile/provider/trace observability.
- `tests/test_config.py`: config profile and key validation.
- `tests/test_providers.py`: provider contracts and redaction.
- `tests/test_ranking.py`: deterministic scoring behavior.
- `tests/test_agent_loop.py`: fake-provider AgentLoop behavior.
- `tests/test_api.py`: task state, WebSocket, and failure hardening.
- `tests/test_eval_runner.py`: evaluation output contract.

---

### Task 1: Typed Config And Profiles

**Files:**
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Create: `app/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Profile = Literal["dev", "submission", "test"]`
- Produces: `ProviderMode = Literal["real", "placeholder", "fake"]`
- Produces: `OmniMatchSettings.from_env() -> OmniMatchSettings`
- Produces: `OmniMatchSettings.provider_modes() -> dict[str, ProviderMode]`
- Produces: `ConfigError`

- [ ] **Step 1: Write failing config tests**

```python
import pytest

from app.config import ConfigError, OmniMatchSettings


def clear_omnimatch_env(monkeypatch):
    for key in list(__import__("os").environ):
        if key.startswith("OMNIMATCH_") or key.endswith("_API_KEY"):
            monkeypatch.delenv(key, raising=False)


def test_dev_profile_requires_real_provider_keys(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "dev")
    monkeypatch.setenv("OMNIMATCH_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OMNIMATCH_LLM_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "http_product")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "http_web_search")
    monkeypatch.setenv("OMNIMATCH_SHIPPING_PROVIDER", "rate_table")
    monkeypatch.setenv("OMNIMATCH_MEMORY_PROVIDER", "memory")
    monkeypatch.setenv("OMNIMATCH_EVAL_PROVIDER", "heuristic")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OmniMatchSettings.from_env()


def test_submission_defaults_to_placeholder_without_keys(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "submission"
    assert settings.llm_provider == "placeholder"
    assert settings.product_provider == "placeholder"
    assert settings.web_search_provider == "placeholder"
    assert settings.provider_modes()["llm"] == "placeholder"


def test_test_profile_uses_fake_or_placeholder_without_network(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "test")

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "test"
    assert set(settings.provider_modes().values()) <= {"fake", "placeholder"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`

Expected: fails because `app.config` does not exist.

- [ ] **Step 3: Add runtime dependencies and update the lockfile**

Modify `pyproject.toml` so runtime dependencies include `httpx` and
`python-dotenv`:

```toml
dependencies = [
    "fastapi>=0.116.0",
    "httpx>=0.27.0",
    "pydantic>=2.8.0",
    "python-dotenv>=1.0.0",
    "uvicorn[standard]>=0.30.0",
]
```

Run: `uv sync`

Expected: `uv.lock` is updated and the environment contains `python-dotenv`.

- [ ] **Step 4: Implement `app/config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


Profile = Literal["dev", "submission", "test"]
ProviderMode = Literal["real", "placeholder", "fake"]


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class OmniMatchSettings:
    profile: Profile
    llm_provider: str
    llm_model: str
    product_provider: str
    web_search_provider: str
    shipping_provider: str
    memory_provider: str
    eval_provider: str
    product_api_url: str | None = None
    web_search_api_url: str | None = None

    @classmethod
    def from_env(cls) -> "OmniMatchSettings":
        load_dotenv()
        profile = os.getenv("OMNIMATCH_PROFILE", "dev")
        if profile not in {"dev", "submission", "test"}:
            raise ConfigError("OMNIMATCH_PROFILE must be dev, submission, or test")

        if profile == "submission":
            settings = cls(
                profile="submission",
                llm_provider=os.getenv("OMNIMATCH_LLM_PROVIDER", "placeholder"),
                llm_model=os.getenv("OMNIMATCH_LLM_MODEL", "placeholder-llm"),
                product_provider=os.getenv("OMNIMATCH_PRODUCT_PROVIDER", "placeholder"),
                web_search_provider=os.getenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "placeholder"),
                shipping_provider=os.getenv("OMNIMATCH_SHIPPING_PROVIDER", "placeholder"),
                memory_provider=os.getenv("OMNIMATCH_MEMORY_PROVIDER", "placeholder"),
                eval_provider=os.getenv("OMNIMATCH_EVAL_PROVIDER", "placeholder"),
            )
        elif profile == "test":
            settings = cls(
                profile="test",
                llm_provider=os.getenv("OMNIMATCH_LLM_PROVIDER", "placeholder"),
                llm_model=os.getenv("OMNIMATCH_LLM_MODEL", "fake-llm"),
                product_provider=os.getenv("OMNIMATCH_PRODUCT_PROVIDER", "placeholder"),
                web_search_provider=os.getenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "placeholder"),
                shipping_provider=os.getenv("OMNIMATCH_SHIPPING_PROVIDER", "placeholder"),
                memory_provider=os.getenv("OMNIMATCH_MEMORY_PROVIDER", "memory"),
                eval_provider=os.getenv("OMNIMATCH_EVAL_PROVIDER", "heuristic"),
            )
        else:
            settings = cls(
                profile="dev",
                llm_provider=os.getenv("OMNIMATCH_LLM_PROVIDER", "openai"),
                llm_model=os.getenv("OMNIMATCH_LLM_MODEL", "gpt-4.1-mini"),
                product_provider=os.getenv("OMNIMATCH_PRODUCT_PROVIDER", "http_product"),
                web_search_provider=os.getenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "http_web_search"),
                shipping_provider=os.getenv("OMNIMATCH_SHIPPING_PROVIDER", "rate_table"),
                memory_provider=os.getenv("OMNIMATCH_MEMORY_PROVIDER", "memory"),
                eval_provider=os.getenv("OMNIMATCH_EVAL_PROVIDER", "heuristic"),
                product_api_url=os.getenv("OMNIMATCH_PRODUCT_API_URL"),
                web_search_api_url=os.getenv("OMNIMATCH_WEB_SEARCH_API_URL"),
            )
        settings.validate()
        return settings

    def provider_modes(self) -> dict[str, ProviderMode]:
        return {
            "llm": self._mode_for(self.llm_provider, fake_allowed=self.profile == "test"),
            "product": self._mode_for(self.product_provider, fake_allowed=self.profile == "test"),
            "web_search": self._mode_for(self.web_search_provider, fake_allowed=self.profile == "test"),
            "shipping": self._mode_for(self.shipping_provider, fake_allowed=self.profile == "test"),
            "memory": self._mode_for(self.memory_provider, fake_allowed=self.profile == "test"),
            "eval": self._mode_for(self.eval_provider, fake_allowed=self.profile == "test"),
        }

    def validate(self) -> None:
        if self.profile != "dev":
            return
        if self.llm_provider != "placeholder" and not os.getenv("OPENAI_API_KEY"):
            raise ConfigError("OPENAI_API_KEY is required for dev LLM provider")
        if self.product_provider != "placeholder" and not self.product_api_url:
            raise ConfigError("OMNIMATCH_PRODUCT_API_URL is required for dev product provider")
        if self.web_search_provider != "placeholder" and not self.web_search_api_url:
            raise ConfigError("OMNIMATCH_WEB_SEARCH_API_URL is required for dev web search provider")

    @staticmethod
    def _mode_for(provider: str, fake_allowed: bool) -> ProviderMode:
        if provider == "placeholder":
            return "placeholder"
        if fake_allowed:
            return "fake"
        return "real"
```

- [ ] **Step 5: Update `.env.example`**

```dotenv
OMNIMATCH_PROFILE=dev
OMNIMATCH_LLM_PROVIDER=openai
OMNIMATCH_LLM_MODEL=gpt-4.1-mini
OPENAI_API_KEY=

OMNIMATCH_PRODUCT_PROVIDER=http_product
OMNIMATCH_PRODUCT_API_URL=
OMNIMATCH_PRODUCT_API_KEY=

OMNIMATCH_WEB_SEARCH_PROVIDER=http_web_search
OMNIMATCH_WEB_SEARCH_API_URL=
OMNIMATCH_WEB_SEARCH_API_KEY=

OMNIMATCH_SHIPPING_PROVIDER=rate_table
OMNIMATCH_MEMORY_PROVIDER=memory
OMNIMATCH_EVAL_PROVIDER=heuristic
```

- [ ] **Step 6: Run config tests**

Run: `uv run pytest tests/test_config.py -v`

Expected: all config tests pass.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example app/config.py tests/test_config.py
git commit -m "feat: add competition agent config profiles"
```

---

### Task 2: Provider Contracts And Registry

**Files:**
- Create: `app/providers/__init__.py`
- Create: `app/providers/base.py`
- Create: `app/providers/placeholder.py`
- Create: `app/providers/registry.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `OmniMatchSettings`
- Produces: `ProviderResult[T]`
- Produces: `ProviderError`
- Produces: `LLMProvider`, `ProductSearchProvider`, `WebSearchProvider`, `ShippingProvider`
- Produces: `ProviderRegistry.from_settings(settings: OmniMatchSettings) -> ProviderRegistry`

- [ ] **Step 1: Write failing provider contract tests**

```python
import pytest

from app.config import OmniMatchSettings
from app.providers.base import ProviderResult
from app.providers.registry import ProviderRegistry


@pytest.mark.asyncio
async def test_submission_registry_uses_placeholder_providers():
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )

    registry = ProviderRegistry.from_settings(settings)
    result = await registry.product.search("旅行三件套", platforms=["Amazon"])

    assert isinstance(result, ProviderResult)
    assert result.provider_mode == "placeholder"
    assert result.latency_ms >= 0
    assert result.data
    assert "api_key" not in result.response_summary.lower()


def test_provider_result_redacts_secret_like_values():
    result = ProviderResult(
        provider="unit",
        provider_mode="real",
        latency_ms=1,
        data={"ok": True},
        warnings=[],
        response_summary="Authorization: Bearer secret-token api_key=abc",
    )

    assert "secret-token" not in result.redacted_summary()
    assert "api_key=abc" not in result.redacted_summary()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`

Expected: fails because provider modules do not exist.

- [ ] **Step 3: Implement `app/providers/base.py`**

```python
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
        text = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", self.response_summary)
        text = re.sub(r"api_key=([^&\s]+)", "api_key=[REDACTED]", text, flags=re.IGNORECASE)
        return text


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
    async def estimate(self, product: dict[str, Any], destination: str | None) -> ProviderResult[dict[str, Any]]:
        ...
```

- [ ] **Step 4: Implement deterministic placeholder providers**

```python
from __future__ import annotations

import time
from typing import Any

from app.providers.base import ProviderResult


def _latency_ms(start: float) -> int:
    return max(0, int((time.perf_counter() - start) * 1000))


class PlaceholderLLMProvider:
    provider = "placeholder_llm"

    async def plan_next_action(self, messages: list[dict[str, Any]]) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data={"action": "finish_if_enough_else_search", "arguments": {}},
            warnings=["placeholder LLM used"],
            response_summary="deterministic placeholder plan",
        )


class PlaceholderProductSearchProvider:
    provider = "placeholder_product"

    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        data: list[dict[str, Any]] = []
        for platform in platforms:
            slug = platform.lower().replace(" ", "-")
            data.append(
                {
                    "id": f"{slug}-canvas-travel-set",
                    "platform": platform,
                    "title": f"{platform} canvas travel set",
                    "price": 198,
                    "currency": "CNY",
                    "rating": 4.6,
                    "url": f"https://example.com/{slug}/canvas-travel-set",
                    "evidence": ["placeholder catalog fixture"],
                    "material": "canvas",
                }
            )
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data=data,
            warnings=["placeholder product data used"],
            response_summary=f"placeholder products for query={query!r}",
        )


class PlaceholderWebSearchProvider:
    provider = "placeholder_web_search"

    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data=[{"title": "Travel set buying guide", "url": "https://example.com/guide"}],
            warnings=["placeholder web evidence used"],
            response_summary=f"placeholder web result for query={query!r}",
        )


class PlaceholderShippingProvider:
    provider = "placeholder_shipping"

    async def estimate(self, product: dict[str, Any], destination: str | None) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        return ProviderResult(
            provider=self.provider,
            provider_mode="placeholder",
            latency_ms=_latency_ms(start),
            data={"shipping": 20, "tax": 0, "destination": destination},
            warnings=[],
            response_summary="placeholder shipping estimate",
        )
```

- [ ] **Step 5: Implement provider registry**

```python
from __future__ import annotations

from dataclasses import dataclass

from app.config import OmniMatchSettings
from app.providers.placeholder import (
    PlaceholderLLMProvider,
    PlaceholderProductSearchProvider,
    PlaceholderShippingProvider,
    PlaceholderWebSearchProvider,
)


@dataclass(frozen=True)
class ProviderRegistry:
    llm: object
    product: object
    web_search: object
    shipping: object

    @classmethod
    def from_settings(cls, settings: OmniMatchSettings) -> "ProviderRegistry":
        return cls(
            llm=PlaceholderLLMProvider(),
            product=PlaceholderProductSearchProvider(),
            web_search=PlaceholderWebSearchProvider(),
            shipping=PlaceholderShippingProvider(),
        )
```

- [ ] **Step 6: Run provider tests**

Run: `uv run pytest tests/test_providers.py -v`

Expected: all provider tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/providers tests/test_providers.py
git commit -m "feat: add provider contracts and placeholder registry"
```

---

### Task 3: Real HTTP Providers

**Files:**
- Create: `app/providers/http_product.py`
- Create: `app/providers/http_web_search.py`
- Create: `app/providers/openai_llm.py`
- Modify: `app/providers/registry.py`
- Test: `tests/test_real_provider_adapters.py`

**Interfaces:**
- Consumes: `ProviderResult`
- Produces: `HttpProductSearchProvider.search(query: str, platforms: list[str])`
- Produces: `HttpWebSearchProvider.search(query: str)`
- Produces: `OpenAILLMProvider.plan_next_action(messages: list[dict])`

- [ ] **Step 1: Write failing adapter tests using `httpx.MockTransport`**

```python
import httpx
import pytest

from app.providers.http_product import HttpProductSearchProvider
from app.providers.http_web_search import HttpWebSearchProvider


@pytest.mark.asyncio
async def test_http_product_provider_normalizes_items():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "unit-key"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "raw-1",
                        "platform": "Amazon",
                        "title": "Canvas travel set",
                        "price": 199,
                        "currency": "CNY",
                        "url": "https://example.com/raw-1",
                        "rating": 4.7,
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpProductSearchProvider(
        api_url="https://product.example/search",
        api_key="unit-key",
        client=client,
    )

    result = await provider.search("旅行三件套", platforms=["Amazon"])

    assert result.provider_mode == "real"
    assert result.data[0]["id"] == "raw-1"
    assert result.data[0]["platform"] == "Amazon"


@pytest.mark.asyncio
async def test_http_web_search_provider_normalizes_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"results": [{"title": "Guide", "url": "https://example.com/guide", "snippet": "ok"}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpWebSearchProvider(
        api_url="https://search.example/query",
        api_key="unit-key",
        client=client,
    )

    result = await provider.search("travel set material")

    assert result.provider_mode == "real"
    assert result.data[0]["title"] == "Guide"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_real_provider_adapters.py -v`

Expected: fails because real adapter modules do not exist.

- [ ] **Step 3: Implement real HTTP product provider**

```python
from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class HttpProductSearchProvider:
    provider = "http_product"

    def __init__(self, api_url: str, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str, platforms: list[str]) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.get(
            self.api_url,
            params={"q": query, "platforms": ",".join(platforms)},
            headers={"x-api-key": self.api_key},
        )
        if response.status_code >= 400:
            raise ProviderError(self.provider, f"product provider returned {response.status_code}")
        payload = response.json()
        items = [self._normalize_item(item) for item in payload.get("items", [])]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=items,
            warnings=[],
            response_summary=f"normalized {len(items)} product items",
        )

    @staticmethod
    def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item["id"]),
            "platform": str(item.get("platform", "unknown")),
            "title": str(item["title"]),
            "price": float(item["price"]),
            "currency": str(item.get("currency", "CNY")),
            "url": str(item["url"]),
            "rating": float(item.get("rating", 0)),
            "evidence": item.get("evidence", []),
            "material": item.get("material"),
        }
```

- [ ] **Step 4: Implement real HTTP web search provider**

```python
from __future__ import annotations

import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class HttpWebSearchProvider:
    provider = "http_web_search"

    def __init__(self, api_url: str, api_key: str, client: httpx.AsyncClient | None = None) -> None:
        self.api_url = api_url
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=20)

    async def search(self, query: str) -> ProviderResult[list[dict[str, Any]]]:
        start = time.perf_counter()
        response = await self.client.get(
            self.api_url,
            params={"q": query},
            headers={"x-api-key": self.api_key},
        )
        if response.status_code >= 400:
            raise ProviderError(self.provider, f"web search provider returned {response.status_code}")
        payload = response.json()
        results = [
            {
                "title": str(item.get("title", "")),
                "url": str(item.get("url", "")),
                "snippet": str(item.get("snippet", "")),
            }
            for item in payload.get("results", [])
        ]
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=results,
            warnings=[],
            response_summary=f"normalized {len(results)} web results",
        )
```

- [ ] **Step 5: Add real provider selection in registry**

`ProviderRegistry.from_settings()` should instantiate real HTTP providers when
the configured provider is not `placeholder`. Use env vars for keys:

```python
import os

from app.providers.http_product import HttpProductSearchProvider
from app.providers.http_web_search import HttpWebSearchProvider


product = (
    PlaceholderProductSearchProvider()
    if settings.product_provider == "placeholder"
    else HttpProductSearchProvider(
        api_url=settings.product_api_url or "",
        api_key=os.getenv("OMNIMATCH_PRODUCT_API_KEY", ""),
    )
)
web_search = (
    PlaceholderWebSearchProvider()
    if settings.web_search_provider == "placeholder"
    else HttpWebSearchProvider(
        api_url=settings.web_search_api_url or "",
        api_key=os.getenv("OMNIMATCH_WEB_SEARCH_API_KEY", ""),
    )
)
```

- [ ] **Step 6: Add OpenAI LLM adapter**

Create `app/providers/openai_llm.py` with strict JSON parsing. It uses the
OpenAI-compatible chat completions HTTP API so the first implementation does not
need an SDK dependency:

```python
from __future__ import annotations

import json
import time
from typing import Any

import httpx

from app.providers.base import ProviderError, ProviderResult


class OpenAILLMProvider:
    provider = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.client = client or httpx.AsyncClient(timeout=30)

    async def plan_next_action(self, messages: list[dict[str, Any]]) -> ProviderResult[dict[str, Any]]:
        start = time.perf_counter()
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code >= 400:
            raise ProviderError(self.provider, f"LLM provider returned {response.status_code}")
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        try:
            action = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(self.provider, "LLM response was not valid JSON") from exc
        if "action" not in action or "arguments" not in action:
            raise ProviderError(self.provider, "LLM response must include action and arguments")
        return ProviderResult(
            provider=self.provider,
            provider_mode="real",
            latency_ms=max(0, int((time.perf_counter() - start) * 1000)),
            data=action,
            warnings=[],
            response_summary=f"action={action['action']}",
        )
```

- [ ] **Step 7: Run adapter tests**

Run: `uv run pytest tests/test_real_provider_adapters.py tests/test_providers.py -v`

Expected: all provider tests pass.

- [ ] **Step 8: Commit**

```bash
git add app/providers tests/test_real_provider_adapters.py
git commit -m "feat: add real API provider adapters"
```

---

### Task 4: Candidate Schemas And Ranking

**Files:**
- Modify: `app/schemas.py`
- Create: `app/ranking/__init__.py`
- Create: `app/ranking/scorer.py`
- Test: `tests/test_ranking.py`

**Interfaces:**
- Produces: `ShoppingIntent`
- Produces: `ProductCandidate`
- Produces: `CandidateScore`
- Produces: `ScoredProduct`
- Produces: `score_candidates(intent: ShoppingIntent, candidates: list[ProductCandidate]) -> list[ScoredProduct]`

- [ ] **Step 1: Write failing ranking tests**

```python
from app.ranking.scorer import score_candidates
from app.schemas import ProductCandidate, ShoppingIntent


def test_ranking_penalizes_forbidden_material():
    intent = ShoppingIntent(
        original_query="旅行三件套，预算300，不要塑料",
        category="旅行三件套",
        budget=300,
        preferences=["耐用"],
        negative_constraints=["塑料"],
    )
    candidates = [
        ProductCandidate(
            id="plastic",
            platform="A",
            title="Plastic travel set",
            price=100,
            currency="CNY",
            url="https://example.com/plastic",
            material="plastic",
            evidence=["catalog"],
        ),
        ProductCandidate(
            id="canvas",
            platform="B",
            title="Canvas travel set",
            price=190,
            currency="CNY",
            url="https://example.com/canvas",
            material="canvas",
            evidence=["catalog", "guide"],
        ),
    ]

    scored = score_candidates(intent, candidates)

    assert scored[0].candidate.id == "canvas"
    assert any("negative constraint" in reason for reason in scored[1].score.rejection_reasons)


def test_ranking_marks_over_budget_candidate():
    intent = ShoppingIntent(
        original_query="预算100旅行三件套",
        category="旅行三件套",
        budget=100,
        preferences=[],
        negative_constraints=[],
    )
    candidate = ProductCandidate(
        id="expensive",
        platform="A",
        title="Expensive set",
        price=180,
        currency="CNY",
        shipping=30,
        tax=0,
        url="https://example.com/expensive",
        evidence=["catalog"],
    )

    scored = score_candidates(intent, [candidate])

    assert scored[0].score.total_landed_cost == 210
    assert "over budget" in scored[0].score.rejection_reasons
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ranking.py -v`

Expected: fails because ranking models and scorer do not exist.

- [ ] **Step 3: Extend `app/schemas.py`**

Add Pydantic models while keeping existing models compatible:

```python
class ShoppingIntent(BaseModel):
    original_query: str
    category: str
    budget: float | None = None
    preferences: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    destination: str | None = None


class ProductCandidate(BaseModel):
    id: str
    platform: str
    title: str
    price: float = Field(..., ge=0)
    currency: str = "CNY"
    shipping: float = Field(default=0, ge=0)
    tax: float = Field(default=0, ge=0)
    rating: float = Field(default=0, ge=0, le=5)
    url: str
    material: str | None = None
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def total_landed_cost(self) -> float:
        return round(self.price + self.shipping + self.tax, 2)


class CandidateScore(BaseModel):
    total: float
    constraint_score: float
    evidence_score: float
    price_score: float
    preference_score: float
    risk_penalty: float
    total_landed_cost: float
    rejection_reasons: list[str] = Field(default_factory=list)


class ScoredProduct(BaseModel):
    candidate: ProductCandidate
    score: CandidateScore
```

- [ ] **Step 4: Implement deterministic scorer**

```python
from __future__ import annotations

from app.schemas import CandidateScore, ProductCandidate, ScoredProduct, ShoppingIntent


def score_candidates(intent: ShoppingIntent, candidates: list[ProductCandidate]) -> list[ScoredProduct]:
    scored = [_score_one(intent, candidate) for candidate in candidates]
    return sorted(scored, key=lambda item: item.score.total, reverse=True)


def _score_one(intent: ShoppingIntent, candidate: ProductCandidate) -> ScoredProduct:
    reasons: list[str] = []
    constraint_score = 40.0
    if intent.budget is not None and candidate.total_landed_cost > intent.budget:
        constraint_score -= 25.0
        reasons.append("over budget")

    material = (candidate.material or candidate.title).lower()
    for forbidden in intent.negative_constraints:
        if forbidden.lower() in material or _material_matches_zh(forbidden, material):
            constraint_score -= 35.0
            reasons.append(f"negative constraint matched: {forbidden}")

    evidence_score = min(20.0, 8.0 * len(candidate.evidence))
    price_score = 20.0
    if intent.budget:
        price_score = max(0.0, 20.0 * (1 - candidate.total_landed_cost / (intent.budget * 1.5)))
    preference_score = 10.0 if any(pref.lower() in material for pref in intent.preferences) else 4.0
    risk_penalty = 10.0 if not candidate.url or not candidate.evidence else 0.0
    total = max(0.0, constraint_score + evidence_score + price_score + preference_score - risk_penalty)

    return ScoredProduct(
        candidate=candidate,
        score=CandidateScore(
            total=round(total, 2),
            constraint_score=round(constraint_score, 2),
            evidence_score=round(evidence_score, 2),
            price_score=round(price_score, 2),
            preference_score=round(preference_score, 2),
            risk_penalty=round(risk_penalty, 2),
            total_landed_cost=candidate.total_landed_cost,
            rejection_reasons=reasons,
        ),
    )


def _material_matches_zh(forbidden: str, material: str) -> bool:
    if forbidden == "塑料":
        return "plastic" in material
    return False
```

- [ ] **Step 5: Run ranking tests**

Run: `uv run pytest tests/test_ranking.py tests/test_schemas.py -v`

Expected: ranking and existing schema tests pass.

- [ ] **Step 6: Commit**

```bash
git add app/schemas.py app/ranking tests/test_ranking.py
git commit -m "feat: add structured candidate ranking"
```

---

### Task 5: Tool Layer Uses Providers

**Files:**
- Create: `app/tools/context.py`
- Modify: `app/tools/planner.py`
- Modify: `app/tools/category_insight.py`
- Modify: `app/tools/item_search.py`
- Modify: `app/tools/shipping_calc.py`
- Modify: `app/tools/price_compare.py`
- Modify: `app/tools/item_picker.py`
- Modify: `app/tools/shopping_summary.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ProviderRegistry`
- Consumes: `ShoppingIntent`, `ProductCandidate`, `ScoredProduct`
- Produces: `ToolContext(settings, providers, trace)`
- Produces: provider-backed async tool functions

- [ ] **Step 1: Rewrite tool tests around fake provider context**

```python
import pytest

from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry
from app.tools.category_insight import get_category_insight
from app.tools.context import ToolContext
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping
from app.tools.shopping_summary import build_summary


@pytest.mark.asyncio
async def test_tool_chain_uses_provider_backed_candidates():
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )
    ctx = ToolContext(settings=settings, providers=ProviderRegistry.from_settings(settings))

    intent = await plan_query("我想买旅行三件套，预算300，不要塑料", ctx)
    insight = await get_category_insight(intent, ctx)
    candidates = await search_items(intent, insight, ctx)
    shipped = await calculate_shipping(candidates, ctx)
    compared = await compare_prices(shipped, intent, ctx)
    picked = await pick_items(compared, intent, ctx)
    summary = await build_summary("原始需求", picked, ctx)

    assert intent.negative_constraints == ["塑料"]
    assert candidates[0].evidence
    assert picked[0].score.total >= picked[-1].score.total
    assert summary.products
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`

Expected: fails because tools still use the old mock signatures.

- [ ] **Step 3: Add `ToolContext`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry


@dataclass
class ToolContext:
    settings: OmniMatchSettings
    providers: ProviderRegistry
    observations: list[dict[str, Any]] = field(default_factory=list)
```

- [ ] **Step 4: Update tool signatures**

Use these signatures consistently:

```python
async def plan_query(query: str, ctx: ToolContext) -> ShoppingIntent: ...
async def get_category_insight(intent: ShoppingIntent, ctx: ToolContext) -> dict: ...
async def search_items(intent: ShoppingIntent, insight: dict, ctx: ToolContext) -> list[ProductCandidate]: ...
async def calculate_shipping(candidates: list[ProductCandidate], ctx: ToolContext) -> list[ProductCandidate]: ...
async def compare_prices(candidates: list[ProductCandidate], intent: ShoppingIntent, ctx: ToolContext) -> list[ScoredProduct]: ...
async def pick_items(scored: list[ScoredProduct], intent: ShoppingIntent, ctx: ToolContext) -> list[ScoredProduct]: ...
async def build_summary(query: str, picked: list[ScoredProduct], ctx: ToolContext) -> ShoppingSummary: ...
```

- [ ] **Step 5: Implement provider-backed `search_items`**

Replace the old fixed mock implementation with provider-backed normalization:

```python
from __future__ import annotations

from app.schemas import ProductCandidate, ShoppingIntent
from app.tools.context import ToolContext


DEFAULT_PLATFORMS = ["Amazon", "eBay", "AliExpress", "Shopee"]


async def search_items(
    intent: ShoppingIntent,
    insight: dict,
    ctx: ToolContext,
) -> list[ProductCandidate]:
    query = f"{intent.category} {' '.join(intent.preferences)}".strip()
    platforms = insight.get("platforms", DEFAULT_PLATFORMS)
    result = await ctx.providers.product.search(query, platforms=platforms)
    ctx.observations.append(
        {
            "tool": "ItemSearch",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": result.warnings,
        }
    )
    return [ProductCandidate(**item) for item in result.data]
```

Apply the same pattern to shipping and web/category tools: call the provider,
append a compact observation, and return normalized Pydantic models or dicts.

- [ ] **Step 6: Run tool tests**

Run: `uv run pytest tests/test_tools.py tests/test_ranking.py -v`

Expected: tool and ranking tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/tools tests/test_tools.py
git commit -m "feat: route tools through providers"
```

---

### Task 6: Observation-Driven AgentLoop

**Files:**
- Create: `app/agent/tool_registry.py`
- Modify: `app/agent/main_agent.py`
- Modify: `app/agent/dispatch_tool.py`
- Test: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: `ToolContext`
- Consumes: `ProviderRegistry`
- Produces: `CompetitionAgentLoop.run(query: str) -> ShoppingSummary`
- Produces: `ToolRegistry.run(action: str, arguments: dict) -> object`

- [ ] **Step 1: Write failing loop tests**

```python
import pytest

from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry


def submission_settings() -> OmniMatchSettings:
    return OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )


@pytest.mark.asyncio
async def test_competition_loop_emits_provider_and_ranking_events(tmp_path):
    settings = submission_settings()
    monitor = EventCollector(thread_id="thread_test")
    loop = CompetitionAgentLoop(
        thread_id="thread_test",
        session_dir=tmp_path,
        settings=settings,
        providers=ProviderRegistry.from_settings(settings),
        monitor=monitor,
    )

    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")

    event_types = [event.type for event in monitor.events]
    assert "provider_start" in event_types
    assert "provider_end" in event_types
    assert "ranking_decision" in event_types
    assert "task_result" in event_types
    assert summary.products
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "candidates.json").exists()
    assert (tmp_path / "trace.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_agent_loop.py -v`

Expected: fails because `CompetitionAgentLoop` does not exist.

- [ ] **Step 3: Implement tool registry**

`ToolRegistry` should own the mapping from action names to tool functions:

```python
class ToolRegistry:
    def __init__(self, ctx: ToolContext) -> None:
        self.ctx = ctx
        self.intent = None
        self.insight = None
        self.candidates = []
        self.scored = []

    async def run(self, action: str, arguments: dict):
        if action == "plan":
            self.intent = await plan_query(arguments["query"], self.ctx)
            return self.intent
        if action == "category_insight":
            self.insight = await get_category_insight(self.intent, self.ctx)
            return self.insight
        if action == "item_search":
            self.candidates = await search_items(self.intent, self.insight or {}, self.ctx)
            return self.candidates
        if action == "shipping":
            self.candidates = await calculate_shipping(self.candidates, self.ctx)
            return self.candidates
        if action == "rank":
            self.scored = await compare_prices(self.candidates, self.intent, self.ctx)
            return self.scored
        if action == "pick":
            return await pick_items(self.scored, self.intent, self.ctx)
        raise ValueError(f"unknown tool action: {action}")
```

- [ ] **Step 4: Implement `CompetitionAgentLoop`**

Implement the first loop with a deterministic policy while preserving the real
action/observation shape:

```text
plan -> category_insight -> item_search -> shipping -> rank -> pick -> summary
```

Each provider-backed step must emit `provider_start` and `provider_end`. Ranking
must emit `ranking_decision`. The loop must write `summary.json`,
`candidates.json`, and `trace.jsonl`.

Use this structure in `app/agent/main_agent.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.agent.tool_registry import ToolRegistry
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.providers.registry import ProviderRegistry
from app.schemas import ShoppingSummary
from app.tools.context import ToolContext
from app.tools.shopping_summary import build_summary


class CompetitionAgentLoop:
    def __init__(
        self,
        thread_id: str,
        session_dir: str | Path,
        settings: OmniMatchSettings,
        providers: ProviderRegistry,
        monitor: EventCollector,
    ) -> None:
        self.thread_id = thread_id
        self.session_dir = Path(session_dir)
        self.settings = settings
        self.providers = providers
        self.monitor = monitor

    async def run(self, query: str) -> ShoppingSummary:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(settings=self.settings, providers=self.providers)
        tools = ToolRegistry(ctx)
        trace: list[dict] = []

        await self.monitor.emit(
            "task_started",
            "Competition Agent started.",
            payload={"profile": self.settings.profile, "provider_modes": self.settings.provider_modes()},
        )
        picked = None
        for action, arguments in [
            ("plan", {"query": query}),
            ("category_insight", {}),
            ("item_search", {}),
            ("shipping", {}),
            ("rank", {}),
            ("pick", {}),
        ]:
            await self.monitor.emit("tool_start", f"{action} started", tool=action)
            if action == "item_search":
                await self.monitor.emit("provider_start", "Product provider search started.", tool=action)
            result = await tools.run(action, arguments)
            trace.append({"action": action, "observation_count": len(ctx.observations)})
            await self.monitor.emit("tool_end", f"{action} finished", tool=action)
            if action == "item_search":
                await self.monitor.emit("provider_end", "Product provider returned candidates", tool=action)
            if action == "rank":
                await self.monitor.emit(
                    "ranking_decision",
                    "Candidates scored.",
                    payload={"candidate_count": len(result)},
                )
            if action == "pick":
                picked = result

        summary = await build_summary(query, picked or [], ctx)
        self._write_json("summary.json", summary.model_dump())
        self._write_json("candidates.json", [item.model_dump() for item in tools.scored])
        self._write_jsonl("trace.jsonl", trace)
        await self.monitor.emit("task_result", "Shopping summary generated.", payload={"summary": summary.model_dump()})
        return summary

    def _write_json(self, filename: str, payload: object) -> None:
        (self.session_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_jsonl(self, filename: str, rows: list[dict]) -> None:
        text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        (self.session_dir / filename).write_text(text + "\n", encoding="utf-8")


MockAgentLoop = CompetitionAgentLoop
```

- [ ] **Step 5: Keep compatibility alias**

Keep `MockAgentLoop = CompetitionAgentLoop` only if old tests or examples still
import `MockAgentLoop`. Update README in Task 10 to point users at the
competition loop.

- [ ] **Step 6: Run loop tests**

Run: `uv run pytest tests/test_agent_loop.py -v`

Expected: loop tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/agent tests/test_agent_loop.py
git commit -m "feat: add observation-driven competition agent loop"
```

---

### Task 7: Task API Hardening And Trace State

**Files:**
- Modify: `app/api/server.py`
- Modify: `app/api/connection.py`
- Modify: `app/api/monitor.py`
- Modify: `app/schemas.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `OmniMatchSettings.from_env()`
- Consumes: `CompetitionAgentLoop`
- Produces: task state fields `warnings`, `profile`, `provider_modes`, `trace_paths`

- [ ] **Step 1: Write failing API hardening tests**

```python
from fastapi.testclient import TestClient

from app.api.server import TASKS, app
from app.schemas import TaskState


def test_get_task_includes_profile_and_trace_paths():
    client = TestClient(app)
    TASKS["thread_done"] = TaskState(
        thread_id="thread_done",
        status="completed",
        profile="submission",
        provider_modes={"llm": "placeholder"},
        trace_paths={"summary": "output/thread_done/summary.json"},
    )

    response = client.get("/api/tasks/thread_done")

    assert response.status_code == 200
    data = response.json()
    assert data["profile"] == "submission"
    assert data["provider_modes"]["llm"] == "placeholder"
    assert data["trace_paths"]["summary"].endswith("summary.json")


def test_unknown_websocket_thread_is_rejected():
    client = TestClient(app)

    try:
        with client.websocket_connect("/ws/thread_missing"):
            raise AssertionError("unknown websocket should not stay connected")
    except Exception as exc:
        assert "1008" in str(exc) or "WebSocketDisconnect" in exc.__class__.__name__
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`

Expected: fails because task state lacks new fields and unknown WebSocket stays connected.

- [ ] **Step 3: Extend `TaskState`**

Add:

```python
warnings: list[str] = Field(default_factory=list)
profile: str | None = None
provider_modes: dict[str, str] = Field(default_factory=dict)
trace_paths: dict[str, str] = Field(default_factory=dict)
```

- [ ] **Step 4: Reject unknown WebSocket threads**

In `websocket_events`, check task existence before `connect`:

```python
state = TASKS.get(thread_id)
if state is None:
    await websocket.close(code=1008, reason="Task not found")
    return
await manager.connect(thread_id, websocket)
```

- [ ] **Step 5: Use settings and provider registry in task execution**

`create_task` should load settings, store profile and provider modes, then pass
settings/providers to `CompetitionAgentLoop`.

- [ ] **Step 6: Preserve final answer if output writing fails**

Wrap trace file writing in the loop or server so task result still completes and
adds a warning event when output persistence fails.

- [ ] **Step 7: Run API tests**

Run: `uv run pytest tests/test_api.py tests/test_agent_loop.py -v`

Expected: API and loop tests pass.

- [ ] **Step 8: Commit**

```bash
git add app/api app/schemas.py tests/test_api.py
git commit -m "feat: harden task API for competition traces"
```

---

### Task 8: Evaluation Harness

**Files:**
- Create: `app/eval/cases.py`
- Create: `app/eval/runner.py`
- Create: `app/eval/fixtures/competition_smoke.jsonl`
- Test: `tests/test_eval_runner.py`

**Interfaces:**
- Produces: `EvalCase`
- Produces: `EvalResult`
- Produces: `run_eval_cases(cases: list[EvalCase], settings: OmniMatchSettings) -> list[EvalResult]`

- [ ] **Step 1: Write failing eval tests**

```python
import pytest

from app.config import OmniMatchSettings
from app.eval.cases import EvalCase
from app.eval.runner import run_eval_cases


@pytest.mark.asyncio
async def test_eval_runner_returns_scores_for_submission_profile(tmp_path):
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="heuristic",
    )
    cases = [
        EvalCase(
            id="budget_no_plastic",
            query="旅行三件套，预算300，不要塑料",
            required_terms=["旅行", "塑料"],
            forbidden_terms=["无法推荐"],
        )
    ]

    results = await run_eval_cases(cases, settings=settings, output_dir=tmp_path)

    assert results[0].case_id == "budget_no_plastic"
    assert 0 <= results[0].score <= 1
    assert results[0].trace_dir.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_runner.py -v`

Expected: fails because eval case and runner modules do not exist.

- [ ] **Step 3: Implement eval case models**

```python
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class EvalCase(BaseModel):
    id: str
    query: str
    required_terms: list[str] = []
    forbidden_terms: list[str] = []


class EvalResult(BaseModel):
    case_id: str
    score: float
    passed: bool
    notes: list[str]
    trace_dir: Path
```

- [ ] **Step 4: Implement heuristic eval runner**

The first runner should execute `CompetitionAgentLoop` and score
required/forbidden terms against the final summary message and product titles:

```python
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.eval.cases import EvalCase, EvalResult
from app.providers.registry import ProviderRegistry


async def run_eval_cases(
    cases: list[EvalCase],
    settings: OmniMatchSettings,
    output_dir: Path,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    providers = ProviderRegistry.from_settings(settings)
    for case in cases:
        thread_id = f"eval_{case.id}_{uuid4().hex[:6]}"
        trace_dir = output_dir / thread_id
        monitor = EventCollector(thread_id=thread_id)
        loop = CompetitionAgentLoop(
            thread_id=thread_id,
            session_dir=trace_dir,
            settings=settings,
            providers=providers,
            monitor=monitor,
        )
        summary = await loop.run(case.query)
        text = " ".join([summary.message, *[product.title for product in summary.products]])
        notes: list[str] = []
        required_hits = sum(1 for term in case.required_terms if term in text)
        forbidden_hits = [term for term in case.forbidden_terms if term.lower() in text.lower()]
        if forbidden_hits:
            notes.append(f"forbidden terms present: {', '.join(forbidden_hits)}")
        required_score = required_hits / max(1, len(case.required_terms))
        score = max(0.0, required_score - 0.5 * len(forbidden_hits))
        results.append(
            EvalResult(
                case_id=case.id,
                score=round(score, 2),
                passed=score >= 0.7,
                notes=notes,
                trace_dir=trace_dir,
            )
        )
    return results
```

- [ ] **Step 5: Add smoke cases**

Create `app/eval/fixtures/competition_smoke.jsonl` with:

```jsonl
{"id":"budget_no_plastic","query":"旅行三件套，预算300，不要塑料","required_terms":["旅行"],"forbidden_terms":["无法推荐"]}
{"id":"ambiguous_phone","query":"给我推荐一个手机","required_terms":["推荐"],"forbidden_terms":["traceback"]}
{"id":"cheap_durable","query":"便宜耐用的通勤背包，预算200","required_terms":["推荐"],"forbidden_terms":["traceback"]}
```

- [ ] **Step 6: Run eval tests**

Run: `uv run pytest tests/test_eval_runner.py -v`

Expected: eval tests pass.

- [ ] **Step 7: Commit**

```bash
git add app/eval tests/test_eval_runner.py
git commit -m "feat: add competition evaluation harness"
```

---

### Task 9: Frontend Observability

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.css`

**Interfaces:**
- Consumes: task state fields `profile`, `provider_modes`, `warnings`, `trace_paths`
- Consumes: event types `provider_start`, `provider_end`, `provider_error`, `ranking_decision`
- Produces: visible profile/provider/trace status in the React console

- [ ] **Step 1: Extend frontend types**

Add fields to the TypeScript task and event types:

```ts
type TaskState = {
  thread_id: string;
  status: string;
  profile?: string | null;
  provider_modes?: Record<string, string>;
  warnings?: string[];
  trace_paths?: Record<string, string>;
};

type ProviderPayload = {
  provider?: string;
  provider_mode?: string;
  latency_ms?: number;
  warnings?: string[];
};
```

- [ ] **Step 2: Fetch final task state after result**

When `task_result` arrives, call `GET /api/tasks/{thread_id}` and render
profile, provider modes, warnings, and trace paths from the response.

```ts
const [taskState, setTaskState] = useState<TaskState | null>(null);

async function refreshTaskState(nextThreadId: string) {
  const response = await fetch(`/api/tasks/${nextThreadId}`);
  if (!response.ok) {
    return;
  }
  setTaskState((await response.json()) as TaskState);
}
```

- [ ] **Step 3: Render provider events distinctly**

Provider events should show provider id, mode, latency, and warnings from
`event.payload` without interpreting ranking logic on the client.

```tsx
function ProviderMeta({ event }: { event: AgentEvent }) {
  if (!event.type.startsWith("provider_")) {
    return null;
  }
  const payload = event.payload as ProviderPayload;
  return (
    <div className="provider-meta">
      {payload.provider && <span>{payload.provider}</span>}
      {payload.provider_mode && <span>{payload.provider_mode}</span>}
      {typeof payload.latency_ms === "number" && <span>{payload.latency_ms} ms</span>}
    </div>
  );
}
```

- [ ] **Step 4: Run frontend build**

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite build pass.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/App.tsx frontend/src/App.css
git commit -m "feat: show competition agent observability"
```

---

### Task 10: Submission Profile, README, And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `.env.example`
- Modify: `examples/run_mock_agent.py`
- Create: `examples/run_competition_agent.py`

**Interfaces:**
- Produces: documented dev run with real APIs.
- Produces: documented submission run without secrets.
- Produces: CLI smoke test for competition loop.

- [ ] **Step 1: Update README direction**

README should state:

```markdown
OmniMatch is now a competition-grade shopping agent project. The old mock MVP is
kept as historical context, but new development should target the provider-backed
competition agent.
```

- [ ] **Step 2: Document dev profile**

Add commands:

```bash
cp .env.example .env
# fill OPENAI_API_KEY, OMNIMATCH_PRODUCT_API_URL, OMNIMATCH_PRODUCT_API_KEY,
# OMNIMATCH_WEB_SEARCH_API_URL, and OMNIMATCH_WEB_SEARCH_API_KEY
OMNIMATCH_PROFILE=dev uv run uvicorn app.api.server:app --reload
```

- [ ] **Step 3: Document submission profile**

Add commands:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
OMNIMATCH_PROFILE=submission uv run pytest -v
```

- [ ] **Step 4: Add CLI example for competition loop**

`examples/run_competition_agent.py` should load settings, build providers, run
`CompetitionAgentLoop`, print the summary JSON, and write trace files under
`output/thread_example/`.

- [ ] **Step 5: Run backend tests**

Run: `uv run pytest -v`

Expected: all backend tests pass.

- [ ] **Step 6: Run frontend build**

Run: `cd frontend && npm run build`

Expected: build exits 0.

- [ ] **Step 7: Run submission smoke**

Run: `OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py`

Expected: exits 0, prints a summary, and writes output files without real API keys.

- [ ] **Step 8: Check changed files**

Run: `git status --short`

Expected: only intentional files are modified.

- [ ] **Step 9: Commit**

```bash
git add README.md .env.example examples/run_mock_agent.py examples/run_competition_agent.py
git commit -m "docs: document competition agent workflows"
```
