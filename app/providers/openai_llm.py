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
