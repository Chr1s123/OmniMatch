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
                llm_provider="placeholder",
                llm_model=os.getenv("OMNIMATCH_LLM_MODEL", "placeholder-llm"),
                product_provider="placeholder",
                web_search_provider="placeholder",
                shipping_provider="placeholder",
                memory_provider="placeholder",
                eval_provider="placeholder",
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
            "web_search": self._mode_for(
                self.web_search_provider,
                fake_allowed=self.profile == "test",
            ),
            "shipping": self._mode_for(
                self.shipping_provider,
                fake_allowed=self.profile == "test",
            ),
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
