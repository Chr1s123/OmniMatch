from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv


Profile = Literal["dev", "submission", "test"]
ProviderMode = Literal["real", "placeholder", "fake"]


class ConfigError(RuntimeError):
    pass


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


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
    max_fork_depth: int = 1
    max_parallel_subagents: int = 4
    subagent_max_steps: int = 4
    subagent_timeout_seconds: float = 30.0

    @classmethod
    def from_env(cls) -> "OmniMatchSettings":
        process_env = dict(os.environ)
        load_dotenv()
        profile = process_env.get("OMNIMATCH_PROFILE") or os.getenv("OMNIMATCH_PROFILE", "dev")
        if profile not in {"dev", "submission", "test"}:
            raise ConfigError("OMNIMATCH_PROFILE must be dev, submission, or test")
        fork_limits = {
            "max_fork_depth": _env_int("OMNIMATCH_MAX_FORK_DEPTH", 1),
            "max_parallel_subagents": _env_int(
                "OMNIMATCH_MAX_PARALLEL_SUBAGENTS", 4
            ),
            "subagent_max_steps": _env_int("OMNIMATCH_SUBAGENT_MAX_STEPS", 4),
            "subagent_timeout_seconds": _env_float(
                "OMNIMATCH_SUBAGENT_TIMEOUT_SECONDS", 30.0
            ),
        }

        if profile == "submission":
            settings = cls(
                profile="submission",
                llm_provider=process_env.get("OMNIMATCH_LLM_PROVIDER", "placeholder"),
                llm_model=process_env.get("OMNIMATCH_LLM_MODEL", "placeholder-llm"),
                product_provider=process_env.get("OMNIMATCH_PRODUCT_PROVIDER", "placeholder"),
                web_search_provider=process_env.get("OMNIMATCH_WEB_SEARCH_PROVIDER", "placeholder"),
                shipping_provider=process_env.get("OMNIMATCH_SHIPPING_PROVIDER", "placeholder"),
                memory_provider=process_env.get("OMNIMATCH_MEMORY_PROVIDER", "placeholder"),
                eval_provider=process_env.get("OMNIMATCH_EVAL_PROVIDER", "placeholder"),
                product_api_url=os.getenv("OMNIMATCH_PRODUCT_API_URL"),
                web_search_api_url=os.getenv("OMNIMATCH_WEB_SEARCH_API_URL"),
                **fork_limits,
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
                **fork_limits,
            )
        else:
            settings = cls(
                profile=profile,
                llm_provider=os.getenv("OMNIMATCH_LLM_PROVIDER", "openai"),
                llm_model=os.getenv("OMNIMATCH_LLM_MODEL", "gpt-4.1-mini"),
                product_provider=os.getenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi"),
                web_search_provider=os.getenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper"),
                shipping_provider=os.getenv("OMNIMATCH_SHIPPING_PROVIDER", "rate_table"),
                memory_provider=os.getenv("OMNIMATCH_MEMORY_PROVIDER", "memory"),
                eval_provider=os.getenv("OMNIMATCH_EVAL_PROVIDER", "heuristic"),
                product_api_url=os.getenv("OMNIMATCH_PRODUCT_API_URL"),
                web_search_api_url=os.getenv("OMNIMATCH_WEB_SEARCH_API_URL"),
                **fork_limits,
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
        if self.max_fork_depth < 0:
            raise ConfigError("max_fork_depth must be >= 0")
        if self.max_parallel_subagents < 1:
            raise ConfigError("max_parallel_subagents must be >= 1")
        if self.subagent_max_steps < 1:
            raise ConfigError("subagent_max_steps must be >= 1")
        if not math.isfinite(self.subagent_timeout_seconds) or self.subagent_timeout_seconds <= 0:
            raise ConfigError("subagent_timeout_seconds must be > 0")
        if self.profile == "test":
            return
        if self.profile == "dev":
            for name, provider in {
                "OMNIMATCH_LLM_PROVIDER": self.llm_provider,
                "OMNIMATCH_PRODUCT_PROVIDER": self.product_provider,
                "OMNIMATCH_WEB_SEARCH_PROVIDER": self.web_search_provider,
                "OMNIMATCH_SHIPPING_PROVIDER": self.shipping_provider,
            }.items():
                if provider == "placeholder":
                    raise ConfigError(f"{name}=placeholder is not allowed for dev profile")
        if self.llm_provider != "placeholder" and not os.getenv("OPENAI_API_KEY"):
            raise ConfigError(f"OPENAI_API_KEY is required for {self.profile} LLM provider")
        if self.product_provider == "serpapi":
            if not os.getenv("SERPAPI_API_KEY"):
                raise ConfigError(
                    f"SERPAPI_API_KEY is required for {self.profile} SerpApi product provider"
                )
        elif self.product_provider != "placeholder" and not self.product_api_url:
            raise ConfigError(
                f"OMNIMATCH_PRODUCT_API_URL is required for {self.profile} product provider"
            )
        if self.web_search_provider == "serper":
            if not os.getenv("SERPER_API_KEY"):
                raise ConfigError(
                    f"SERPER_API_KEY is required for {self.profile} Serper web search provider"
                )
            return
        if self.web_search_provider != "placeholder" and not self.web_search_api_url:
            raise ConfigError(
                f"OMNIMATCH_WEB_SEARCH_API_URL is required for {self.profile} web search provider"
            )

    @staticmethod
    def _mode_for(provider: str, fake_allowed: bool) -> ProviderMode:
        if provider == "placeholder":
            return "placeholder"
        if fake_allowed:
            return "fake"
        return "real"
