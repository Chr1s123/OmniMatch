import os

import pytest

from app.config import ConfigError, OmniMatchSettings


def clear_omnimatch_env(monkeypatch):
    for key in list(os.environ):
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
