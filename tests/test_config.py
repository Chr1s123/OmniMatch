import os

import pytest

from app.config import ConfigError, OmniMatchSettings


def clear_omnimatch_env(monkeypatch):
    monkeypatch.setattr("app.config.load_dotenv", lambda: None)
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


def test_submission_profile_defaults_to_placeholder_without_keys(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "submission"
    assert settings.llm_provider == "placeholder"
    assert settings.product_provider == "placeholder"
    assert settings.web_search_provider == "placeholder"
    assert settings.shipping_provider == "placeholder"
    assert settings.memory_provider == "placeholder"
    assert settings.eval_provider == "placeholder"
    assert set(settings.provider_modes().values()) == {"placeholder"}


def test_submission_profile_ignores_dotenv_real_provider_defaults(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")

    def load_dev_dotenv_values() -> None:
        monkeypatch.setenv("OMNIMATCH_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
        monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi")
        monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
        monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper")
        monkeypatch.setenv("SERPER_API_KEY", "serper-key")
        monkeypatch.setenv("OMNIMATCH_SHIPPING_PROVIDER", "rate_table")

    monkeypatch.setattr("app.config.load_dotenv", load_dev_dotenv_values)

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "submission"
    assert settings.llm_provider == "placeholder"
    assert settings.product_provider == "placeholder"
    assert settings.web_search_provider == "placeholder"
    assert settings.shipping_provider == "placeholder"
    assert set(settings.provider_modes().values()) == {"placeholder"}


def test_submission_profile_uses_real_provider_env_values(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")
    monkeypatch.setenv("OMNIMATCH_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")
    monkeypatch.setenv("OMNIMATCH_SHIPPING_PROVIDER", "rate_table")

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "submission"
    assert settings.llm_provider == "openai"
    assert settings.product_provider == "serpapi"
    assert settings.web_search_provider == "serper"
    assert settings.shipping_provider == "rate_table"
    assert settings.provider_modes()["llm"] == "real"
    assert settings.provider_modes()["product"] == "real"
    assert settings.provider_modes()["web_search"] == "real"
    assert settings.provider_modes()["shipping"] == "real"
    assert settings.provider_modes()["memory"] == "placeholder"
    assert settings.provider_modes()["eval"] == "placeholder"


def test_submission_profile_requires_keys_only_for_explicit_real_providers(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")
    monkeypatch.setenv("OMNIMATCH_LLM_PROVIDER", "openai")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OmniMatchSettings.from_env()


def test_test_profile_uses_fake_or_placeholder_without_network(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "test")

    settings = OmniMatchSettings.from_env()

    assert settings.profile == "test"
    assert set(settings.provider_modes().values()) <= {"fake", "placeholder"}


def test_dev_profile_serper_web_search_uses_serper_api_key_without_web_search_url(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")

    settings = OmniMatchSettings.from_env()

    assert settings.web_search_provider == "serper"
    assert settings.web_search_api_url is None


def test_dev_profile_serpapi_product_uses_serpapi_key_without_product_url(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi")
    monkeypatch.setenv("SERPAPI_API_KEY", "serpapi-key")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")

    settings = OmniMatchSettings.from_env()

    assert settings.product_provider == "serpapi"
    assert settings.product_api_url is None


def test_dev_profile_serpapi_product_requires_serpapi_key(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "dev")
    monkeypatch.setenv("OPENAI_API_KEY", "llm-key")
    monkeypatch.setenv("OMNIMATCH_PRODUCT_PROVIDER", "serpapi")
    monkeypatch.setenv("OMNIMATCH_WEB_SEARCH_PROVIDER", "serper")
    monkeypatch.setenv("SERPER_API_KEY", "serper-key")

    with pytest.raises(ConfigError, match="SERPAPI_API_KEY"):
        OmniMatchSettings.from_env()
