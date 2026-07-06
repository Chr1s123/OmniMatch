# Submission Profile Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Current Progress - 2026-07-06

Status: implementation complete and verified.

- Steps 1-10 are complete.
- Step 11 remains only if the current branch still needs publish housekeeping.
- Current git log includes `ff0b4cc fix: restore submission profile placeholder defaults`.
- Verification rerun on 2026-07-06:
  - `uv run pytest -q` -> `48 passed, 1 warning`
  - `OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py` -> exits 0 with placeholder-provider warning

**Goal:** Make `OMNIMATCH_PROFILE=submission` default to deterministic placeholder providers and run without real provider secrets.

**Architecture:** Keep profile behavior centralized in `app/config.py`. Tests define the contract in `tests/test_config.py`; README and `.env.example` document the runtime expectation without changing provider adapters, AgentLoop behavior, ranking, or frontend code.

**Tech Stack:** Python 3.10, uv, pytest, python-dotenv, dataclass-based settings.

## Global Constraints

- `OMNIMATCH_PROFILE=dev` defaults to real providers and fails fast when required provider credentials are absent.
- `OMNIMATCH_PROFILE=submission` defaults every provider family to `placeholder`.
- `OMNIMATCH_PROFILE=submission` must run without `OPENAI_API_KEY`, `SERPAPI_API_KEY`, or `SERPER_API_KEY` when providers are left at their defaults.
- `OMNIMATCH_PROFILE=submission` must ignore real-provider selector values loaded from a local `.env` when the current process only sets the profile.
- Explicit real providers in `submission` must still validate their required keys or URLs.
- `OMNIMATCH_PROFILE=test` must not call network APIs.
- This plan does not change AgentLoop decision-making, ranking, provider adapters, frontend UI, or eval scoring.

---

## File Structure

- `app/config.py`: profile-specific defaults and provider credential validation.
- `tests/test_config.py`: regression tests for `dev`, `submission`, and `test` contracts.
- `README.md`: user-facing run instructions for `dev` and `submission`.
- `.env.example`: default real-provider `dev` example plus submission no-secret note.
- `docs/superpowers/specs/2026-07-03-submission-profile-contract-design.md`: design contract.
- `docs/superpowers/plans/2026-07-03-submission-profile-contract-implementation.md`: implementation record.

---

### Task 1: Lock The Submission Profile Contract

**Files:**
- Modify: `tests/test_config.py`
- Modify: `app/config.py`
- Modify: `README.md`
- Modify: `.env.example`
- Create: `docs/superpowers/specs/2026-07-03-submission-profile-contract-design.md`
- Create: `docs/superpowers/plans/2026-07-03-submission-profile-contract-implementation.md`

**Interfaces:**
- Consumes: `OmniMatchSettings.from_env() -> OmniMatchSettings`
- Consumes: `OmniMatchSettings.provider_modes() -> dict[str, ProviderMode]`
- Produces: unchanged public API with corrected `submission` defaults and validation.

- [x] **Step 1: Write the failing regression test**

Add this test to `tests/test_config.py`:

```python
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
```

- [x] **Step 2: Verify the test fails against the old behavior**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected old-behavior failure:

```text
FAILED tests/test_config.py::test_submission_profile_defaults_to_placeholder_without_keys
ConfigError: OPENAI_API_KEY is required for dev LLM provider
```

- [x] **Step 3: Correct `OmniMatchSettings.from_env()` defaults**

In `app/config.py`, branch `submission` before `test` and use placeholder
defaults:

```python
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
        product_api_url=os.getenv("OMNIMATCH_PRODUCT_API_URL"),
        web_search_api_url=os.getenv("OMNIMATCH_WEB_SEARCH_API_URL"),
    )
```

- [x] **Step 4: Correct validation**

In `app/config.py`, only reject placeholder providers for `dev`; allow them for
`submission` and still validate explicitly selected real providers:

```python
if self.profile == "dev":
    for name, provider in {
        "OMNIMATCH_LLM_PROVIDER": self.llm_provider,
        "OMNIMATCH_PRODUCT_PROVIDER": self.product_provider,
        "OMNIMATCH_WEB_SEARCH_PROVIDER": self.web_search_provider,
        "OMNIMATCH_SHIPPING_PROVIDER": self.shipping_provider,
    }.items():
        if provider == "placeholder":
            raise ConfigError(f"{name}=placeholder is not allowed for dev profile")
```

- [x] **Step 5: Preserve explicit real-provider validation in submission**

Add this test to `tests/test_config.py`:

```python
def test_submission_profile_requires_keys_only_for_explicit_real_providers(monkeypatch):
    clear_omnimatch_env(monkeypatch)
    monkeypatch.setenv("OMNIMATCH_PROFILE", "submission")
    monkeypatch.setenv("OMNIMATCH_LLM_PROVIDER", "openai")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        OmniMatchSettings.from_env()
```

- [x] **Step 6: Prevent local `.env` dev values from changing submission defaults**

Add this test to `tests/test_config.py`:

```python
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
```

Update `app/config.py` so submission provider selectors are read from the
pre-`.env` process environment:

```python
process_env = dict(os.environ)
load_dotenv()
profile = process_env.get("OMNIMATCH_PROFILE") or os.getenv("OMNIMATCH_PROFILE", "dev")
```

- [x] **Step 7: Update docs**

Update `README.md` so the submission command is documented as no-secret by
default:

````markdown
Run the submission profile without real provider secrets:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

`submission` defaults all providers to deterministic placeholders so uploaded
artifacts can run in environments without secrets. If you explicitly select a
real provider in `submission`, the matching API key is still required.
````

Update `.env.example` with this note:

```dotenv
# dev uses real providers and requires the keys below.
# submission defaults to placeholder providers and can run without secrets.
OMNIMATCH_PROFILE=dev
```

- [x] **Step 8: Run targeted verification**

Run:

```bash
uv run pytest tests/test_config.py -q
```

Expected:

```text
9 passed
```

- [x] **Step 9: Run full backend verification**

Run:

```bash
uv run pytest -q
```

Expected:

```text
all tests pass
```

- [x] **Step 10: Run no-secret submission smoke**

Run:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

Expected:

```text
The command exits 0 and the summary warnings include provider modes: placeholder.
```

- [ ] **Step 11: Commit and push**

Run:

```bash
git add .env.example README.md app/config.py tests/test_config.py docs/superpowers/specs/2026-07-03-submission-profile-contract-design.md docs/superpowers/plans/2026-07-03-submission-profile-contract-implementation.md
git commit -m "fix: restore submission profile placeholder defaults"
git push
```
