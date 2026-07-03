# Submission Profile Contract Design

## Background

OmniMatch has three runtime profiles: `dev`, `submission`, and `test`. The active
competition-agent design requires `dev` to use real providers by default, while
`submission` must remain uploadable and runnable in environments where API
secrets are unavailable.

The previous implementation treated `submission` like `dev`: it defaulted to
OpenAI, SerpApi, Serper, and rate-table providers, then failed when API keys were
missing. That contradicted the competition packaging requirement.

## Goal

Restore the profile contract so `OMNIMATCH_PROFILE=submission` runs without real
secrets by default, while preserving fail-fast validation when real providers
are explicitly selected.

## Scope

This change only covers configuration defaults, validation, tests, and docs.

In scope:

- `submission` defaults LLM, product, web search, shipping, memory, and eval
  providers to `placeholder`.
- `submission` reports all default provider modes as `placeholder`.
- `submission` ignores real-provider defaults loaded from a local `.env` file
  when the current process only sets `OMNIMATCH_PROFILE=submission`.
- `submission` permits explicit real providers when provider selector variables
  are set in the current process environment.
- Explicit real providers in `submission` still require their matching keys or
  URLs.
- `dev` continues to default to real providers and fail fast when required keys
  are missing.
- `test` continues to avoid network providers.

Out of scope:

- AgentLoop decision-making.
- Ranking changes.
- Provider adapter behavior changes.
- Frontend UI changes.
- Evaluation harness changes.

## Runtime Behavior

With only `OMNIMATCH_PROFILE=submission` set:

```text
llm_provider=placeholder
product_provider=placeholder
web_search_provider=placeholder
shipping_provider=placeholder
memory_provider=placeholder
eval_provider=placeholder
provider_modes={all: placeholder}
```

With `OMNIMATCH_PROFILE=submission` and explicit real providers:

```text
OMNIMATCH_LLM_PROVIDER=openai requires OPENAI_API_KEY
OMNIMATCH_PRODUCT_PROVIDER=serpapi requires SERPAPI_API_KEY
OMNIMATCH_PRODUCT_PROVIDER=http_product requires OMNIMATCH_PRODUCT_API_URL
OMNIMATCH_WEB_SEARCH_PROVIDER=serper requires SERPER_API_KEY
OMNIMATCH_WEB_SEARCH_PROVIDER=http_web_search requires OMNIMATCH_WEB_SEARCH_API_URL
```

`OMNIMATCH_SHIPPING_PROVIDER=rate_table` is considered a real local provider
and does not require a network API key.

For local development, `.env` often contains `dev` provider selectors and real
keys. Running `OMNIMATCH_PROFILE=submission ...` must not accidentally reuse
those `.env` provider selectors. Real providers in `submission` must be selected
explicitly in the current process environment, for example:

```bash
OMNIMATCH_PROFILE=submission OMNIMATCH_LLM_PROVIDER=openai OPENAI_API_KEY=... uv run python examples/run_competition_agent.py
```

## Testing

The config tests must prove the corrected contract:

- `dev` still requires real provider credentials.
- default `submission` loads without credentials and uses placeholders.
- default `submission` ignores real-provider selector values loaded from `.env`.
- explicit real providers in `submission` require credentials.
- `test` remains fake or placeholder only.
- SerpApi and Serper provider-specific key validation still works.

The full backend test suite must pass after the config change.

## Documentation

The README must say that `submission` runs without real provider secrets by
default. `.env.example` must keep `dev` as the real-provider example while
making clear that `submission` does not require those secrets.
