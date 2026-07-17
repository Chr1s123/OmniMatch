# Real API Worktree Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the homogeneous-AgentLoop feature worktree reuse the existing real API configuration and verify each external provider with exactly one minimal request.

**Architecture:** Add only an ignored relative `.env` symlink in the feature worktree, leaving tracked provider defaults unchanged. Validate configuration without revealing secrets, then call the configured LLM, SerpApi product search, and Serper web search once each and report only normalized, non-secret metadata.

**Tech Stack:** zsh, Python 3.10, python-dotenv, existing OmniMatch provider adapters.

## Global Constraints

- Do not modify or commit secret values.
- Do not change deterministic `test` or `submission` defaults.
- The worktree `.env` must be an ignored relative symlink to the main checkout `.env`.
- Do not overwrite an existing worktree `.env`.
- Print provider names, modes, counts, actions, and latency only; never print keys or authorization headers.
- Make at most one request to each external provider and do not retry failures.
- Rate-table shipping, in-memory state, and heuristic evaluation require no external request.

---

## File Structure

- Create, ignored: `.env`
  - Relative symlink to `../../.env`; supplies the existing real provider settings to the worktree.
- No tracked runtime source or test files are created or modified.

### Task 1: Link And Validate The Real Provider Configuration

**Files:**
- Create, ignored: `.env` (relative symlink to `../../.env`)
- Verify: `app/config.py`
- Verify: `app/providers/registry.py`

**Interfaces:**
- Consumes: `OmniMatchSettings.from_env() -> OmniMatchSettings`
- Consumes: `OmniMatchSettings.provider_modes() -> dict[str, ProviderMode]`
- Consumes: `ProviderRegistry.from_settings(settings) -> ProviderRegistry`
- Produces: a worktree runtime configuration in which every provider mode is `real`

- [ ] **Step 1: Verify that no worktree `.env` entry exists**

Run from `/Users/chris/develop/OmniMatch/.worktrees/homogeneous-agentloop-fork`:

```bash
test ! -e .env
```

Expected: exit code `0`. If this fails, stop without changing the existing entry.

- [ ] **Step 2: Create the ignored relative symlink**

Run:

```bash
ln -s ../../.env .env
```

Expected: exit code `0` and `.env` resolves to the main checkout configuration.

- [ ] **Step 3: Verify symlink provenance and Git ignore behavior**

Run:

```bash
test -L .env
readlink .env
git check-ignore -v .env
git status --short
```

Expected:

- `readlink .env` prints `../../.env`.
- `git check-ignore` identifies the repository `.gitignore` rule for `.env`.
- `git status --short` does not list `.env`.

- [ ] **Step 4: Load and validate settings without making an API request**

Run:

```bash
/Users/chris/develop/OmniMatch/.venv/bin/python -c 'from app.config import OmniMatchSettings; s = OmniMatchSettings.from_env(); assert s.profile == "dev"; assert s.llm_model == "qwen3.5-flash"; print({"providers": {"llm": s.llm_provider, "product": s.product_provider, "web_search": s.web_search_provider, "shipping": s.shipping_provider, "memory": s.memory_provider, "eval": s.eval_provider}, "modes": s.provider_modes()})'
```

Expected: the profile and LLM model assertions pass without printing either value; providers are `openai`, `serpapi`, `serper`, `rate_table`, `memory`, and `heuristic`; every value in `modes` is `real`. The output contains no secret values.

### Task 2: Make Three Minimal Real Provider Requests

**Files:**
- Verify: `app/providers/openai_llm.py`
- Verify: `app/providers/serpapi_product.py`
- Verify: `app/providers/serper_web_search.py`

**Interfaces:**
- Consumes: `LLMProvider.plan_next_action(messages) -> ProviderResult[dict[str, Any]]`
- Consumes: `ProductSearchProvider.search(query, platforms) -> ProviderResult[list[dict[str, Any]]]`
- Consumes: `WebSearchProvider.search(query) -> ProviderResult[list[dict[str, Any]]]`
- Produces: one non-secret connectivity result for each external provider

- [ ] **Step 1: Validate the real LLM with one request**

Run with network permission:

```bash
/Users/chris/develop/OmniMatch/.venv/bin/python -c 'import asyncio; from app.config import OmniMatchSettings; from app.providers.registry import ProviderRegistry; r = ProviderRegistry.from_settings(OmniMatchSettings.from_env()); x = asyncio.run(r.llm.plan_next_action([{"role": "system", "content": "Return only a JSON object with action and arguments. Use action finish."}, {"role": "user", "content": "Connectivity check."}])); print({"provider": x.provider, "mode": x.provider_mode, "action": x.data.get("action"), "latency_ms": x.latency_ms})'
```

Expected: exit code `0`, `mode` is `real`, and `action` is present. This is exactly one LLM request.

- [ ] **Step 2: Validate SerpApi product search with one request**

Run with network permission:

```bash
/Users/chris/develop/OmniMatch/.venv/bin/python -c 'import asyncio; from app.config import OmniMatchSettings; from app.providers.registry import ProviderRegistry; r = ProviderRegistry.from_settings(OmniMatchSettings.from_env()); x = asyncio.run(r.product.search("wireless headphones", [])); print({"provider": x.provider, "mode": x.provider_mode, "result_count": len(x.data), "latency_ms": x.latency_ms})'
```

Expected: exit code `0`, provider is `serpapi_product`, mode is `real`, and `result_count` is present. This is exactly one SerpApi request.

- [ ] **Step 3: Validate Serper web search with one request**

Run with network permission:

```bash
/Users/chris/develop/OmniMatch/.venv/bin/python -c 'import asyncio; from app.config import OmniMatchSettings; from app.providers.registry import ProviderRegistry; r = ProviderRegistry.from_settings(OmniMatchSettings.from_env()); x = asyncio.run(r.web_search.search("wireless headphones material guide")); print({"provider": x.provider, "mode": x.provider_mode, "result_count": len(x.data), "latency_ms": x.latency_ms})'
```

Expected: exit code `0`, provider is `serper`, mode is `real`, and `result_count` is present. This is exactly one Serper request.

- [ ] **Step 4: Report failures without retries**

For any failed command, report the provider name and the adapter's sanitized error. Classify it as missing configuration, authentication, quota, network, or response compatibility. Do not retry and do not expose environment values.

- [ ] **Step 5: Verify the tracked worktree remains clean**

Run:

```bash
git status --short --branch
git diff --check
```

Expected: the branch has no tracked modifications and `.env` is absent from status output.

No commit is required for the ignored symlink or verification output.

## Plan Self-Review

- Spec coverage: the plan covers safe `.env` reuse, non-secret configuration validation, one request per external provider, no retries, and clean Git state.
- Placeholder scan: there are no unresolved implementation markers.
- Interface consistency: all provider calls match the existing protocol signatures in `app/providers/base.py`.
