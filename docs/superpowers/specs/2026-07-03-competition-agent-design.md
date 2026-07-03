# OmniMatch Competition Agent Design

## Background

OmniMatch is no longer scoped as a teaching-oriented mock MVP. The active goal is
to build a competition-grade conversational shopping agent that can use real
external APIs during development, produce defensible recommendations, and remain
uploadable in environments where secrets or live API access are unavailable.

The current codebase is still useful as a runnable scaffold: it already has
FastAPI task APIs, WebSocket event streaming, a React console, typed schemas,
tool modules, and an async mock loop. The next stage should replace the fixed
mock chain with real agent behavior, real providers, traceable decisions, and an
evaluation loop.

## Goals

- Make real external APIs the default development path.
- Add a config layer that controls LLM, search, product, shipping, memory, and
  evaluation providers without code changes.
- Keep placeholder providers only for upload/submission packaging or explicit
  offline smoke tests.
- Upgrade the AgentLoop from a scripted tool sequence to a decision loop where
  observations affect the next action.
- Produce structured, explainable shopping recommendations with cited evidence,
  prices, constraints, and trade-offs.
- Persist traces, provider latency, errors, selected candidates, rejected
  candidates, and final scoring rationale for debugging and competition tuning.
- Add an evaluation harness with repeatable cases so prompt, ranking, and tool
  changes can be measured instead of judged by manual demos.

## Non-Goals

- No payment, order placement, account authorization, or logistics tracking.
- No anti-scraping evasion, proxy pool, or platform ToS bypass.
- No large distributed RL training stack in this phase.
- No requirement that every provider have production-grade quota management in
  the first implementation pass.
- Placeholder providers must not become the main product path. They exist to
  keep upload/submission artifacts runnable without secrets.

## Operating Profiles

Configuration is mandatory. Runtime behavior must be selected through
environment variables and config files, not by editing code.

### `dev`

`dev` is the normal development profile and is real API first.

- Missing required API keys fail fast at startup or task creation.
- LLM calls use a real configured model.
- Product and web search use real configured providers.
- Provider latency, raw response summaries, normalization warnings, and errors
  are written to the trace.
- Placeholder providers are allowed only when explicitly selected for one
  provider during local debugging.

### `submission`

`submission` is used when packaging or uploading files to a competition system
  where secrets cannot be included.

- Placeholder providers are the default.
- The agent still exercises the same schemas, tools, ranking logic, event stream,
  and frontend.
- Placeholder data must be deterministic and marked in traces with
  `provider_mode="placeholder"`.
- The final answer must disclose when placeholder evidence was used.

### `test`

`test` is deterministic and should not call network APIs.

- Providers are in-memory fakes or placeholder fixtures.
- Tests assert contracts, error handling, and ranking behavior.
- No real secrets are required.

## Configuration Requirements

The config layer should expose a single typed settings object loaded from
environment variables and optional `.env` values.

Required settings:

- `OMNIMATCH_PROFILE`: `dev`, `submission`, or `test`.
- `OMNIMATCH_LLM_PROVIDER`: provider id such as `openai`, `anthropic`, or
  `placeholder`.
- `OMNIMATCH_LLM_MODEL`: concrete model name.
- `OMNIMATCH_PRODUCT_PROVIDER`: provider id such as `serpapi`, `rapidapi`,
  `custom`, or `placeholder`.
- `OMNIMATCH_WEB_SEARCH_PROVIDER`: provider id such as `brave`, `tavily`,
  `serpapi`, or `placeholder`.
- `OMNIMATCH_SHIPPING_PROVIDER`: provider id such as `custom_rate_table` or
  `placeholder`.
- `OMNIMATCH_MEMORY_PROVIDER`: `memory`, `sqlite`, `opensearch`, or
  `placeholder`.
- `OMNIMATCH_EVAL_PROVIDER`: `llm_judge`, `heuristic`, or `placeholder`.
- Provider API keys are read from provider-specific env vars and never committed.

In `dev`, required keys for selected real providers must be present. In
`submission` and `test`, real provider keys are optional.

## Provider Architecture

Every external capability is accessed through a narrow async provider interface.
Tools call providers; providers call external APIs.

Core provider interfaces:

- `LLMProvider`: chat completion, structured output, and optional tool-call
  planning.
- `ProductSearchProvider`: query a platform or product source and return
  normalized candidates.
- `WebSearchProvider`: fetch open-web evidence and market context.
- `ShippingProvider`: estimate shipping, duties, taxes, and delivery window.
- `MemoryProvider`: read and write user preferences and negative constraints.
- `EvalProvider`: judge traces and final answers against rubrics.

Each provider result must include:

- `provider`: provider id.
- `provider_mode`: `real`, `placeholder`, or `fake`.
- `latency_ms`.
- `warnings`.
- Normalized data used by downstream tools.
- A redacted response summary suitable for trace logs.

Provider code must not leak raw secrets into events, output files, or frontend
payloads.

## Agent Architecture

The competition agent keeps the existing task API shape but replaces the scripted
mock loop.

The main loop is:

```text
Think -> Act -> Observe -> Reflect
```

The loop should continue until one of these terminal conditions is met:

- A high-confidence `ShoppingSummary` can be produced.
- The agent needs user clarification.
- Provider failures or evidence gaps prevent a defensible recommendation.
- A max-step or max-latency budget is reached.

The loop has access to these tool families:

- `Planner`: converts the user request into structured constraints and missing
  information.
- `MemoryRead` and `MemoryWrite`: applies persistent preferences.
- `WebSearch`: collects category, trend, and market evidence.
- `CategoryInsight`: summarizes useful product attributes and red flags.
- `ItemSearch`: searches configured product providers and platforms.
- `ShippingCalc`: estimates landed cost and delivery impact.
- `PriceCompare`: normalizes total cost and filters impossible candidates.
- `ItemPicker`: ranks candidates using constraints, evidence, and score weights.
- `ShoppingSummary`: returns the final recommendation with rationale.

Homogeneous sub-agent fork remains useful, but it should be driven by evidence
and budget:

- Fork platform searches when parallelism reduces latency.
- Fork large candidate review when context isolation prevents prompt pollution.
- Avoid fork when a single provider query is enough.

## Ranking And Recommendation

The final answer must be based on structured candidate scoring, not only LLM
language generation.

Candidate score inputs:

- Constraint satisfaction: budget, category, material, size, delivery, location.
- Evidence quality: source count, provider confidence, recency, field completeness.
- Total landed cost: price, shipping, tax, currency conversion, hidden fees.
- Preference fit: long-term preferences and current-query preferences.
- Risk penalties: missing price, unverifiable seller, unavailable product,
  suspiciously low price, weak evidence.

The summary should show:

- Top recommendations.
- Why each item was selected.
- Important rejected alternatives or trade-offs.
- Any uncertainty or missing evidence.
- Whether evidence came from real or placeholder providers.

## Event And Trace Requirements

The frontend event stream remains part of the product. Events should be useful
for debugging a real agent, not just demo progress.

Event types should include:

- `task_started`
- `thought`
- `tool_start`
- `tool_end`
- `provider_start`
- `provider_end`
- `provider_error`
- `subagent_started`
- `subagent_finished`
- `ranking_decision`
- `task_result`
- `task_error`

Trace files should be written under `output/{thread_id}/`:

- `summary.json`: final normalized result.
- `trace.jsonl`: step-by-step events and observations.
- `candidates.json`: normalized candidate pool and scores.
- `provider_calls.jsonl`: redacted provider requests/responses and latency.

Output writing failures must not discard a valid final answer. They should add a
warning to task state and emit a warning event.

## API And Frontend Requirements

The existing FastAPI routes can remain:

- `POST /api/tasks`
- `GET /api/tasks/{thread_id}`
- `WebSocket /ws/{thread_id}`

Required hardening:

- Unknown WebSocket `thread_id` must be rejected or closed with a clear reason.
- Task state should include `warnings`, `profile`, `provider_modes`, and
  `trace_paths`.
- Task failures must distinguish provider errors, validation errors, budget
  exhaustion, and internal exceptions.
- Reconnected WebSocket clients should receive historical events already stored
  in task state.

Frontend changes should stay thin:

- Show active profile and provider modes.
- Show provider latency and failures in the event stream.
- Render recommendations, rejected alternatives, uncertainty, and warnings.
- Avoid duplicating ranking or tool logic in React.

## Evaluation Strategy

The competition agent needs a small but serious evaluation loop before more API
surface is added.

Evaluation cases should cover:

- Budget-constrained shopping.
- Hard negative constraints, such as "no plastic".
- Ambiguous queries that require clarification.
- Cross-platform price comparison.
- Provider partial failure.
- Placeholder/submission profile behavior.

Each run should produce:

- Final answer.
- Candidate pool.
- Trace.
- Structured scores.
- Judge output or heuristic rubric output.

Core metrics:

- Constraint satisfaction.
- Evidence coverage.
- Ranking quality.
- Explanation quality.
- Robustness to provider failure.
- Latency and token/API cost.

## Security And Secrets

- `.env` may contain real local keys and must not be committed.
- `.env.example` documents all settings with empty or placeholder values.
- Trace logs must redact API keys, auth headers, and raw provider credentials.
- Submission artifacts should run without secrets using `OMNIMATCH_PROFILE=submission`.

## Acceptance Criteria

- The repository documents that the active direction is a competition-grade
  agent, not a teaching mock.
- A typed config layer can select `dev`, `submission`, or `test`.
- `dev` fails fast when a selected real provider lacks required credentials.
- `submission` runs without secrets using placeholder providers.
- Tools call provider interfaces instead of embedding API-specific code.
- The AgentLoop reacts to observations rather than executing only a fixed mock
  sequence.
- Final recommendations include structured scores, evidence, uncertainty, and
  warnings.
- Trace files are written for every task.
- Backend tests cover config, provider contracts, task failure, and ranking.
- Frontend build passes and the console shows provider/profile observability.
