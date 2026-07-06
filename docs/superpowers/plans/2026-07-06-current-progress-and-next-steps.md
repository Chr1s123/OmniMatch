# Current Progress And Next Steps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reconcile OmniMatch documentation with the implemented competition-agent baseline and define the next concrete work.

**Architecture:** Treat existing MVP, submission-profile, and observation-loop plans as completed historical records. Use this plan as the current handoff entry point for the next implementation phase.

**Tech Stack:** Python 3.10, uv, FastAPI, Pydantic, pytest, React, Vite, TypeScript.

## Global Constraints

- `OMNIMATCH_PROFILE=submission` must keep running without real secrets.
- `OMNIMATCH_PROFILE=test` must not call network APIs.
- `OMNIMATCH_PROFILE=dev` remains real-provider first and fails fast without required credentials.
- Documentation updates must not change runtime behavior.
- Verification evidence must come from commands run in the current workspace.

---

## Current Progress

Completed and verified:

- Original mock MVP scaffold.
- Profile config and provider-mode validation.
- Provider contracts, registry, placeholder providers, real HTTP/search/LLM adapters, and shipping adapter.
- Structured ranking and recommendation summary output.
- FastAPI task state hardening, WebSocket replay, trace-path exposure, and React observability console.
- Evaluation harness with a smoke fixture.
- Submission profile no-secret contract.
- Observation-driven `CompetitionAgentLoop` with action normalization, dynamic LLM-selected tool execution, terminal actions, trace persistence, clarification handling, and max-step exhaustion handling.

Latest verification:

```bash
uv run pytest -q
# 48 passed, 1 warning

OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
# exits 0; summary warnings include placeholder provider modes

cd frontend && npm run build
# exits 0
```

## Current Gaps

- `provider_calls.jsonl` is specified in the competition design but not currently written as a separate file.
- Real-provider behavior needs an end-to-end `dev` profile validation pass with actual credentials.
- Evaluation coverage is still small and should be expanded before ranking/prompt tuning.
- The local documentation updates need commit/push if they should be preserved in git history.

### Task 1: Commit Documentation Reconciliation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-02-omnimatch-mvp-design.md`
- Modify: `docs/superpowers/specs/2026-07-03-competition-agent-design.md`
- Modify: `docs/superpowers/specs/2026-07-03-submission-profile-contract-design.md`
- Modify: `docs/superpowers/plans/2026-07-02-omnimatch-mvp-implementation.md`
- Modify: `docs/superpowers/plans/2026-07-03-submission-profile-contract-implementation.md`
- Modify: `docs/superpowers/plans/2026-07-03-observation-driven-agent-loop-implementation.md`
- Create: `docs/superpowers/specs/2026-07-06-current-progress-design.md`
- Create: `docs/superpowers/plans/2026-07-06-current-progress-and-next-steps.md`

**Interfaces:**
- Consumes: current git state and verification outputs.
- Produces: an accurate current handoff entry point for future work.

- [x] **Step 1: Inspect existing plans and specs**

Run:

```bash
rg --files -g '*plan*' -g '*spec*' docs/superpowers
```

- [x] **Step 2: Verify implementation state**

Run:

```bash
uv run pytest -q
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
cd frontend && npm run build
```

- [x] **Step 3: Update status blocks and create current handoff docs**

Record completed work, remaining gaps, and next implementation direction in
the files listed above.

- [ ] **Step 4: Commit docs**

Run:

```bash
git add docs/superpowers
git commit -m "docs: reconcile current progress and next steps"
```

### Task 2: Add Provider Call Audit Output

**Files:**
- Modify: `app/agent/main_agent.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: provider observations already appended to `ToolContext.observations`.
- Produces: `output/{thread_id}/provider_calls.jsonl`.

- [ ] **Step 1: Add a failing test**

Assert that a completed submission loop writes `provider_calls.jsonl` and that
each row contains `provider`, `provider_mode`, `latency_ms`, and `tool`.

- [ ] **Step 2: Run the focused test**

Run:

```bash
uv run pytest tests/test_agent_loop.py -q
```

- [ ] **Step 3: Implement provider-call extraction**

Write one JSONL row for each observation containing a provider field. Redact any
response summaries before persistence.

- [ ] **Step 4: Verify**

Run:

```bash
uv run pytest -q
```

### Task 3: Real Provider Validation Pass

**Files:**
- Modify only after failures are reproduced in adapters or normalization.

**Interfaces:**
- Consumes: `.env` real-provider settings.
- Produces: a short findings note and targeted adapter fixes if needed.

- [ ] **Step 1: Run dev profile smoke with real credentials**

Run:

```bash
OMNIMATCH_PROFILE=dev uv run python examples/run_competition_agent.py
```

- [ ] **Step 2: Record failures**

Classify failures as config, auth, provider response shape, normalization,
ranking, or prompt/action-selection issues.

- [ ] **Step 3: Fix only reproduced failures**

Use focused tests around the failing provider or normalization path.

- [ ] **Step 4: Verify no-secret profile still works**

Run:

```bash
uv run pytest -q
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

### Task 4: Expand Evaluation Cases

**Files:**
- Modify: `app/eval/fixtures/competition_smoke.jsonl`
- Modify: `tests/test_eval_runner.py`

**Interfaces:**
- Consumes: existing evaluation runner.
- Produces: broader regression coverage for competition-agent changes.

- [ ] **Step 1: Add cases**

Add cases for ambiguous query clarification, cross-platform price comparison,
provider partial failure, and hard negative constraints.

- [ ] **Step 2: Update eval runner tests**

Assert the runner loads multiple cases and reports per-case result rows.

- [ ] **Step 3: Verify**

Run:

```bash
uv run pytest tests/test_eval_runner.py -q
uv run pytest -q
```
