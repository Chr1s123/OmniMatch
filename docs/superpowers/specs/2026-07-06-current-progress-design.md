# Current Progress Design

## Background

The repository moved from the original mock MVP into a provider-backed
competition-agent implementation. Several implementation plans were written
before the latest commits, so their checkbox state no longer matched the actual
code and verification results.

## Current State

As of 2026-07-06, the active baseline is:

- MVP scaffold: complete and superseded.
- Competition config and provider architecture: implemented.
- Submission profile contract: implemented and verified.
- Observation-driven agent loop: implemented and verified.
- API/frontend observability: implemented.
- Evaluation harness: implemented with a small smoke fixture.

Verified commands:

```bash
uv run pytest -q
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
cd frontend && npm run build
```

Observed results:

- Backend tests: `48 passed, 1 warning`.
- Submission smoke: exits 0 and discloses placeholder evidence.
- Frontend build: exits 0.

## Next Direction

The next useful work is not more scaffold construction. The next work should
validate and improve the real competition path:

- Run `dev` with real provider credentials and capture normalization gaps.
- Add `provider_calls.jsonl` if the separate provider-call audit file remains a
  hard requirement.
- Expand evaluation cases for ambiguous queries, provider partial failure,
  cross-platform price comparison, and hard negative constraints.
- Tune prompts/ranking against evaluation output.
- Decide whether to commit/push any local documentation updates.

## Out Of Scope

- Rebuilding the mock MVP.
- Rewriting the React UI before the real-provider path is measured.
- Adding new provider families before existing adapters are validated end to
  end.
