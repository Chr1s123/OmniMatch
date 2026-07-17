# OmniMatch

OmniMatch is now a competition-grade shopping agent project. The old mock MVP is
kept as historical context, but new development should target the provider-backed
competition agent.

The current app includes:

- FastAPI task APIs and WebSocket event replay.
- A provider-backed `CompetitionAgentLoop`.
- Configurable `dev`, `submission`, and `test` profiles.
- Real HTTP/OpenAI-compatible provider adapters for runtime use.
- Deterministic placeholder providers for submission packaging, tests, and
  explicit local fakes.
- Structured candidate scoring, trace files, and a React observability console.
- A small evaluation harness for repeatable regression cases.

Homogeneous sub-agent work uses bounded forks of `CompetitionAgentLoop`. Each child has
isolated tool state, an allowlist, step/time budgets, scoped events, and structured merge
results; the removed `dispatch_tool.py` function-level mock is no longer the sub-agent path.

## Backend

Install dependencies:

```bash
uv sync
```

Run the development profile with real providers:

```bash
cp .env.example .env
# fill OPENAI_API_KEY, SERPAPI_API_KEY for OMNIMATCH_PRODUCT_PROVIDER=serpapi,
# and SERPER_API_KEY for OMNIMATCH_WEB_SEARCH_PROVIDER=serper
# optionally fill OPENAI_BASE_URL for OpenAI-compatible providers
OMNIMATCH_PROFILE=dev uv run uvicorn app.api.server:app --reload
```

Run the submission profile without real provider secrets:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

`submission` defaults all providers to deterministic placeholders so uploaded
artifacts can run in environments without secrets. If you explicitly select a
real provider in `submission`, the matching API key is still required.

Run backend tests:

```bash
uv run pytest -v
```

## Frontend

Install and run the React console:

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL:

```text
http://127.0.0.1:5173
```

Build the frontend:

```bash
cd frontend
npm run build
```

## Local App Run

Start the backend:

```bash
OMNIMATCH_PROFILE=dev uv run uvicorn app.api.server:app --reload
```

Start the frontend in another terminal:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The console shows task status, profile, provider modes, provider latency events,
recommendations, warnings, and trace file paths.

Trace output is written under:

```text
output/{thread_id}/
```

## CLI Examples

Run the competition loop directly:

```bash
OMNIMATCH_PROFILE=dev uv run python examples/run_competition_agent.py
```

The legacy mock entrypoint remains as a compatibility wrapper:

```bash
uv run python examples/run_mock_agent.py
```
