# OmniMatch

OmniMatch is now a competition-grade shopping agent project. The old mock MVP is
kept as historical context, but new development should target the provider-backed
competition agent.

The current app includes:

- FastAPI task APIs and WebSocket event replay.
- A provider-backed `CompetitionAgentLoop`.
- Configurable `dev`, `submission`, and `test` profiles.
- Real HTTP/OpenAI-compatible provider adapters for development.
- Deterministic placeholder providers for submission and smoke tests.
- Structured candidate scoring, trace files, and a React observability console.
- A small evaluation harness for repeatable regression cases.

## Backend

Install dependencies:

```bash
uv sync
```

Run the development profile with real providers:

```bash
cp .env.example .env
# fill OPENAI_API_KEY, OMNIMATCH_PRODUCT_API_URL, OMNIMATCH_PRODUCT_API_KEY,
# OMNIMATCH_WEB_SEARCH_API_URL, and OMNIMATCH_WEB_SEARCH_API_KEY
# optionally fill OPENAI_BASE_URL for OpenAI-compatible providers
OMNIMATCH_PROFILE=dev uv run uvicorn app.api.server:app --reload
```

Run the submission profile without secrets:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
OMNIMATCH_PROFILE=submission uv run pytest -v
```

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
OMNIMATCH_PROFILE=submission uv run uvicorn app.api.server:app --reload
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
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

The legacy mock entrypoint remains as a compatibility wrapper:

```bash
uv run python examples/run_mock_agent.py
```
