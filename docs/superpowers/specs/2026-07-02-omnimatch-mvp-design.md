# OmniMatch MVP Design

> Superseded direction, 2026-07-03: this document described the original
> teaching/mock MVP. The active product direction is now the competition-grade
> agent design in
> `docs/superpowers/specs/2026-07-03-competition-agent-design.md`.

## Background

OmniMatch is a conversational cross-platform shopping Agent. The long-term product described in `idea.md` includes a main AgentLoop, homogeneous forked sub-AgentLoops, nine core tools, vector recall, long-term memory, context compression, AGUI events, and an evaluation/training loop.

The MVP will build a complete teaching-oriented project skeleton and one runnable mock end-to-end flow. It will not connect to real LLMs, real ecommerce APIs, real vector stores, or real training systems.

## Goals

- Create a project structure that maps clearly to `idea.md`.
- Run a frontend and backend locally.
- Let a user submit a shopping request from a React page.
- Start an asynchronous backend task with a generated `thread_id`.
- Stream AGUI-style events over WebSocket while the mock AgentLoop runs.
- Demonstrate mock tool execution and mock homogeneous sub-agent fork behavior.
- Render a final mock shopping summary in the frontend.
- Keep non-MVP systems present as clear stubs so later chapters can replace them incrementally.

## Non-Goals

- No real LLM integration.
- No real ecommerce platform API or scraping.
- No real vector recall service.
- No real OpenSearch, Faiss, Milvus, Redis, or database dependency.
- No real long-term preference persistence beyond stub behavior.
- No payment, order placement, logistics, OAuth, anti-scraping, privacy compliance, or RL training.
- No browser-level WebSocket end-to-end test requirement in the first MVP.

## Scope

The MVP uses the "complete directory skeleton with runnable mock chain" approach.

The backend should include these directories:

- `app/agent`
- `app/api`
- `app/tools`
- `app/recall`
- `app/memory`
- `app/compress`
- `app/eval`
- `app/prompt`
- `app/utils`

The repository should also include:

- `frontend`
- `examples`
- `tests`
- `docker`
- `output`
- `uploaded`
- `.env.example`
- `.python-version`
- `pyproject.toml`
- `uv.lock` after dependencies are resolved

Many modules will be stubs, but they should have explicit names and narrow interfaces matching their future responsibilities.

The backend uses `uv` with Python 3.10, matching `idea.md`. The frontend uses Vite, React, TypeScript, and plain CSS files scoped by component or page.

## User Experience

The frontend is a single-page teaching console.

The page is split into two main regions:

- Left side: query input, task status, final shopping summary, and mock product cards.
- Right side: real-time AGUI event stream and tool/sub-agent trace.

The user enters a shopping request, submits it, sees the task move into a running state, watches events appear as the mock AgentLoop advances, and finally sees a mock product list with purchase reasoning.

## Backend Architecture

The backend uses FastAPI and asyncio.

`app/api/server.py` owns the HTTP and WebSocket routes:

- `POST /api/tasks`: accepts `{ "query": "..." }`, creates a task, starts the mock AgentLoop in the background, and returns `{ "thread_id": "...", "status": "running" }`.
- `GET /api/tasks/{thread_id}`: returns current task state and final result if available.
- `WebSocket /ws/{thread_id}`: streams AGUI-style events for a task.

`app/api/connection.py` owns `ConnectionManager`, which tracks WebSocket connections by `thread_id`.

`app/api/monitor.py` owns event emission. Agent and tool code should call this module instead of importing WebSocket details directly.

`app/api/context.py` stores task-local context such as `thread_id` and `session_dir` using `ContextVar`, so lower layers can access task context without manually threading arguments through every function.

Task state can live in an in-memory registry for the MVP. The final summary should also be written to `output/{thread_id}/summary.json` to demonstrate output-file handling.

## Agent Architecture

`app/agent/main_agent.py` contains `MockAgentLoop`.

The loop simulates:

```text
Think -> Act -> Observe -> Reflect
```

It calls the mock tools in this order:

1. `Planner`
2. `CategoryInsight`
3. `ItemSearch`
4. `ShippingCalc`
5. `PriceCompare`
6. `ItemPicker`
7. `ShoppingSummary`

The loop emits events before and after important steps. The loop should be deterministic enough for tests and demos, while still using the user's query in generated messages and results.

`app/agent/dispatch_tool.py` implements mock homogeneous fork behavior. For cross-platform search, it creates async sub-tasks for platforms such as Amazon, eBay, AliExpress, and Shopee. Each sub-task gets a `sub-{uuid}` identifier, emits sub-agent events, returns mock products, and merges back into the main loop as if it were a tool result.

The MVP does not need real model reasoning. The fork decision is hardcoded for cross-platform item search.

## Tool Boundaries

Each core tool lives in its own module under `app/tools`.

MVP-active tools:

- `planner.py`: converts the user query into a simple structured shopping intent.
- `category_insight.py`: returns mock category hints and common attributes.
- `item_search.py`: returns mock products for one platform.
- `shipping_calc.py`: adds mock shipping and tax estimates.
- `price_compare.py`: calculates mock total prices.
- `item_picker.py`: selects the best few products according to the mock intent.
- `shopping_summary.py`: creates the final response object.

Stub-only tools:

- `chat_fallback.py`
- `web_search.py`

All tools should expose small async APIs and return typed data using Pydantic models. Shared request, event, task, and product schemas live in `app/schemas.py`; tool-local helper schemas may live beside the tool when they are not shared.

## Infrastructure Stubs

The following directories exist to preserve the teaching structure from `idea.md`:

- `app/recall`: user/query/item tower and ANN placeholders.
- `app/memory`: preference store and prompt injection placeholders.
- `app/compress`: cache breakpoint and compressor placeholders.
- `app/eval`: rubric, judge, and trace logger placeholders.
- `app/prompt`: prompt configuration placeholder.
- `app/utils`: path and thread-context helpers.

These stubs should not introduce external services. They should be importable and simple to test.

## Event Protocol

Events are JSON objects with a stable envelope:

```json
{
  "type": "tool_start",
  "thread_id": "thread_xxx",
  "timestamp": "2026-07-02T00:00:00Z",
  "run_id": "run_xxx",
  "tool": "ItemSearch",
  "message": "ItemSearch 正在跨 4 个平台并行检索...",
  "payload": {}
}
```

MVP event types:

- `task_started`
- `thought`
- `tool_start`
- `tool_end`
- `subagent_started`
- `subagent_finished`
- `task_result`
- `task_error`

The frontend should treat unknown event types as displayable log entries, so later chapters can add events without breaking the UI.

## Frontend Architecture

The frontend uses React and Vite.

The MVP page contains:

- Query input form.
- Current `thread_id` and task status.
- Final summary and product cards.
- Real-time event list.
- Tool/sub-agent trace grouped by event metadata when practical.

Frontend data flow:

1. Submit query to `POST /api/tasks`.
2. Store returned `thread_id`.
3. Open `WebSocket /ws/{thread_id}`.
4. Append incoming events to the event stream.
5. Render final summary from the `task_result` event.
6. If the WebSocket closes before result, allow the task status endpoint to be queried manually or on refresh in a later iteration.

The frontend should stay thin. It should not duplicate backend ranking, tool logic, or Agent behavior.

## Error Handling

If the Agent task raises an exception, the backend should:

- Mark the task status as `failed`.
- Emit `task_error`.
- Store a concise error message in task state.

If a WebSocket disconnects, the backend should keep the task running. Reconnection is supported by connecting again with the same `thread_id`. Historical event replay is out of scope for the MVP; the reconnected client receives only new events and can fetch final state through `GET /api/tasks/{thread_id}`.

If output-file writing fails, the task can still return the final result, but should include a warning event or warning field.

## Testing Strategy

Backend unit tests:

- Tool stubs return structured results.
- `ConnectionManager` can connect, disconnect, and route events.
- `MockAgentLoop` can produce a final shopping summary.
- Mock dispatch returns merged platform results.

Backend API tests:

- `POST /api/tasks` returns `thread_id` and `running`.
- `GET /api/tasks/{thread_id}` returns task state.

Frontend verification:

- Vite build or TypeScript check passes.
- Manual local run verifies query submission, WebSocket events, and final product rendering.

Browser-level WebSocket E2E tests are deferred until after the first runnable MVP.

## Acceptance Criteria

- The repository has a Git history containing this design spec.
- Backend dependencies install with `uv sync`.
- Frontend dependencies install with `npm install` inside `frontend`.
- Backend server starts locally.
- Frontend dev server starts locally.
- A user can submit one shopping request from the page.
- The right-side event stream updates while the backend task runs.
- The left-side result area shows a final mock shopping summary and product cards.
- The created directory structure corresponds to the structure in `idea.md`.
- Stub modules are importable and have clear responsibilities.
- Backend and frontend startup commands are documented separately in the repository README.
- Runtime directories `output` and `uploaded` are committed with `.gitkeep` files so the teaching skeleton is visible before the app runs.
