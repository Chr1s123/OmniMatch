# OmniMatch MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Current Progress - 2026-07-06

Status: completed as historical scaffold and superseded by the competition-agent
plans.

- Backend schemas, tools, API, WebSocket path, examples, and tests exist.
- Frontend React/Vite console exists and builds successfully.
- The active runtime is now `CompetitionAgentLoop`; `MockAgentLoop` is retained
  as an alias for compatibility.
- Verification rerun on 2026-07-06:
  - `uv run pytest -q` -> `48 passed, 1 warning`
  - `cd frontend && npm run build` -> exits 0

**Goal:** Build a runnable mock OmniMatch MVP with a complete teaching skeleton, FastAPI backend, WebSocket AGUI-style events, mock AgentLoop/tool chain, and React/Vite frontend console.

**Architecture:** The backend exposes task HTTP/WebSocket APIs and runs an in-memory async mock AgentLoop. Tools return Pydantic schemas, dispatch simulates homogeneous forked sub-agents, and the frontend stays thin by rendering events and final results from backend messages.

**Tech Stack:** Python 3.10, uv, FastAPI, Pydantic, pytest, httpx, React, Vite, TypeScript, plain CSS.

## Global Constraints

- Backend uses `uv` with Python 3.10.
- Frontend uses Vite, React, TypeScript, and plain CSS.
- No real LLM integration.
- No real ecommerce API, scraping, vector database, Redis, or database dependency.
- Shared schemas live in `app/schemas.py`.
- Runtime directories `output` and `uploaded` are committed with `.gitkeep`.
- Backend and frontend startup commands are documented separately in `README.md`.
- Implementation happens directly on `main`, per user instruction.

---

### Task 1: Backend Project Skeleton and Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/schemas.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Produces: `ShoppingQuery`, `AgentEvent`, `Product`, `ShoppingSummary`, `TaskState` Pydantic models.
- Produces: task statuses `running`, `completed`, `failed`.

- [ ] **Step 1: Write failing schema tests**

```python
from app.schemas import AgentEvent, Product, ShoppingQuery, ShoppingSummary, TaskState


def test_shopping_query_strips_query_text():
    query = ShoppingQuery(query="  旅行三件套，不要塑料  ")
    assert query.query == "旅行三件套，不要塑料"


def test_product_total_price_includes_shipping_and_tax():
    product = Product(
        id="p1",
        platform="Amazon",
        title="旅行收纳三件套",
        price=199.0,
        currency="CNY",
        shipping=20.0,
        tax=5.0,
        rating=4.6,
        reason="便宜耐用",
        url="https://example.com/p1",
    )
    assert product.total_price == 224.0


def test_task_state_defaults_to_running():
    state = TaskState(thread_id="thread_abc")
    assert state.status == "running"
    assert state.events == []


def test_summary_contains_products_and_message():
    product = Product(
        id="p1",
        platform="eBay",
        title="帆布旅行套装",
        price=180,
        currency="CNY",
        shipping=30,
        tax=0,
        rating=4.4,
        reason="材质非塑料",
        url="https://example.com/p1",
    )
    summary = ShoppingSummary(message="推荐 1 件商品", products=[product])
    assert summary.products[0].total_price == 210
    assert "推荐" in summary.message


def test_agent_event_has_display_fields():
    event = AgentEvent(
        type="tool_start",
        thread_id="thread_abc",
        run_id="run_abc",
        tool="Planner",
        message="Planner 正在拆解需求...",
    )
    assert event.type == "tool_start"
    assert event.payload == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: fails because `pyproject.toml`, dependencies, or `app.schemas` do not exist yet.

- [ ] **Step 3: Implement project config and schemas**

Create `pyproject.toml` with FastAPI, uvicorn, Pydantic, pytest, httpx, and ruff dependencies. Create `.python-version` with `3.10`. Implement the Pydantic models in `app/schemas.py` exactly as consumed by tests.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schemas.py -v`

Expected: all schema tests pass.

---

### Task 2: Mock Tools and Infrastructure Stubs

**Files:**
- Create: `app/tools/*.py`
- Create: `app/recall/*.py`
- Create: `app/memory/*.py`
- Create: `app/compress/*.py`
- Create: `app/eval/*.py`
- Create: `app/prompt/prompts.yml`
- Create: `app/utils/*.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Consumes: `Product`, `ShoppingSummary`.
- Produces: async tool functions `plan_query`, `get_category_insight`, `search_items`, `calculate_shipping`, `compare_prices`, `pick_items`, `build_summary`.

- [ ] **Step 1: Write failing tool tests**

```python
import pytest

from app.tools.category_insight import get_category_insight
from app.tools.item_picker import pick_items
from app.tools.item_search import search_items
from app.tools.planner import plan_query
from app.tools.price_compare import compare_prices
from app.tools.shipping_calc import calculate_shipping
from app.tools.shopping_summary import build_summary


@pytest.mark.asyncio
async def test_mock_tool_chain_returns_ranked_summary():
    intent = await plan_query("我想买旅行三件套，预算300，不要塑料")
    assert intent["budget"] == 300
    assert "不要塑料" in intent["preferences"]

    insight = await get_category_insight(intent)
    assert "旅行" in insight["category"]

    products = await search_items("Amazon", intent, insight)
    assert products[0].platform == "Amazon"

    shipped = await calculate_shipping(products)
    assert all(product.shipping >= 0 for product in shipped)

    compared = await compare_prices(shipped)
    assert compared == sorted(compared, key=lambda product: product.total_price)

    picked = await pick_items(compared, intent)
    assert len(picked) <= 3

    summary = await build_summary("原始需求", picked)
    assert summary.products == picked
    assert "原始需求" in summary.message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools.py -v`

Expected: fails because tool modules do not exist.

- [ ] **Step 3: Implement minimal tool behavior and importable stubs**

Implement deterministic async mock tools. Add importable stub modules for recall, memory, compress, eval, prompt, and utils with simple functions/classes that do not use external services.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tools.py -v`

Expected: tool chain test passes.

---

### Task 3: Monitor, Dispatch, and Mock AgentLoop

**Files:**
- Create: `app/api/context.py`
- Create: `app/api/monitor.py`
- Create: `app/agent/dispatch_tool.py`
- Create: `app/agent/main_agent.py`
- Create: `tests/test_agent_loop.py`

**Interfaces:**
- Consumes: tool functions from Task 2.
- Produces: `MockAgentLoop.run(query: str) -> ShoppingSummary`.
- Produces: `EventCollector.emit(event_type, message, tool=None, payload=None)`.

- [ ] **Step 1: Write failing AgentLoop tests**

```python
import pytest

from app.agent.main_agent import MockAgentLoop
from app.api.monitor import EventCollector


@pytest.mark.asyncio
async def test_agent_loop_emits_events_and_returns_summary(tmp_path):
    collector = EventCollector(thread_id="thread_test")
    loop = MockAgentLoop(thread_id="thread_test", session_dir=tmp_path, monitor=collector)

    summary = await loop.run("我想买一套旅行三件套，预算300，不要塑料")

    event_types = [event.type for event in collector.events]
    assert "task_started" in event_types
    assert "tool_start" in event_types
    assert "subagent_started" in event_types
    assert "subagent_finished" in event_types
    assert "task_result" in event_types
    assert summary.products
    assert (tmp_path / "summary.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_agent_loop.py -v`

Expected: fails because AgentLoop and monitor modules do not exist.

- [ ] **Step 3: Implement monitor, dispatch, and AgentLoop**

Implement `EventCollector`, context helpers, mock platform dispatch with `asyncio.gather`, and `MockAgentLoop.run`. Write final summary JSON to the task session directory.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_agent_loop.py -v`

Expected: AgentLoop test passes.

---

### Task 4: FastAPI Task API and WebSocket Routing

**Files:**
- Create: `app/api/connection.py`
- Create: `app/api/server.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `MockAgentLoop`.
- Produces: FastAPI app object `app`.
- Produces: routes `POST /api/tasks`, `GET /api/tasks/{thread_id}`, `WebSocket /ws/{thread_id}`.

- [ ] **Step 1: Write failing API tests**

```python
from fastapi.testclient import TestClient

from app.api.server import app


def test_create_task_returns_thread_id():
    client = TestClient(app)
    response = client.post("/api/tasks", json={"query": "旅行三件套，预算300"})
    assert response.status_code == 200
    data = response.json()
    assert data["thread_id"].startswith("thread_")
    assert data["status"] == "running"


def test_get_unknown_task_returns_404():
    client = TestClient(app)
    response = client.get("/api/tasks/thread_missing")
    assert response.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_api.py -v`

Expected: fails because `app.api.server` does not exist.

- [ ] **Step 3: Implement FastAPI server and task registry**

Implement in-memory `TASKS`, task creation, background execution, task lookup, CORS for the Vite dev server, and WebSocket connection handling. Use the monitor to broadcast events.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_api.py -v`

Expected: API tests pass.

---

### Task 5: React/Vite Frontend Console

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/App.css`

**Interfaces:**
- Consumes: `POST /api/tasks`, `GET /api/tasks/{thread_id}`, `WebSocket /ws/{thread_id}`.
- Produces: single-page teaching console with query input, status, results, and event stream.

- [ ] **Step 1: Create frontend app files**

Implement a Vite React app that submits a query, opens WebSocket for returned `thread_id`, appends incoming events, and renders `task_result.payload.summary`.

- [ ] **Step 2: Run frontend build**

Run: `cd frontend && npm install && npm run build`

Expected: Vite build exits 0.

---

### Task 6: Runtime Directories, README, and Full Verification

**Files:**
- Create: `README.md`
- Create: `output/.gitkeep`
- Create: `uploaded/.gitkeep`
- Create: `docker/.gitkeep`
- Create: `examples/run_mock_agent.py`

**Interfaces:**
- Consumes: backend and frontend startup commands.
- Produces: documented local run workflow.

- [ ] **Step 1: Add docs and runtime directories**

Document backend install/start, frontend install/start, tests, and the mock nature of the MVP. Add `.gitkeep` files for empty directories and an example script that runs the mock AgentLoop from CLI.

- [ ] **Step 2: Run backend tests**

Run: `uv run pytest -v`

Expected: all backend tests pass.

- [ ] **Step 3: Run frontend build**

Run: `cd frontend && npm run build`

Expected: build exits 0.

- [ ] **Step 4: Check git diff**

Run: `git status --short`

Expected: only intended MVP files are changed or untracked.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add .
git commit -m "Build runnable OmniMatch MVP"
```

Expected: commit succeeds.
