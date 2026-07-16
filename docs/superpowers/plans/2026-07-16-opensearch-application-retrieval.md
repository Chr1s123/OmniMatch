# OpenSearch Application Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement OpenSearch-backed long-term memory and `CategoryInsight` Hybrid RAG while keeping product ANN exclusively in Faiss.

**Architecture:** A small async HTTP transport owns OpenSearch requests, index aliases, health checks, and redaction. Memory and knowledge providers share the versioned Query-tower embedding provider from the Faiss phase; knowledge retrieval uses a top-level OpenSearch `hybrid` query plus a `normalization-processor` search pipeline with vector/text weights `0.6/0.4`, while hard constraints use complete filtered reads that never silently degrade.

**Tech Stack:** Python 3.10, httpx, Pydantic 2, OpenSearch 2.19-compatible REST APIs, Docker Compose, pytest, pytest-asyncio.

## Global Constraints

- OpenSearch serves only long-term memory and `CategoryInsight` knowledge RAG.
- OpenSearch must not replace, parallel, filter, or rerank Faiss product ANN results.
- Query-side vectors use the same versioned Query-tower service as Faiss requests.
- OpenSearch memory and knowledge document vectors are encoded by that same service and store `model_bundle_version`.
- Hybrid score weights default to `vector_weight=0.6` and `text_weight=0.4`; both are in `[0, 1]` and sum to `1`.
- Hybrid search uses a top-level `hybrid` query and a `normalization-processor` search pipeline.
- Blacklist and hard-constraint read failure must produce `clarify` or `fail`, never silent continuation.
- Preference read failure may continue without personalization only with an explicit warning.
- Category RAG failure may fall back to the existing WebSearch Provider with source disclosure.
- `test` and `submission` use deterministic network-free providers.
- Faiss, Milvus, model training, and local GPU inference are outside this plan.

---

## Current State

- `PreferenceStore` is an unscoped process-local list of strings and is not connected to AgentLoop.
- `inject_preferences()` only concatenates strings into a prompt.
- `CategoryInsight` calls `WebSearchProvider` and returns fixed attributes.
- `ProviderRegistry` has no memory or knowledge provider objects.
- `OmniMatchSettings.memory_provider` is recorded but does not select an implementation.
- No OpenSearch transport, index mapping, alias, pipeline, health check, or integration test exists.
- `memory_read` and `memory_write` are not Agent actions or Tool Registry actions.

## Implementation References

- [OpenSearch Hybrid Query](https://docs.opensearch.org/latest/query-dsl/compound/hybrid/)
- [OpenSearch normalization processor](https://docs.opensearch.org/2.16/search-plugins/search-pipelines/normalization-processor/)

## File Structure

- Create: `app/opensearch/transport.py`
  - Owns authenticated async REST requests, health checks, aliases, and redacted errors.
- Create: `app/opensearch/schema.py`
  - Builds memory/knowledge mappings and the normalization search pipeline.
- Create: `app/opensearch/query.py`
  - Builds top-level Hybrid Query DSL and exact filtered reads.
- Create: `app/memory/models.py`
  - Owns `MemoryRecord` and `MemorySnapshot`.
- Replace: `app/memory/store.py`
  - Adds deterministic and OpenSearch memory providers.
- Modify: `app/memory/injector.py`
  - Injects typed preferences and hard constraints.
- Create: `app/knowledge/models.py`
- Create: `app/knowledge/opensearch.py`
  - Implements Category knowledge indexing and Hybrid retrieval.
- Modify: `app/providers/base.py`, `app/providers/registry.py`
  - Adds optional memory and knowledge provider contracts/instances.
- Modify: `app/config.py`, `.env.example`
  - Adds OpenSearch endpoint, aliases, pipeline, analyzer, dimension, weights, and timeout.
- Modify: `app/agent/actions.py`, `app/agent/tool_registry.py`, `app/agent/main_agent.py`
  - Adds `memory_read` and `memory_write` tools and planner visibility.
- Modify: `app/schemas.py`, `app/api/server.py`
  - Accepts an optional opaque user ID and passes it to root and child Tool contexts without exposing it in task state.
- Modify: `app/tools/context.py`, `app/tools/category_insight.py`
  - Carries user identity and uses knowledge retrieval with explicit fallback.
- Create: `examples/bootstrap_opensearch.py`
- Create: `docker/opensearch-compose.yml`
- Create: `tests/test_opensearch.py`
- Create: `tests/test_memory.py`
- Modify: `tests/test_tools.py`, `tests/test_config.py`, `tests/test_providers.py`
- Modify: `README.md`, `pyproject.toml`

### Task 1: Build OpenSearch Transport, Mappings, Pipeline, And Query DSL

**Files:**
- Create: `app/opensearch/__init__.py`
- Create: `app/opensearch/transport.py`
- Create: `app/opensearch/schema.py`
- Create: `app/opensearch/query.py`
- Create: `tests/test_opensearch.py`

**Interfaces:**
- Produces: `OpenSearchTransport.request(method, path, json=None, params=None) -> dict`
- Produces: `memory_index_body(dimension, analyzer) -> dict`
- Produces: `knowledge_index_body(dimension, analyzer) -> dict`
- Produces: `normalization_pipeline_body(vector_weight, text_weight) -> dict`
- Produces: `hybrid_query(text, vector, filters, size) -> dict`

- [ ] **Step 1: Write failing pure-builder and transport tests**

Create `tests/test_opensearch.py`:

```python
import httpx
import pytest

from app.opensearch.query import hybrid_query
from app.opensearch.schema import normalization_pipeline_body
from app.opensearch.transport import OpenSearchTransport


def test_normalization_pipeline_uses_approved_weights():
    body = normalization_pipeline_body(0.6, 0.4)
    processor = body["phase_results_processors"][0]["normalization-processor"]
    assert processor["normalization"]["technique"] == "min_max"
    assert processor["combination"]["technique"] == "arithmetic_mean"
    assert processor["combination"]["parameters"]["weights"] == [0.6, 0.4]


def test_hybrid_query_is_top_level_and_filters_all_subqueries():
    body = hybrid_query(
        text="旅行收纳",
        vector=[1.0, 0.0],
        filters={"language": "zh", "market": "CN"},
        size=10,
        range_filters={"published_at": {"gte": "now-365d"}},
    )
    hybrid = body["query"]["hybrid"]
    assert len(hybrid["queries"]) == 2
    assert hybrid["filter"] == {
        "bool": {
            "filter": [
                {"term": {"language": "zh"}},
                {"term": {"market": "CN"}},
                {"range": {"published_at": {"gte": "now-365d"}}},
            ]
        }
    }


@pytest.mark.asyncio
async def test_transport_redacts_authorization_from_errors():
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="failure Authorization: Basic secret")

    transport = OpenSearchTransport(
        base_url="https://search.example",
        username="admin",
        password="secret",
        timeout_seconds=5,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await transport.request("GET", "/_cluster/health")

    assert "secret" not in str(exc_info.value)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_opensearch.py -q
```

Expected: FAIL because `app.opensearch` does not exist.

- [ ] **Step 3: Implement the transport**

Create `app/opensearch/transport.py`:

```python
from __future__ import annotations

import re
from typing import Any

import httpx


class OpenSearchTransport:
    def __init__(self, base_url: str, username: str, password: str, timeout_seconds: float, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout_seconds = timeout_seconds
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def request(self, method: str, path: str, json: dict[str, Any] | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self.client.request(
            method,
            f"{self.base_url}/{path.lstrip('/')}",
            json=json,
            params=params,
            auth=(self.username, self.password) if self.username else None,
        )
        if response.is_error:
            detail = re.sub(
                r"(?i)authorization\s*:\s*[^\r\n]+",
                "Authorization: [REDACTED]",
                response.text[:500],
            )
            detail = re.sub(
                r"(?i)(password|api[_-]?key)(\s*[:=]\s*)\S+",
                r"\1\2[REDACTED]",
                detail,
            )
            raise RuntimeError(f"OpenSearch {method} {path} failed with {response.status_code}: {detail}")
        if not response.content:
            return {}
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("OpenSearch response must be a JSON object")
        return payload

    async def health(self) -> dict[str, Any]:
        return await self.request("GET", "/_cluster/health")
```

- [ ] **Step 4: Implement mappings, pipeline, and query builders**

Create `app/opensearch/schema.py`:

```python
def _index_body(dimension: int, analyzer: str, properties: dict) -> dict:
    return {
        "settings": {"index": {"knn": True}},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                **properties,
                "text": {"type": "text", "analyzer": analyzer},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "lucene",
                    },
                },
                "model_bundle_version": {"type": "keyword"},
            },
        },
    }


def memory_index_body(dimension: int, analyzer: str) -> dict:
    return _index_body(
        dimension,
        analyzer,
        {
            "user_id": {"type": "keyword"},
            "memory_type": {"type": "keyword"},
            "category": {"type": "keyword"},
            "language": {"type": "keyword"},
            "market": {"type": "keyword"},
            "created_at": {"type": "date"},
        },
    )


def knowledge_index_body(dimension: int, analyzer: str) -> dict:
    return _index_body(
        dimension,
        analyzer,
        {
            "title": {"type": "text", "analyzer": analyzer},
            "category": {"type": "keyword"},
            "language": {"type": "keyword"},
            "market": {"type": "keyword"},
            "source_url": {"type": "keyword", "index": False},
            "published_at": {"type": "date"},
        },
    )


def normalization_pipeline_body(vector_weight: float, text_weight: float) -> dict:
    if abs(vector_weight + text_weight - 1.0) > 1e-6:
        raise ValueError("hybrid weights must sum to 1")
    return {
        "description": "OmniMatch vector and BM25 normalization",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [vector_weight, text_weight]},
                    },
                }
            }
        ],
    }
```

Create `app/opensearch/query.py`:

```python
from typing import Any


def hybrid_query(
    text: str,
    vector: list[float],
    filters: dict[str, str],
    size: int,
    range_filters: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    filter_clauses = [{"term": {key: value}} for key, value in sorted(filters.items())]
    filter_clauses.extend(
        {"range": {key: value}}
        for key, value in sorted((range_filters or {}).items())
    )
    return {
        "size": size,
        "query": {
            "hybrid": {
                "queries": [
                    {"knn": {"embedding": {"vector": vector, "k": max(size, 100)}}},
                    {"multi_match": {"query": text, "fields": ["title^2", "text"]}},
                ],
                "filter": {"bool": {"filter": filter_clauses}},
            }
        },
    }


def exact_memory_query(user_id: str, memory_types: list[str], size: int = 1000) -> dict[str, Any]:
    return {
        "size": size,
        "query": {
            "bool": {
                "filter": [
                    {"term": {"user_id": user_id}},
                    {"terms": {"memory_type": memory_types}},
                ]
            }
        },
    }
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
uv run pytest tests/test_opensearch.py -q
```

Expected: `3 passed`.

```bash
git add app/opensearch tests/test_opensearch.py
git commit -m "feat: add opensearch hybrid query foundation"
```

### Task 2: Implement Typed Long-Term Memory

**Files:**
- Create: `app/memory/models.py`
- Replace: `app/memory/store.py`
- Modify: `app/memory/injector.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Produces: `MemoryRecord`, `MemorySnapshot`
- Produces: `MemoryProvider.read_hard_constraints/read_preferences/write`
- Produces: `OpenSearchMemoryProvider`
- Produces: `DeterministicMemoryProvider`

- [ ] **Step 1: Write failing memory tests**

Create `tests/test_memory.py` with tests for:

```python
import json

import httpx
import pytest

from app.memory.models import MemoryRecord
from app.memory.store import DeterministicMemoryProvider, OpenSearchMemoryProvider
from app.opensearch.transport import OpenSearchTransport
from app.recall.placeholder import DeterministicEmbeddingProvider


@pytest.mark.asyncio
async def test_memory_reads_all_hard_constraints_without_topk_truncation():
    provider = DeterministicMemoryProvider(
        [
            MemoryRecord(id="b1", user_id="u1", memory_type="blacklist", text="no plastic", model_bundle_version="bundle-v1"),
            MemoryRecord(id="p1", user_id="u1", memory_type="preference", text="likes canvas", model_bundle_version="bundle-v1"),
        ]
    )
    result = await provider.read_hard_constraints("u1")
    assert [record.id for record in result.data] == ["b1"]


@pytest.mark.asyncio
async def test_opensearch_memory_scopes_queries_to_user():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"hits": {"hits": []}})

    transport = OpenSearchTransport(
        "https://search.example",
        "",
        "",
        5,
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    embedding = DeterministicEmbeddingProvider(8, "bundle-v1")
    provider = OpenSearchMemoryProvider(
        transport=transport,
        embedding=embedding,
        index_alias="memory-current",
        search_pipeline="hybrid-v1",
        model_bundle_version="bundle-v1",
    )

    await provider.read_hard_constraints("u1")
    await provider.read_preferences("u1", "旅行收纳", top_k=20)

    assert bodies[0]["query"]["bool"]["filter"][0] == {"term": {"user_id": "u1"}}
    hybrid_filters = bodies[1]["query"]["hybrid"]["filter"]["bool"]["filter"]
    assert {"term": {"user_id": "u1"}} in hybrid_filters
    assert {"term": {"model_bundle_version": "bundle-v1"}} in hybrid_filters
```

Use concrete `httpx.MockTransport` responses with empty `hits.hits` for the second test; do not require Docker for unit tests.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_memory.py -q
```

Expected: FAIL because typed memory providers do not exist.

- [ ] **Step 3: Add typed models and protocol**

Create `app/memory/models.py`:

```python
from typing import Literal

from pydantic import BaseModel, Field


MemoryType = Literal["preference", "blacklist", "hard_constraint"]


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    memory_type: MemoryType
    text: str = Field(min_length=1)
    category: str | None = None
    language: str = "zh"
    market: str = "CN"
    created_at: str | None = None
    model_bundle_version: str
    embedding: list[float] | None = None


class MemorySnapshot(BaseModel):
    hard_constraints: list[MemoryRecord] = Field(default_factory=list)
    preferences: list[MemoryRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
```

Define this protocol in `app/memory/store.py`:

```python
class MemoryProvider(Protocol):
    async def read_hard_constraints(self, user_id: str) -> ProviderResult[list[MemoryRecord]]:
        raise NotImplementedError

    async def read_preferences(self, user_id: str, query: str, top_k: int) -> ProviderResult[list[MemoryRecord]]:
        raise NotImplementedError

    async def write(self, record: MemoryRecord) -> ProviderResult[MemoryRecord]:
        raise NotImplementedError
```

- [ ] **Step 4: Implement deterministic and OpenSearch providers**

Implement `DeterministicMemoryProvider` as a dictionary keyed by record ID. `read_hard_constraints()` returns every matching `blacklist` and `hard_constraint`, sorted by ID. `read_preferences()` ranks the user's `preference` records by lowercase token overlap with the query and slices `top_k`. `write()` replaces the matching ID and returns the validated record.

Implement the OpenSearch provider with these methods in `app/memory/store.py`:

```python
class OpenSearchMemoryProvider:
    def __init__(self, transport, embedding, index_alias, search_pipeline, model_bundle_version):
        self.transport = transport
        self.embedding = embedding
        self.index_alias = index_alias
        self.search_pipeline = search_pipeline
        self.model_bundle_version = model_bundle_version
        if embedding.manifest.model_bundle_version != model_bundle_version:
            raise ValueError("memory/embedding model_bundle_version mismatch")

    @staticmethod
    def _parse_hits(payload: dict) -> list[MemoryRecord]:
        records = []
        for hit in payload.get("hits", {}).get("hits", []):
            source = dict(hit["_source"])
            source.pop("embedding", None)
            source.setdefault("id", hit["_id"])
            records.append(MemoryRecord.model_validate(source))
        return records

    async def read_hard_constraints(self, user_id: str):
        started = perf_counter()
        payload = await self.transport.request(
            "POST",
            f"/{self.index_alias}/_search",
            json=exact_memory_query(user_id, ["blacklist", "hard_constraint"]),
        )
        records = self._parse_hits(payload)
        return ProviderResult(
            "opensearch_memory",
            "real",
            int((perf_counter() - started) * 1000),
            records,
            response_summary=f"alias={self.index_alias} hard_hits={len(records)}",
        )

    async def read_preferences(self, user_id: str, query: str, top_k: int):
        started = perf_counter()
        encoded = await self.embedding.encode_query(query)
        body = hybrid_query(
            query,
            encoded.data,
            {
                "user_id": user_id,
                "memory_type": "preference",
                "model_bundle_version": self.model_bundle_version,
            },
            top_k,
        )
        payload = await self.transport.request(
            "POST",
            f"/{self.index_alias}/_search",
            json=body,
            params={"search_pipeline": self.search_pipeline},
        )
        records = self._parse_hits(payload)
        return ProviderResult(
            "opensearch_memory",
            "real",
            int((perf_counter() - started) * 1000),
            records,
            warnings=encoded.warnings,
            response_summary=f"alias={self.index_alias} preference_hits={len(records)}",
        )

    async def write(self, record: MemoryRecord):
        if record.model_bundle_version != self.model_bundle_version:
            raise ValueError("memory record model_bundle_version mismatch")
        encoded = await self.embedding.encode_query(record.text)
        source = record.model_dump(exclude={"id", "embedding"})
        source["embedding"] = encoded.data
        await self.transport.request(
            "PUT",
            f"/{self.index_alias}/_doc/{record.id}",
            json=source,
            params={"refresh": "wait_for"},
        )
        return ProviderResult(
            "opensearch_memory",
            "real",
            encoded.latency_ms,
            record.model_copy(update={"embedding": None}),
            warnings=encoded.warnings,
            response_summary=f"alias={self.index_alias} write=1",
        )
```

Import `perf_counter`, `ProviderResult`, `hybrid_query`, and `exact_memory_query` explicitly. Never put raw memory text into `response_summary`.

- [ ] **Step 5: Make prompt injection typed**

Replace `inject_preferences()` with:

```python
def inject_memory(prompt: str, snapshot: MemorySnapshot) -> str:
    lines = [prompt]
    if snapshot.hard_constraints:
        lines.append("用户硬约束：" + "；".join(item.text for item in snapshot.hard_constraints))
    if snapshot.preferences:
        lines.append("用户长期偏好：" + "；".join(item.text for item in snapshot.preferences))
    return "\n".join(lines)
```

- [ ] **Step 6: Run memory tests and commit**

Run:

```bash
uv run pytest tests/test_memory.py -q
```

Expected: all memory tests PASS.

```bash
git add app/memory tests/test_memory.py
git commit -m "feat: add opensearch long term memory"
```

### Task 3: Implement Category Knowledge Hybrid RAG

**Files:**
- Create: `app/knowledge/__init__.py`
- Create: `app/knowledge/models.py`
- Create: `app/knowledge/opensearch.py`
- Modify: `app/tools/category_insight.py`
- Modify: `tests/test_tools.py`

**Interfaces:**
- Produces: `KnowledgeDocument`, `KnowledgeHit`
- Produces: `KnowledgeProvider.search(query, filters, top_k)`
- Produces: `KnowledgeProvider.index(document)` for versioned incremental writes.
- Preserves: `get_category_insight(intent, ctx) -> dict`

- [ ] **Step 1: Write failing knowledge and fallback tests**

Add to `tests/test_tools.py`:

```python
from dataclasses import replace

from app.knowledge.models import KnowledgeDocument, KnowledgeHit


class FakeKnowledgeProvider:
    async def search(self, query, filters, top_k):
        document = KnowledgeDocument(
            id="guide-1",
            title="旅行收纳指南",
            text="优先选择耐用帆布材质",
            category="旅行三件套",
            language="zh",
            market="CN",
            source_url="https://example.com/guide-1",
            model_bundle_version="fixture-v1",
        )
        return ProviderResult(
            provider="fake_knowledge",
            provider_mode="fake",
            latency_ms=1,
            data=[KnowledgeHit(document=document, score=0.9)],
        )


class FailingKnowledgeProvider:
    async def search(self, query, filters, top_k):
        raise RuntimeError("knowledge index offline")


class ForbiddenWebSearchProvider:
    async def search(self, query):
        raise AssertionError("WebSearch must not run when knowledge retrieval succeeds")


@pytest.mark.asyncio
async def test_category_insight_prefers_opensearch_knowledge():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = replace(
        base,
        knowledge=FakeKnowledgeProvider(),
        web_search=ForbiddenWebSearchProvider(),
    )
    ctx = ToolContext(settings=settings, providers=providers)
    intent = ShoppingIntent(original_query="旅行三件套", category="旅行三件套")

    result = await get_category_insight(intent, ctx)

    assert result["evidence"][0]["source_url"] == "https://example.com/guide-1"
    assert ctx.observations[-1]["retrieval_mode"] == "opensearch_hybrid"


@pytest.mark.asyncio
async def test_category_insight_discloses_web_fallback():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, knowledge=FailingKnowledgeProvider())
    ctx = ToolContext(settings=settings, providers=providers)
    intent = ShoppingIntent(original_query="旅行三件套", category="旅行三件套")

    result = await get_category_insight(intent, ctx)

    assert result["evidence"]
    assert ctx.observations[-1]["retrieval_mode"] == "web_fallback"
    assert "knowledge index offline" in ctx.observations[-1]["warnings"][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tools.py -k "knowledge or web_fallback" -q
```

Expected: FAIL because knowledge providers are missing.

- [ ] **Step 3: Implement knowledge models and provider**

Create models:

```python
class KnowledgeDocument(BaseModel):
    id: str
    title: str
    text: str
    category: str
    language: str = "zh"
    market: str = "CN"
    source_url: str
    published_at: str | None = None
    model_bundle_version: str
    embedding: list[float] | None = None


class KnowledgeHit(BaseModel):
    document: KnowledgeDocument
    score: float
```

Create `app/knowledge/opensearch.py`:

```python
from __future__ import annotations

from time import perf_counter
from typing import Protocol

from app.knowledge.models import KnowledgeDocument, KnowledgeHit
from app.opensearch.query import hybrid_query
from app.opensearch.transport import OpenSearchTransport
from app.providers.base import ProviderResult
from app.recall.providers import EmbeddingProvider


class KnowledgeProvider(Protocol):
    async def search(
        self,
        query: str,
        filters: dict[str, str],
        top_k: int,
    ) -> ProviderResult[list[KnowledgeHit]]:
        raise NotImplementedError
    async def index(
        self,
        document: KnowledgeDocument,
    ) -> ProviderResult[KnowledgeDocument]:
        raise NotImplementedError


class OpenSearchKnowledgeProvider:
    def __init__(self, transport: OpenSearchTransport, embedding: EmbeddingProvider, index_alias: str, search_pipeline: str, model_bundle_version: str) -> None:
        self.transport = transport
        self.embedding = embedding
        self.index_alias = index_alias
        self.search_pipeline = search_pipeline
        self.model_bundle_version = model_bundle_version
        if embedding.manifest.model_bundle_version != model_bundle_version:
            raise ValueError("knowledge/embedding model_bundle_version mismatch")

    async def search(self, query: str, filters: dict[str, str], top_k: int):
        started = perf_counter()
        encoded = await self.embedding.encode_query(query)
        body = hybrid_query(
            query,
            encoded.data,
            {**filters, "model_bundle_version": self.model_bundle_version},
            top_k,
        )
        payload = await self.transport.request(
            "POST",
            f"/{self.index_alias}/_search",
            json=body,
            params={"search_pipeline": self.search_pipeline},
        )
        hits: list[KnowledgeHit] = []
        for row in payload.get("hits", {}).get("hits", []):
            source = dict(row["_source"])
            source.pop("embedding", None)
            source.setdefault("id", row["_id"])
            hits.append(
                KnowledgeHit(
                    document=KnowledgeDocument.model_validate(source),
                    score=float(row.get("_score", 0.0)),
                )
            )
        return ProviderResult(
            provider="opensearch_knowledge",
            provider_mode="real",
            latency_ms=int((perf_counter() - started) * 1000),
            data=hits,
            warnings=encoded.warnings,
            response_summary=f"alias={self.index_alias} hits={len(hits)}",
        )

    async def index(self, document: KnowledgeDocument):
        if document.model_bundle_version != self.model_bundle_version:
            raise ValueError("knowledge document model_bundle_version mismatch")
        encoded = await self.embedding.encode_query(document.text)
        source = document.model_dump(exclude={"id", "embedding"})
        source["embedding"] = encoded.data
        await self.transport.request(
            "PUT",
            f"/{self.index_alias}/_doc/{document.id}",
            json=source,
            params={"refresh": "wait_for"},
        )
        return ProviderResult(
            provider="opensearch_knowledge",
            provider_mode="real",
            latency_ms=encoded.latency_ms,
            data=document.model_copy(update={"embedding": None}),
            warnings=encoded.warnings,
            response_summary=f"alias={self.index_alias} write=1",
        )
```

- [ ] **Step 4: Replace fixed CategoryInsight retrieval with provider-first logic**

Implement this control flow in `get_category_insight()`:

```python
    query = f"{intent.category} buying guide"
    if ctx.providers.knowledge is not None:
        try:
            result = await ctx.providers.knowledge.search(
                query,
                filters={"category": intent.category, "language": "zh", "market": "CN"},
                top_k=10,
            )
            evidence = [hit.document.model_dump(exclude={"embedding"}) for hit in result.data]
            retrieval_mode = "opensearch_hybrid"
            warnings = result.warnings
        except Exception as exc:
            result = await ctx.providers.web_search.search(query)
            evidence = result.data
            retrieval_mode = "web_fallback"
            warnings = [f"Category RAG failed; used WebSearch: {exc}", *result.warnings]
    else:
        result = await ctx.providers.web_search.search(query)
        evidence = result.data
        retrieval_mode = "web_fallback"
        warnings = ["Category RAG not configured", *result.warnings]
```

Keep the returned category/avoid attributes and platform fields, replace fixed popular attributes only when knowledge hits provide structured attributes, and append an observation containing provider, mode, latency, warnings, retrieval mode, and hit count.

- [ ] **Step 5: Run tool tests and commit**

Run:

```bash
uv run pytest tests/test_tools.py -q
```

Expected: all tool tests PASS.

```bash
git add app/knowledge app/tools/category_insight.py tests/test_tools.py
git commit -m "feat: add category knowledge hybrid rag"
```

### Task 4: Configure And Register Application Retrieval

**Files:**
- Modify: `app/config.py`
- Modify: `app/providers/registry.py`
- Modify: `.env.example`
- Modify: `tests/test_config.py`
- Modify: `tests/test_providers.py`

**Interfaces:**
- Produces: OpenSearch endpoint, aliases, pipeline, analyzer, weights, and timeout settings.
- Produces: `ProviderRegistry.memory` and `ProviderRegistry.knowledge` optional fields.

- [ ] **Step 1: Write failing settings tests**

Add to `tests/test_config.py`:

```python
from dataclasses import replace


def test_opensearch_hybrid_weight_defaults():
    settings = submission_settings()
    assert settings.opensearch_vector_weight == 0.6
    assert settings.opensearch_text_weight == 0.4


def test_opensearch_hybrid_weights_must_sum_to_one():
    settings = replace(
        submission_settings(),
        opensearch_vector_weight=0.7,
        opensearch_text_weight=0.4,
    )
    with pytest.raises(ConfigError, match="weights must sum to 1"):
        settings.validate()


def test_opensearch_provider_requires_url():
    settings = replace(
        submission_settings(),
        memory_provider="opensearch",
        knowledge_provider="opensearch",
        opensearch_url=None,
    )
    with pytest.raises(ConfigError, match="OMNIMATCH_OPENSEARCH_URL"):
        settings.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_config.py -k opensearch -q
```

Expected: FAIL because OpenSearch settings do not exist.

- [ ] **Step 3: Add settings and exact validation**

Add:

```python
    knowledge_provider: str = "placeholder"
    opensearch_url: str | None = None
    opensearch_memory_alias: str = "omnimatch-memory-current"
    opensearch_knowledge_alias: str = "omnimatch-knowledge-current"
    opensearch_search_pipeline: str = "omnimatch-hybrid-v1"
    opensearch_text_analyzer: str = "standard"
    opensearch_vector_weight: float = 0.6
    opensearch_text_weight: float = 0.4
    opensearch_timeout_seconds: float = 10.0
    opensearch_security_disabled: bool = False
    memory_top_k: int = 20
    knowledge_top_k: int = 10
```

Read matching environment variables, including `OMNIMATCH_KNOWLEDGE_PROVIDER`, and reuse `_env_bool()` from the Faiss plan for `OMNIMATCH_OPENSEARCH_SECURITY_DISABLED`. When memory or knowledge provider is `opensearch`, require `OMNIMATCH_OPENSEARCH_URL`, a configured `ProviderRegistry.embedding`, and a positive embedding dimension. Username/password remain secret environment variables read only by the transport. Allow both to be empty only when `opensearch_security_disabled=True` and the URL host is `127.0.0.1`, `localhost`, or `::1`; otherwise require both. Validate weights and Top-K values. Add `knowledge` to `provider_modes()`.

- [ ] **Step 4: Extend registry construction**

Add optional final fields after the recall provider:

```python
    memory: MemoryProvider | None = None
    knowledge: KnowledgeProvider | None = None
```

- `submission`/`test` build deterministic providers when selected.
- `opensearch` requires `registry.embedding` from the Faiss/runtime plan, builds one shared `OpenSearchTransport`, reuses that exact Query-tower instance, and constructs both OpenSearch providers. Raise `ConfigError("OpenSearch requires a configured embedding provider")` if it is absent.
- Existing `memory` configuration maps to the deterministic in-memory provider until callers explicitly select `opensearch`.

- [ ] **Step 5: Document environment variables**

Add:

```dotenv
OMNIMATCH_MEMORY_PROVIDER=opensearch
OMNIMATCH_KNOWLEDGE_PROVIDER=opensearch
OMNIMATCH_OPENSEARCH_URL=http://127.0.0.1:9200
OMNIMATCH_OPENSEARCH_USERNAME=admin
OMNIMATCH_OPENSEARCH_PASSWORD=
OMNIMATCH_OPENSEARCH_MEMORY_ALIAS=omnimatch-memory-current
OMNIMATCH_OPENSEARCH_KNOWLEDGE_ALIAS=omnimatch-knowledge-current
OMNIMATCH_OPENSEARCH_SEARCH_PIPELINE=omnimatch-hybrid-v1
OMNIMATCH_OPENSEARCH_TEXT_ANALYZER=standard
OMNIMATCH_OPENSEARCH_VECTOR_WEIGHT=0.6
OMNIMATCH_OPENSEARCH_TEXT_WEIGHT=0.4
OMNIMATCH_OPENSEARCH_TIMEOUT_SECONDS=10
OMNIMATCH_OPENSEARCH_SECURITY_DISABLED=false
```

- [ ] **Step 6: Run configuration and registry tests and commit**

Run:

```bash
uv run pytest tests/test_config.py tests/test_providers.py -q
```

Expected: all tests PASS.

```bash
git add app/config.py app/providers/registry.py .env.example tests/test_config.py tests/test_providers.py
git commit -m "feat: register opensearch application retrieval"
```

### Task 5: Integrate Memory Tools Into AgentLoop

**Files:**
- Modify: `app/agent/actions.py`
- Modify: `app/agent/tool_registry.py`
- Modify: `app/agent/main_agent.py`
- Modify: `app/tools/context.py`
- Modify: `app/schemas.py`
- Modify: `app/api/server.py`
- Modify: `tests/test_agent_loop.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**
- Produces tool actions: `memory_read`, `memory_write`
- Produces: `ToolRegistry.memory_snapshot`
- Produces: hard-constraint fail-closed behavior
- Produces: `ShoppingQuery.user_id` and `CompetitionAgentLoop.user_id`

- [ ] **Step 1: Write failing memory tool tests**

Add this fake and tests to `tests/test_tools.py`:

```python
class FakeMemoryProvider:
    def __init__(self, fail_hard=False, fail_preferences=False):
        self.fail_hard = fail_hard
        self.fail_preferences = fail_preferences
        self.writes = []

    async def read_hard_constraints(self, user_id):
        if self.fail_hard:
            raise RuntimeError("hard memory offline")
        record = MemoryRecord(
            id="hard-1",
            user_id=user_id,
            memory_type="blacklist",
            text="塑料",
            model_bundle_version="fixture-v1",
        )
        return ProviderResult("fake_memory", "fake", 1, [record])

    async def read_preferences(self, user_id, query, top_k):
        if self.fail_preferences:
            raise RuntimeError("preference memory offline")
        record = MemoryRecord(
            id="pref-1",
            user_id=user_id,
            memory_type="preference",
            text="帆布",
            model_bundle_version="fixture-v1",
        )
        return ProviderResult("fake_memory", "fake", 1, [record])

    async def write(self, record):
        self.writes.append(record)
        return ProviderResult("fake_memory", "fake", 1, record)


@pytest.mark.asyncio
async def test_memory_read_merges_constraints_and_preferences():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, memory=FakeMemoryProvider())
    ctx = ToolContext(settings=settings, providers=providers, user_id="u1")
    registry = ToolRegistry(ctx)
    await registry.run("plan", {"query": "旅行三件套"})

    snapshot = await registry.run("memory_read", {})

    assert [record.text for record in snapshot.hard_constraints] == ["塑料"]
    assert registry.intent.negative_constraints == ["塑料"]
    assert ctx.user_profile["preferences"] == ["帆布"]


@pytest.mark.asyncio
async def test_memory_read_fails_closed_when_hard_constraints_unavailable():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, memory=FakeMemoryProvider(fail_hard=True))
    ctx = ToolContext(settings=settings, providers=providers, user_id="u1")
    registry = ToolRegistry(ctx)
    await registry.run("plan", {"query": "旅行三件套"})

    with pytest.raises(RuntimeError, match="hard memory offline"):
        await registry.run("memory_read", {})


@pytest.mark.asyncio
async def test_memory_read_degrades_only_preferences():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, memory=FakeMemoryProvider(fail_preferences=True))
    ctx = ToolContext(settings=settings, providers=providers, user_id="u1")
    registry = ToolRegistry(ctx)
    await registry.run("plan", {"query": "旅行三件套"})

    snapshot = await registry.run("memory_read", {})

    assert snapshot.preferences == []
    assert "preference memory offline" in snapshot.warnings[0]


@pytest.mark.asyncio
async def test_memory_write_forces_context_user_id():
    settings = submission_settings()
    base = ProviderRegistry.from_settings(settings)
    memory = FakeMemoryProvider()
    ctx = ToolContext(
        settings=settings,
        providers=replace(base, memory=memory),
        user_id="u1",
    )
    registry = ToolRegistry(ctx)

    await registry.run(
        "memory_write",
        {
            "id": "pref-new",
            "memory_type": "preference",
            "text": "喜欢轻量款",
            "model_bundle_version": "fixture-v1",
        },
    )

    assert memory.writes[0].user_id == "u1"
```

Add to `tests/test_schemas.py`:

```python
def test_shopping_query_accepts_opaque_user_id():
    query = ShoppingQuery(query="旅行收纳", user_id="user_123")
    assert query.user_id == "user_123"


def test_shopping_query_rejects_user_id_with_path_characters():
    with pytest.raises(ValidationError):
        ShoppingQuery(query="旅行收纳", user_id="../../etc/passwd")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tools.py tests/test_schemas.py -k "memory or user_id" -q
```

Expected: FAIL because memory actions and `ShoppingQuery.user_id` do not exist.

- [ ] **Step 3: Add action names and planner prompt**

Extend `ToolActionName` and `TOOL_ACTIONS` with `memory_read` and `memory_write`. Add both names to the allowed-tool string in `_plan_next_action()`.

Add an opaque optional ID to `ShoppingQuery`; do not add it to `TaskState` or trace payloads:

```python
    user_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )
```

Change `_run_task()` to accept `user_id`, pass `request.user_id` from `create_task()`, and construct the root loop with `user_id=user_id`.

- [ ] **Step 4: Extend ToolContext and ToolRegistry**

Add to `ToolContext`:

```python
    user_id: str | None = None
```

Add `user_id: str | None = None` to the end of `CompetitionAgentLoop.__init__()`, assign `self.user_id`, and construct `ToolContext` with `user_id=self.user_id`. In `_execute_fork()`, pass `user_id=self.user_id` to every homogeneous child loop; children must not receive a different user identity from LLM-controlled `context_snapshot`.

Add `self.memory_snapshot = MemorySnapshot()` in `ToolRegistry.__init__()`. Implement `memory_read` before `category_insight`:

```python
        if action == "memory_read":
            self._require_intent()
            if self.ctx.providers.memory is None or not self.ctx.user_id:
                return self.memory_snapshot
            try:
                hard = await self.ctx.providers.memory.read_hard_constraints(self.ctx.user_id)
            except Exception as exc:
                self.ctx.observations.append(
                    {
                        "tool": "MemoryRead",
                        "warnings": [f"hard constraint memory unavailable: {exc}"],
                        "hard_constraint_read_failed": True,
                    }
                )
                raise
            try:
                preferences = await self.ctx.providers.memory.read_preferences(
                    self.ctx.user_id,
                    self.intent.original_query,
                    self.ctx.settings.memory_top_k,
                )
                preference_records = preferences.data
                warnings = [*hard.warnings, *preferences.warnings]
            except Exception as exc:
                preference_records = []
                warnings = [*hard.warnings, f"preference memory unavailable: {exc}"]
            self.memory_snapshot = MemorySnapshot(
                hard_constraints=hard.data,
                preferences=preference_records,
                warnings=warnings,
            )
            self.intent.negative_constraints = list(dict.fromkeys([
                *self.intent.negative_constraints,
                *[record.text for record in hard.data],
            ]))
            self.ctx.user_profile["preferences"] = [record.text for record in preference_records]
            self.ctx.observations.append(
                {
                    "tool": "MemoryRead",
                    "provider": hard.provider,
                    "provider_mode": hard.provider_mode,
                    "latency_ms": hard.latency_ms,
                    "warnings": warnings,
                    "hard_constraint_count": len(hard.data),
                    "preference_count": len(preference_records),
                }
            )
            return self.memory_snapshot
```

Do not continue after `read_hard_constraints()` failures. Add `memory_write` with `MemoryRecord.model_validate({**arguments, "user_id": self.ctx.user_id, "model_bundle_version": self.ctx.settings.embedding_model_bundle_version})`; both identity and model version are forced from trusted context, never accepted from the LLM action.

- [ ] **Step 5: Add memory state to snapshots and decision messages**

Add `memory_hard_constraint_count` and `memory_preference_count` to `ToolRegistry.snapshot()`. The planner then sees whether memory was read before recall without receiving raw full history.

- [ ] **Step 6: Run AgentLoop and tool tests and commit**

Run:

```bash
uv run pytest tests/test_tools.py tests/test_agent_loop.py -q
```

Expected: all tests PASS, including fail-closed hard constraints.

```bash
git add app/agent app/tools/context.py app/schemas.py app/api/server.py tests/test_agent_loop.py tests/test_tools.py tests/test_schemas.py tests/test_api.py
git commit -m "feat: add agent memory tools"
```

### Task 6: Bootstrap Docker OpenSearch And Verify The Phase

**Files:**
- Create: `docker/opensearch-compose.yml`
- Create: `examples/bootstrap_opensearch.py`
- Modify: `tests/test_opensearch.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: local OpenSearch at `http://127.0.0.1:9200`
- Produces: bootstrap CLI creating versioned indices, aliases, and search pipeline

- [ ] **Step 1: Add an integration marker**

Add to `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: requires a local external service",
]
```

- [ ] **Step 2: Create a single-node development stack**

Create `docker/opensearch-compose.yml`:

```yaml
# Local development only. Production deployments must enable TLS and authentication.
services:
  opensearch:
    image: opensearchproject/opensearch:2.19.2
    environment:
      discovery.type: single-node
      plugins.security.disabled: "true"
      OPENSEARCH_JAVA_OPTS: -Xms1g -Xmx1g
    ports:
      - "127.0.0.1:9200:9200"
    volumes:
      - omnimatch-opensearch-data:/usr/share/opensearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -fsS http://127.0.0.1:9200/_cluster/health >/dev/null"]
      interval: 10s
      timeout: 5s
      retries: 20

volumes:
  omnimatch-opensearch-data:
```

- [ ] **Step 3: Implement idempotent bootstrap**

`examples/bootstrap_opensearch.py` must:

1. Load `OmniMatchSettings` and call `transport.health()`.
2. Create physical indices `omnimatch-memory-{model_bundle_version}` and `omnimatch-knowledge-{model_bundle_version}` when absent.
3. Create or replace `omnimatch-hybrid-v1` with `normalization_pipeline_body(0.6, 0.4)`.
4. Atomically assign the configured aliases with `POST /_aliases` remove/add actions.
5. Refuse to attach an alias to an index whose mapping dimension or stored model version differs from configuration.
6. Print only index names, aliases, pipeline name, model version, dimension, and cluster status.

- [ ] **Step 4: Add an opt-in integration test**

Mark the test with `@pytest.mark.integration` and skip unless `OMNIMATCH_OPENSEARCH_TEST_URL` is set. It must bootstrap two small knowledge documents and three memory records, then verify:

- hard reads return every blacklist/hard constraint for the user;
- another user's memory never appears;
- Hybrid Query returns a linked knowledge document;
- the search pipeline uses weights `[0.6, 0.4]`;
- OpenSearch is not invoked by the Faiss `ItemSearch` integration test.

- [ ] **Step 5: Document and run verification**

Document these commands in `README.md`:

```bash
docker compose -f docker/opensearch-compose.yml up -d
OMNIMATCH_OPENSEARCH_SECURITY_DISABLED=true OMNIMATCH_OPENSEARCH_URL=http://127.0.0.1:9200 uv run python examples/bootstrap_opensearch.py
OMNIMATCH_OPENSEARCH_TEST_URL=http://127.0.0.1:9200 uv run pytest -m integration -q
```

Run unit verification:

```bash
uv run pytest tests/test_opensearch.py tests/test_memory.py tests/test_tools.py tests/test_providers.py tests/test_config.py -q
```

Expected: all unit tests PASS without Docker.

Run full regression verification:

```bash
uv run pytest -q
```

Expected: all non-integration tests PASS.

- [ ] **Step 6: Commit**

```bash
git add docker/opensearch-compose.yml examples/bootstrap_opensearch.py pyproject.toml README.md tests/test_opensearch.py
git commit -m "test: add opensearch bootstrap and integration coverage"
```

## Phase Acceptance Checklist

- [ ] Memory and knowledge indices store `model_bundle_version` and reject incompatible vectors.
- [ ] Hard constraints use complete user-scoped reads and fail closed.
- [ ] Preferences use semantic Top-K and may degrade only with a warning.
- [ ] Category knowledge uses top-level Hybrid Query plus normalization pipeline.
- [ ] Hybrid vector/text weights are exactly `0.6/0.4` by default and configurable.
- [ ] Category RAG fallback discloses WebSearch as the source.
- [ ] AgentLoop exposes `memory_read` and `memory_write` through normal tool actions.
- [ ] OpenSearch never enters the product Faiss ANN path.
- [ ] Unit tests do not require Docker; integration tests are explicitly opt-in.
- [ ] Local bootstrap is idempotent and alias switching validates vector compatibility.
