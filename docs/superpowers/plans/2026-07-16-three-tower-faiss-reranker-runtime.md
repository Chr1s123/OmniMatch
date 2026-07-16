# Three-Tower Faiss Recall And Reranker Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-dimensional recall stubs with a versioned three-tower request pipeline that fuses semantic and personalization vectors, retrieves Faiss Top-100 candidates, reranks Top-10, and feeds the existing ItemPicker path.

**Architecture:** Introduce narrow embedding, ANN, and reranker provider contracts plus a `ThreeTowerRecallService`. Query/User embeddings and the personalization projection produce a normalized request vector; a version-checked Faiss HNSW inner-product index returns catalog-backed candidates, and a reranker provider produces Top-10. Existing Product Provider search remains an explicit, observable fallback only.

**Tech Stack:** Python 3.10, NumPy, faiss-cpu, httpx, Pydantic 2, pytest, pytest-asyncio.

## Global Constraints

- Faiss alone handles product ANN; OpenSearch must not enter this request path.
- The request formula is `normalize(alpha * q_sem + beta * q_personal)`.
- Defaults are `alpha=0.7`, `beta=0.3`; both are in `[0, 1]` and sum to `1`.
- Without valid user history, use `alpha=1`, `beta=0`.
- Query, personalization, and Item vectors have one manifest-declared dimension and use L2 normalization.
- Faiss uses HNSW with `METRIC_INNER_PRODUCT`; normalized IP is treated as cosine similarity.
- Product recall returns Top-100; the reranker returns Top-10.
- Model/index incompatibility fails closed.
- Reranker failure may fall back to Faiss score order with an explicit warning.
- Product Provider fallback is configuration-controlled and emits `recall_mode="provider_fallback"`.
- `test` and `submission` remain deterministic and network-free.
- Milvus, OpenSearch, model training, and GPU-local reranker inference are outside this plan.

---

## Current State

- `app/recall/tower_user.py`, `tower_query.py`, and `tower_item.py` return one-dimensional input-length values.
- `app/recall/ann.py` returns invented IDs and reciprocal scores; no vector index is loaded.
- `ItemSearch` calls `ProductSearchProvider.search()` directly.
- `ProviderRegistry` has no embedding, ANN, recall, or reranker fields.
- `pyproject.toml` has no NumPy or Faiss dependency.
- No model or index manifest is validated at runtime.
- No reranker interface or Top-100 to Top-10 path exists.

## Implementation References

- [Faiss metric types and cosine-through-normalized-IP](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
- [BAAI BGE Reranker v2 M3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## File Structure

- Create: `app/recall/models.py`
  - Owns model/index manifests, ANN hits, reranked hits, and recall result metadata.
- Create: `app/recall/fusion.py`
  - Owns L2 normalization, weight validation, cold-start behavior, and request fusion.
- Create: `app/recall/providers.py`
  - Defines `EmbeddingProvider`, `ANNProvider`, `RerankerProvider`, and `RecallProvider` protocols.
- Create: `app/recall/placeholder.py`
  - Implements deterministic, network-free encoders, ANN fixtures, and reranking.
- Create: `app/recall/faiss_index.py`
  - Builds, saves, loads, validates, and searches a catalog-backed Faiss index.
- Create: `app/recall/http_embedding.py`
  - Implements the versioned User/Query/personalization HTTP contract.
- Create: `app/recall/http_reranker.py`
  - Implements the remote cross-encoder reranker contract.
- Create: `app/recall/service.py`
  - Orchestrates embedding, fusion, ANN Top-100, reranking Top-10, and warnings.
- Create: `examples/build_faiss_index.py`
  - Builds an index from JSONL catalog records and Item-tower embeddings.
- Create: `examples/benchmark_faiss.py`
  - Reports P50/P95/P99 separately from pytest.
- Modify: `app/providers/base.py`
  - Exposes recall protocols through provider imports without embedding implementations in tools.
- Modify: `app/providers/registry.py`
  - Builds placeholder or real recall services and exposes the shared embedding instance.
- Modify: `app/config.py`
  - Adds recall, embedding, Faiss, reranker, weight, Top-K, and fallback settings.
- Modify: `app/tools/context.py`
  - Carries a minimal user profile for User-tower encoding.
- Modify: `app/tools/item_search.py`
- Modify: `app/tools/shopping_summary.py`
  - Uses recall first and Product Provider only when explicitly configured as fallback.
- Modify: `app/schemas.py`
  - Adds recall and rerank scores to candidates without changing required API fields.
- Modify: `pyproject.toml`, `uv.lock`, `.env.example`, `README.md`
  - Adds dependencies and configuration documentation.
- Replace: `app/recall/ann.py`, `tower_user.py`, `tower_query.py`, `tower_item.py`
  - Remove misleading stubs after imports move to provider-backed modules.
- Create: `tests/fixtures/recall_catalog.jsonl`
- Create: `tests/test_recall.py`
- Modify: `tests/test_config.py`, `tests/test_providers.py`, `tests/test_tools.py`, `tests/test_agent_loop.py`

### Task 1: Add Vector Manifests And Fusion

**Files:**
- Create: `app/recall/models.py`
- Create: `app/recall/fusion.py`
- Create: `tests/test_recall.py`

**Interfaces:**
- Produces: `ModelManifest`
- Produces: `IndexManifest.assert_compatible(model: ModelManifest) -> None`
- Produces: `normalize_vector(vector: Sequence[float]) -> list[float]`
- Produces: `fuse_request_vectors(semantic, personalization, alpha, beta, has_history) -> list[float]`

- [ ] **Step 1: Write failing fusion and compatibility tests**

Create `tests/test_recall.py`:

```python
import math

import pytest

from app.recall.fusion import fuse_request_vectors, normalize_vector
from app.recall.models import IndexManifest, ModelManifest


def test_normalize_vector_has_unit_norm():
    vector = normalize_vector([3.0, 4.0])
    assert vector == pytest.approx([0.6, 0.8])
    assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0)


def test_fusion_uses_semantic_only_without_history():
    result = fuse_request_vectors(
        semantic=[1.0, 0.0],
        personalization=[0.0, 1.0],
        alpha=0.7,
        beta=0.3,
        has_history=False,
    )
    assert result == pytest.approx([1.0, 0.0])


def test_fusion_rejects_invalid_weights():
    with pytest.raises(ValueError, match="sum to 1"):
        fuse_request_vectors([1.0, 0.0], [0.0, 1.0], 0.8, 0.3, True)


def test_index_manifest_rejects_model_version_mismatch():
    model = ModelManifest(
        model_bundle_version="bundle-v2",
        embedding_dimension=2,
        normalization="l2",
        distance_metric="inner_product",
    )
    index = IndexManifest(
        model_bundle_version="bundle-v1",
        embedding_dimension=2,
        normalization="l2",
        distance_metric="inner_product",
        item_count=10,
        created_at="2026-07-16T00:00:00Z",
        checksum="abc",
    )

    with pytest.raises(ValueError, match="model_bundle_version"):
        index.assert_compatible(model)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_recall.py -q
```

Expected: FAIL because recall models and fusion do not exist.

- [ ] **Step 3: Implement manifests and hit models**

Create `app/recall/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas import ProductCandidate


Normalization = Literal["l2"]
DistanceMetric = Literal["inner_product"]


class ModelManifest(BaseModel):
    model_bundle_version: str = Field(min_length=1)
    embedding_dimension: int = Field(gt=0)
    normalization: Normalization = "l2"
    distance_metric: DistanceMetric = "inner_product"


class IndexManifest(ModelManifest):
    item_count: int = Field(ge=0)
    created_at: str
    checksum: str = Field(min_length=1)

    def assert_compatible(self, model: ModelManifest) -> None:
        fields = (
            "model_bundle_version",
            "embedding_dimension",
            "normalization",
            "distance_metric",
        )
        for field_name in fields:
            if getattr(self, field_name) != getattr(model, field_name):
                raise ValueError(
                    f"index/model {field_name} mismatch: "
                    f"{getattr(self, field_name)!r} != {getattr(model, field_name)!r}"
                )


class ANNHit(BaseModel):
    candidate: ProductCandidate
    ann_score: float
    rerank_score: float | None = None


class RecallMetadata(BaseModel):
    recall_mode: str
    model_bundle_version: str
    index_version: str
    alpha: float
    beta: float
    has_user_history: bool
    ann_candidate_count: int
    reranked_count: int
```

- [ ] **Step 4: Implement normalization and fusion**

Create `app/recall/fusion.py`:

```python
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_vector(vector: Sequence[float]) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("vector must be a non-empty one-dimensional sequence")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return (array / norm).astype(np.float32).tolist()


def fuse_request_vectors(
    semantic: Sequence[float],
    personalization: Sequence[float],
    alpha: float,
    beta: float,
    has_history: bool,
) -> list[float]:
    if not 0 <= alpha <= 1 or not 0 <= beta <= 1:
        raise ValueError("alpha and beta must be in [0, 1]")
    if abs((alpha + beta) - 1.0) > 1e-6:
        raise ValueError("alpha and beta must sum to 1")
    semantic_array = np.asarray(normalize_vector(semantic), dtype=np.float32)
    if not has_history:
        return semantic_array.tolist()
    personal_array = np.asarray(normalize_vector(personalization), dtype=np.float32)
    if semantic_array.shape != personal_array.shape:
        raise ValueError("semantic and personalization dimensions must match")
    return normalize_vector(alpha * semantic_array + beta * personal_array)
```

- [ ] **Step 5: Add dependencies and run tests**

Run:

```bash
uv add numpy faiss-cpu
```

Expected: `pyproject.toml` and `uv.lock` include NumPy and `faiss-cpu`.

Run:

```bash
uv run pytest tests/test_recall.py -q
```

Expected: `4 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/recall/models.py app/recall/fusion.py tests/test_recall.py
git commit -m "feat: define recall vectors and manifests"
```

### Task 2: Add Provider Contracts And Deterministic Recall

**Files:**
- Create: `app/recall/providers.py`
- Create: `app/recall/placeholder.py`
- Create: `app/recall/service.py`
- Test: `tests/test_recall.py`

**Interfaces:**
- Produces: `EmbeddingProvider.encode_query/encode_user/project_personalization`
- Produces: `ANNProvider.search`
- Produces: `RerankerProvider.rerank`
- Produces: `ThreeTowerRecallService.search(query, user_profile, top_k, rerank_k)`

- [ ] **Step 1: Write a failing deterministic service test**

Append to `tests/test_recall.py`:

```python
from app.recall.placeholder import (
    DeterministicANNProvider,
    DeterministicEmbeddingProvider,
    DeterministicRerankerProvider,
)
from app.recall.service import ThreeTowerRecallService
from app.schemas import ProductCandidate


@pytest.mark.asyncio
async def test_three_tower_service_returns_ann_then_reranked_candidates():
    catalog = [
        ProductCandidate(
            id=f"item-{index}",
            platform="fixture",
            title=title,
            price=100 + index,
            rating=4.0,
            url=f"https://example.com/{index}",
            evidence=["recall fixture"],
        )
        for index, title in enumerate(["canvas travel bag", "plastic toy", "travel organizer"])
    ]
    embedding = DeterministicEmbeddingProvider(dimension=8, model_bundle_version="fixture-v1")
    service = ThreeTowerRecallService(
        embedding=embedding,
        ann=DeterministicANNProvider(catalog, model_manifest=embedding.manifest),
        reranker=DeterministicRerankerProvider(),
        alpha=0.7,
        beta=0.3,
    )

    result = await service.search(
        query="travel canvas organizer",
        user_profile={"history": ["canvas bag"]},
        top_k=100,
        rerank_k=2,
    )

    assert len(result.data) == 2
    assert result.data[0].rerank_score is not None
    assert result.response_summary.startswith("ann=3 reranked=2")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_recall.py::test_three_tower_service_returns_ann_then_reranked_candidates -q
```

Expected: FAIL because the provider contracts and service do not exist.

- [ ] **Step 3: Define provider protocols**

Create `app/recall/providers.py`:

```python
from __future__ import annotations

from typing import Any, Protocol

from app.providers.base import ProviderResult
from app.recall.models import ANNHit, ModelManifest


class EmbeddingProvider(Protocol):
    manifest: ModelManifest

    async def encode_query(self, text: str) -> ProviderResult[list[float]]:
        raise NotImplementedError

    async def encode_user(self, profile: dict[str, Any]) -> ProviderResult[list[float]]:
        raise NotImplementedError

    async def project_personalization(
        self,
        query_vector: list[float],
        user_vector: list[float],
    ) -> ProviderResult[list[float]]:
        raise NotImplementedError

    async def encode_items(self, items: list[dict[str, Any]]) -> ProviderResult[list[list[float]]]:
        raise NotImplementedError


class ANNProvider(Protocol):
    index_version: str

    def assert_compatible(self, model: ModelManifest) -> None:
        raise NotImplementedError

    async def search(self, vector: list[float], top_k: int) -> ProviderResult[list[ANNHit]]:
        raise NotImplementedError


class RerankerProvider(Protocol):
    async def rerank(
        self,
        query: str,
        hits: list[ANNHit],
        top_k: int,
    ) -> ProviderResult[list[ANNHit]]:
        raise NotImplementedError


class RecallProvider(Protocol):
    async def search(
        self,
        query: str,
        user_profile: dict[str, Any],
        top_k: int,
        rerank_k: int,
    ) -> ProviderResult[list[ANNHit]]:
        raise NotImplementedError
```

- [ ] **Step 4: Implement deterministic providers**

Create `app/recall/placeholder.py` with deterministic SHA-256-based vectors, catalog hits, and lexical reranking:

```python
from __future__ import annotations

import hashlib
import json
from time import perf_counter
from typing import Any

from app.providers.base import ProviderResult
from app.recall.fusion import normalize_vector
from app.recall.models import ANNHit, ModelManifest
from app.schemas import ProductCandidate


def _elapsed(start: float) -> int:
    return int((perf_counter() - start) * 1000)


def _hash_vector(payload: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{payload}:{counter}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    return normalize_vector(values[:dimension])


class DeterministicEmbeddingProvider:
    def __init__(self, dimension: int, model_bundle_version: str) -> None:
        self.manifest = ModelManifest(
            model_bundle_version=model_bundle_version,
            embedding_dimension=dimension,
        )

    async def encode_query(self, text: str) -> ProviderResult[list[float]]:
        start = perf_counter()
        return ProviderResult("fixture_embedding", "fake", _elapsed(start), _hash_vector(text, self.manifest.embedding_dimension))

    async def encode_user(self, profile: dict[str, Any]) -> ProviderResult[list[float]]:
        start = perf_counter()
        text = json.dumps(profile, ensure_ascii=False, sort_keys=True)
        return ProviderResult("fixture_embedding", "fake", _elapsed(start), _hash_vector(text, self.manifest.embedding_dimension))

    async def project_personalization(self, query_vector: list[float], user_vector: list[float]) -> ProviderResult[list[float]]:
        start = perf_counter()
        merged = [query + user for query, user in zip(query_vector, user_vector)]
        return ProviderResult("fixture_embedding", "fake", _elapsed(start), normalize_vector(merged))

    async def encode_items(self, items: list[dict[str, Any]]) -> ProviderResult[list[list[float]]]:
        start = perf_counter()
        vectors = [_hash_vector(json.dumps(item, sort_keys=True), self.manifest.embedding_dimension) for item in items]
        return ProviderResult("fixture_embedding", "fake", _elapsed(start), vectors)


class DeterministicANNProvider:
    index_version = "fixture-index-v1"

    def __init__(self, catalog: list[ProductCandidate], model_manifest: ModelManifest) -> None:
        self.catalog = catalog
        self.model_manifest = model_manifest

    def assert_compatible(self, model: ModelManifest) -> None:
        if self.model_manifest != model:
            raise ValueError("fixture ANN model manifest mismatch")

    async def search(self, vector: list[float], top_k: int) -> ProviderResult[list[ANNHit]]:
        start = perf_counter()
        hits = [ANNHit(candidate=item, ann_score=1.0 - index * 0.01) for index, item in enumerate(self.catalog[:top_k])]
        return ProviderResult("fixture_ann", "fake", _elapsed(start), hits)


class DeterministicRerankerProvider:
    async def rerank(self, query: str, hits: list[ANNHit], top_k: int) -> ProviderResult[list[ANNHit]]:
        start = perf_counter()
        terms = set(query.lower().split())
        scored = []
        for hit in hits:
            overlap = len(terms & set(hit.candidate.title.lower().split()))
            scored.append(hit.model_copy(update={"rerank_score": float(overlap)}))
        scored.sort(key=lambda hit: (hit.rerank_score or 0.0, hit.ann_score), reverse=True)
        return ProviderResult("fixture_reranker", "fake", _elapsed(start), scored[:top_k])
```

- [ ] **Step 5: Implement the composite recall service**

Create `app/recall/service.py`:

```python
from __future__ import annotations

from time import perf_counter
from typing import Any

from app.providers.base import ProviderResult
from app.recall.fusion import fuse_request_vectors
from app.recall.providers import ANNProvider, EmbeddingProvider, RerankerProvider


class ThreeTowerRecallService:
    def __init__(self, embedding: EmbeddingProvider, ann: ANNProvider, reranker: RerankerProvider, alpha: float, beta: float) -> None:
        self.embedding = embedding
        self.ann = ann
        self.reranker = reranker
        self.alpha = alpha
        self.beta = beta
        self.ann.assert_compatible(self.embedding.manifest)

    async def search(self, query: str, user_profile: dict[str, Any], top_k: int, rerank_k: int):
        started = perf_counter()
        query_result = await self.embedding.encode_query(query)
        has_history = any(
            bool(user_profile.get(key))
            for key in ("history", "preferences", "category_affinities")
        )
        if has_history:
            user_result = await self.embedding.encode_user(user_profile)
            personal_result = await self.embedding.project_personalization(query_result.data, user_result.data)
            personal_vector = personal_result.data
        else:
            personal_vector = query_result.data
        request_vector = fuse_request_vectors(query_result.data, personal_vector, self.alpha, self.beta, has_history)
        ann_result = await self.ann.search(request_vector, top_k=top_k)
        warnings = [*query_result.warnings, *ann_result.warnings]
        try:
            rerank_result = await self.reranker.rerank(query, ann_result.data, top_k=rerank_k)
            hits = rerank_result.data
            warnings.extend(rerank_result.warnings)
        except Exception as exc:
            hits = ann_result.data[:rerank_k]
            warnings.append(f"reranker unavailable; used ANN order: {exc}")
        latency_ms = int((perf_counter() - started) * 1000)
        return ProviderResult(
            provider="three_tower_recall",
            provider_mode=query_result.provider_mode,
            latency_ms=latency_ms,
            data=hits,
            warnings=warnings,
            response_summary=f"ann={len(ann_result.data)} reranked={len(hits)} history={has_history}",
        )
```

- [ ] **Step 6: Run service tests and commit**

Run:

```bash
uv run pytest tests/test_recall.py -q
```

Expected: all recall tests PASS.

```bash
git add app/recall/providers.py app/recall/placeholder.py app/recall/service.py tests/test_recall.py
git commit -m "feat: add deterministic three tower recall service"
```

### Task 3: Build And Search A Real Faiss HNSW Index

**Files:**
- Create: `app/recall/faiss_index.py`
- Create: `tests/fixtures/recall_catalog.jsonl`
- Modify: `tests/test_recall.py`

**Interfaces:**
- Produces: `build_faiss_index(index_dir, candidates, vectors, manifest, hnsw_m=32) -> IndexManifest`
- Produces: `FaissANNProvider(index_dir)`

- [ ] **Step 1: Add a concrete catalog fixture and failing real-index test**

Create `tests/fixtures/recall_catalog.jsonl`:

```jsonl
{"id":"travel-canvas","platform":"fixture","title":"Canvas travel organizer","price":198,"currency":"CNY","shipping":0,"tax":0,"rating":4.7,"url":"https://example.com/travel-canvas","material":"canvas","evidence":["recall fixture"],"warnings":[]}
{"id":"travel-nylon","platform":"fixture","title":"Nylon packing cube set","price":168,"currency":"CNY","shipping":0,"tax":0,"rating":4.5,"url":"https://example.com/travel-nylon","material":"nylon","evidence":["recall fixture"],"warnings":[]}
{"id":"toy-plastic","platform":"fixture","title":"Plastic toy box","price":99,"currency":"CNY","shipping":0,"tax":0,"rating":4.0,"url":"https://example.com/toy-plastic","material":"plastic","evidence":["recall fixture"],"warnings":[]}
```

Add to `tests/test_recall.py`:

```python
from app.recall.faiss_index import FaissANNProvider, build_faiss_index


@pytest.mark.asyncio
async def test_faiss_hnsw_returns_catalog_candidates(tmp_path):
    model = ModelManifest(model_bundle_version="fixture-v1", embedding_dimension=3)
    candidates = [
        ProductCandidate(id="x", platform="fixture", title="x", price=1, url="https://x"),
        ProductCandidate(id="y", platform="fixture", title="y", price=2, url="https://y"),
    ]
    build_faiss_index(
        tmp_path,
        candidates,
        vectors=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        model_manifest=model,
        hnsw_m=8,
    )
    provider = FaissANNProvider(tmp_path)

    result = await provider.search([1.0, 0.0, 0.0], top_k=2)

    assert result.data[0].candidate.id == "x"
    assert result.data[0].ann_score > result.data[1].ann_score
    provider.assert_compatible(model)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run pytest tests/test_recall.py::test_faiss_hnsw_returns_catalog_candidates -q
```

Expected: FAIL because `faiss_index.py` does not exist.

- [ ] **Step 3: Implement index build, persistence, checksum, and search**

Create `app/recall/faiss_index.py`. The implementation must:

```python
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import faiss
import numpy as np

from app.providers.base import ProviderResult
from app.recall.models import ANNHit, IndexManifest, ModelManifest
from app.schemas import ProductCandidate


def build_faiss_index(index_dir: Path, candidates: list[ProductCandidate], vectors: list[list[float]], model_manifest: ModelManifest, hnsw_m: int = 32) -> IndexManifest:
    if len(candidates) != len(vectors):
        raise ValueError("candidate/vector count mismatch")
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.shape != (len(candidates), model_manifest.embedding_dimension):
        raise ValueError("item vector shape does not match model manifest")
    faiss.normalize_L2(matrix)
    index = faiss.IndexHNSWFlat(model_manifest.embedding_dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.add(matrix)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    catalog_path = index_dir / "catalog.json"
    catalog_path.write_text(
        json.dumps([candidate.model_dump() for candidate in candidates], ensure_ascii=False),
        encoding="utf-8",
    )
    checksum = hashlib.sha256(index_path.read_bytes() + catalog_path.read_bytes()).hexdigest()
    manifest = IndexManifest(
        **model_manifest.model_dump(),
        item_count=len(candidates),
        created_at=datetime.now(timezone.utc).isoformat(),
        checksum=checksum,
    )
    (index_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


class FaissANNProvider:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.manifest = IndexManifest.model_validate_json((index_dir / "manifest.json").read_text(encoding="utf-8"))
        checksum = hashlib.sha256(
            (index_dir / "index.faiss").read_bytes()
            + (index_dir / "catalog.json").read_bytes()
        ).hexdigest()
        if checksum != self.manifest.checksum:
            raise ValueError("Faiss index checksum mismatch")
        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        catalog_data = json.loads((index_dir / "catalog.json").read_text(encoding="utf-8"))
        self.catalog = [ProductCandidate.model_validate(item) for item in catalog_data]
        if self.index.ntotal != self.manifest.item_count or len(self.catalog) != self.manifest.item_count:
            raise ValueError("Faiss index, manifest, and catalog counts differ")
        self.index_version = self.manifest.checksum[:12]

    def assert_compatible(self, model: ModelManifest) -> None:
        self.manifest.assert_compatible(model)

    async def search(self, vector: list[float], top_k: int) -> ProviderResult[list[ANNHit]]:
        started = perf_counter()
        query = np.asarray([vector], dtype=np.float32)
        if query.shape[1] != self.manifest.embedding_dimension:
            raise ValueError("query vector dimension does not match Faiss index")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.manifest.item_count))
        hits = [
            ANNHit(candidate=self.catalog[int(index)], ann_score=float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]
        return ProviderResult(
            "faiss_hnsw",
            "real",
            int((perf_counter() - started) * 1000),
            hits,
            response_summary=f"faiss hits={len(hits)}",
        )
```

- [ ] **Step 4: Run the real Faiss test**

Run:

```bash
uv run pytest tests/test_recall.py::test_faiss_hnsw_returns_catalog_candidates -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add app/recall/faiss_index.py tests/test_recall.py tests/fixtures/recall_catalog.jsonl
git commit -m "feat: add versioned faiss hnsw index"
```

### Task 4: Add HTTP Embedding And Reranker Adapters

**Files:**
- Create: `app/recall/http_embedding.py`
- Create: `app/recall/http_reranker.py`
- Modify: `tests/test_real_provider_adapters.py`

**Interfaces:**
- Produces: `HttpEmbeddingProvider(base_url, model_manifest, api_key, client=None)`
- Produces: `HttpRerankerProvider(base_url, model, api_key, timeout_seconds, client=None)`

- [ ] **Step 1: Write adapter tests with `httpx.MockTransport`**

Add tests that assert these exact contracts to `tests/test_real_provider_adapters.py`:

```python
@pytest.mark.asyncio
async def test_http_embedding_provider_checks_response_manifest():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings/query"
        assert request.headers["authorization"] == "Bearer embed-key"
        return httpx.Response(200, json={"model_bundle_version": "bundle-v1", "dimension": 3, "vectors": [[1, 0, 0]]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = HttpEmbeddingProvider(
        base_url="https://embed.example",
        model_manifest=ModelManifest(model_bundle_version="bundle-v1", embedding_dimension=3),
        api_key="embed-key",
        client=client,
    )

    result = await provider.encode_query("travel bag")

    assert result.data == [1.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_http_reranker_preserves_candidate_identity():
    hit = ANNHit(candidate=ProductCandidate(id="x", platform="p", title="bag", price=1, url="https://x"), ann_score=0.7)

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "BAAI/bge-reranker-v2-m3", "scores": [{"item_id": "x", "score": 0.9}]})

    provider = HttpRerankerProvider("https://rerank.example", "BAAI/bge-reranker-v2-m3", "key", 30, httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await provider.rerank("travel", [hit], top_k=10)

    assert result.data[0].candidate.id == "x"
    assert result.data[0].rerank_score == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_real_provider_adapters.py -k "embedding_provider or reranker" -q
```

Expected: FAIL because both adapters are missing.

- [ ] **Step 3: Implement the HTTP contracts**

Create `app/recall/http_embedding.py`:

```python
from __future__ import annotations

from time import perf_counter
from typing import Any

import httpx

from app.providers.base import ProviderResult
from app.recall.models import ModelManifest


class HttpEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model_manifest: ModelManifest,
        api_key: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.manifest = model_manifest
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=30)

    async def _encode(
        self,
        path: str,
        payload: dict[str, Any],
        expected_count: int,
    ) -> ProviderResult[list[list[float]]]:
        started = perf_counter()
        response = await self.client.post(
            f"{self.base_url}{path}",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        body = response.json()
        if body.get("model_bundle_version") != self.manifest.model_bundle_version:
            raise ValueError("embedding response model_bundle_version mismatch")
        if body.get("dimension") != self.manifest.embedding_dimension:
            raise ValueError("embedding response dimension mismatch")
        vectors = body.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != expected_count:
            raise ValueError("embedding response vector count mismatch")
        normalized: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list) or len(vector) != self.manifest.embedding_dimension:
                raise ValueError("embedding response vector length mismatch")
            normalized.append([float(value) for value in vector])
        return ProviderResult(
            provider="http_embedding",
            provider_mode="real",
            latency_ms=int((perf_counter() - started) * 1000),
            data=normalized,
            response_summary=f"{path} vectors={len(normalized)}",
        )

    async def encode_query(self, text: str) -> ProviderResult[list[float]]:
        result = await self._encode("/v1/embeddings/query", {"texts": [text]}, 1)
        return ProviderResult(**{**result.__dict__, "data": result.data[0]})

    async def encode_user(self, profile: dict[str, Any]) -> ProviderResult[list[float]]:
        result = await self._encode("/v1/embeddings/user", {"profiles": [profile]}, 1)
        return ProviderResult(**{**result.__dict__, "data": result.data[0]})

    async def project_personalization(
        self,
        query_vector: list[float],
        user_vector: list[float],
    ) -> ProviderResult[list[float]]:
        result = await self._encode(
            "/v1/embeddings/personalization",
            {"query_vectors": [query_vector], "user_vectors": [user_vector]},
            1,
        )
        return ProviderResult(**{**result.__dict__, "data": result.data[0]})

    async def encode_items(
        self,
        items: list[dict[str, Any]],
    ) -> ProviderResult[list[list[float]]]:
        return await self._encode("/v1/embeddings/item", {"items": items}, len(items))
```

Create `app/recall/http_reranker.py`:

```python
from __future__ import annotations

from time import perf_counter

import httpx

from app.providers.base import ProviderResult
from app.recall.models import ANNHit


class HttpRerankerProvider:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_seconds: float, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(timeout=timeout_seconds)

    async def rerank(self, query: str, hits: list[ANNHit], top_k: int) -> ProviderResult[list[ANNHit]]:
        started = perf_counter()
        response = await self.client.post(
            f"{self.base_url}/v1/rerank",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "query": query,
                "items": [
                    {"item_id": hit.candidate.id, "title": hit.candidate.title}
                    for hit in hits
                ],
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        body = response.json()
        if body.get("model") != self.model:
            raise ValueError("reranker response model mismatch")
        by_id = {hit.candidate.id: hit for hit in hits}
        seen: set[str] = set()
        reranked: list[ANNHit] = []
        for row in body.get("scores", []):
            item_id = str(row.get("item_id", ""))
            if item_id not in by_id:
                raise ValueError(f"reranker returned unknown item_id: {item_id}")
            if item_id in seen:
                raise ValueError(f"reranker returned duplicate item_id: {item_id}")
            seen.add(item_id)
            reranked.append(
                by_id[item_id].model_copy(update={"rerank_score": float(row["score"])})
            )
        reranked.sort(
            key=lambda hit: (
                hit.rerank_score if hit.rerank_score is not None else float("-inf")
            ),
            reverse=True,
        )
        return ProviderResult(
            provider="http_reranker",
            provider_mode="real",
            latency_ms=int((perf_counter() - started) * 1000),
            data=reranked[:top_k],
            response_summary=f"reranked={len(reranked[:top_k])}",
        )
```

- [ ] **Step 4: Run adapter tests and commit**

Run:

```bash
uv run pytest tests/test_real_provider_adapters.py -q
```

Expected: all adapter tests PASS.

```bash
git add app/recall/http_embedding.py app/recall/http_reranker.py tests/test_real_provider_adapters.py
git commit -m "feat: add embedding and reranker adapters"
```

### Task 5: Configure And Register Recall Providers

**Files:**
- Modify: `app/config.py`
- Modify: `app/providers/registry.py`
- Modify: `.env.example`
- Test: `tests/test_config.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Produces settings for embedding, Faiss, reranker, weights, Top-K, and fallback.
- Produces: `ProviderRegistry.recall: RecallProvider | None`
- Produces: `ProviderRegistry.embedding: EmbeddingProvider | None` for OpenSearch reuse.

- [ ] **Step 1: Write failing settings and registry tests**

Reuse `submission_settings()` created in `tests/test_config.py` by the fork plan. Add `from dataclasses import replace` to both test modules, and add this helper to `tests/test_providers.py`:

```python
def submission_settings() -> OmniMatchSettings:
    return OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="placeholder",
    )
```

Add tests that assert:

```python
def test_recall_settings_validate_weights():
    settings = replace(submission_settings(), recall_alpha=0.8, recall_beta=0.3)
    with pytest.raises(ConfigError, match="sum to 1"):
        settings.validate()


def test_submission_registry_builds_deterministic_recall():
    settings = replace(submission_settings(), recall_provider="placeholder")
    registry = ProviderRegistry.from_settings(settings)
    assert isinstance(registry.recall, ThreeTowerRecallService)
    assert registry.embedding is registry.recall.embedding
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_config.py tests/test_providers.py -k recall -q
```

Expected: FAIL because recall configuration and registry fields do not exist.

- [ ] **Step 3: Add settings**

Add defaults after existing optional settings in `OmniMatchSettings`:

```python
    recall_provider: str = "product"
    embedding_provider: str = "placeholder"
    embedding_base_url: str | None = None
    embedding_model_bundle_version: str = "fixture-v1"
    embedding_dimension: int = 8
    faiss_index_dir: str | None = None
    reranker_provider: str = "placeholder"
    reranker_base_url: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    recall_alpha: float = 0.7
    recall_beta: float = 0.3
    recall_top_k: int = 100
    rerank_top_k: int = 10
    allow_product_fallback: bool = False
```

Read matching `OMNIMATCH_*` environment variables in every profile. Validate positive dimensions and Top-K values, `rerank_top_k <= recall_top_k`, weights in `[0, 1]`, and `abs(alpha + beta - 1) <= 1e-6`. For `recall_provider="faiss"`, require `embedding_base_url`, `faiss_index_dir`, and `reranker_base_url` unless the matching provider is explicitly `placeholder`.

Add this parser beside the numeric environment helpers and use it for `OMNIMATCH_ALLOW_PRODUCT_FALLBACK`:

```python
def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean")
```

Document every variable in `.env.example`, including:

```dotenv
OMNIMATCH_RECALL_PROVIDER=faiss
OMNIMATCH_EMBEDDING_PROVIDER=http
OMNIMATCH_EMBEDDING_BASE_URL=
OMNIMATCH_EMBEDDING_MODEL_BUNDLE_VERSION=
OMNIMATCH_EMBEDDING_DIMENSION=768
OMNIMATCH_FAISS_INDEX_DIR=
OMNIMATCH_RERANKER_PROVIDER=http
OMNIMATCH_RERANKER_BASE_URL=
OMNIMATCH_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
OMNIMATCH_RECALL_ALPHA=0.7
OMNIMATCH_RECALL_BETA=0.3
OMNIMATCH_RECALL_TOP_K=100
OMNIMATCH_RERANK_TOP_K=10
OMNIMATCH_ALLOW_PRODUCT_FALLBACK=false
```

- [ ] **Step 4: Extend `ProviderRegistry` without breaking manual test construction**

Add final optional fields so existing tests can continue constructing four providers:

```python
    recall: RecallProvider | None = None
    embedding: EmbeddingProvider | None = None
```

In `from_settings()`, build:

- Build `embedding` first, independently of product recall: deterministic for `embedding_provider="placeholder"`, HTTP for `embedding_provider="http"`.
- Set `recall=None` for `recall_provider="product"`, but keep the configured `embedding` available for the later OpenSearch plan.
- Build `ThreeTowerRecallService` with that exact deterministic embedding instance for `recall_provider="placeholder"`.
- Build `ThreeTowerRecallService(embedding, FaissANNProvider, HttpRerankerProvider, alpha, beta)` for `recall_provider="faiss"`; require the embedding instance to be HTTP in `dev`.

Extend `provider_modes()` with `embedding`, `recall`, and `reranker` keys when configured, using the configured provider IDs rather than inferring from the profile alone.

Use `OMNIMATCH_EMBEDDING_API_KEY` and `OMNIMATCH_RERANKER_API_KEY` only in adapters and never log them.

- [ ] **Step 5: Run configuration and registry tests**

Run:

```bash
uv run pytest tests/test_config.py tests/test_providers.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/providers/registry.py .env.example tests/test_config.py tests/test_providers.py
git commit -m "feat: register three tower recall providers"
```

### Task 6: Route ItemSearch Through Recall With Explicit Fallback

**Files:**
- Modify: `app/tools/context.py`
- Modify: `app/tools/item_search.py`
- Modify: `app/tools/shopping_summary.py`
- Modify: `app/schemas.py`
- Modify: `tests/test_tools.py`
- Modify: `tests/test_agent_loop.py`

**Interfaces:**
- Produces: `ToolContext.user_profile: dict[str, Any]`
- Produces: `ProductCandidate.ann_score` and `.rerank_score`
- Preserves: `search_items(intent, insight, ctx) -> list[ProductCandidate]`

- [ ] **Step 1: Write failing ItemSearch recall and fallback tests**

Add to `tests/test_tools.py`:

```python
from dataclasses import replace

from app.recall.models import ANNHit


class FakeRecallProvider:
    async def search(self, query, user_profile, top_k, rerank_k):
        candidate = ProductCandidate(
            id="recalled",
            platform="fixture",
            title="Canvas travel organizer",
            price=198,
            rating=4.7,
            url="https://example.com/recalled",
            evidence=["faiss fixture"],
        )
        return ProviderResult(
            provider="fake_recall",
            provider_mode="fake",
            latency_ms=2,
            data=[ANNHit(candidate=candidate, ann_score=0.8, rerank_score=0.9)],
        )


class RaisingRecallProvider:
    async def search(self, query, user_profile, top_k, rerank_k):
        raise RuntimeError("index offline")


@pytest.mark.asyncio
async def test_item_search_prefers_three_tower_recall():
    settings = replace(submission_settings(), recall_provider="placeholder")
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, recall=FakeRecallProvider())
    ctx = ToolContext(
        settings=settings,
        providers=providers,
        user_profile={"history": ["canvas"]},
    )
    intent = ShoppingIntent(
        original_query="旅行收纳",
        category="旅行收纳",
        preferences=[],
    )
    candidates = await search_items(intent, {}, ctx)
    assert candidates[0].ann_score == 0.8
    assert candidates[0].rerank_score == 0.9
    assert ctx.observations[-1]["recall_mode"] == "three_tower"


@pytest.mark.asyncio
async def test_item_search_provider_fallback_is_disclosed():
    settings = replace(
        submission_settings(),
        recall_provider="placeholder",
        allow_product_fallback=True,
    )
    base = ProviderRegistry.from_settings(settings)
    providers = replace(base, recall=RaisingRecallProvider())
    ctx = ToolContext(settings=settings, providers=providers)
    intent = ShoppingIntent(
        original_query="旅行收纳",
        category="旅行收纳",
        preferences=[],
    )
    candidates = await search_items(intent, {"platforms": ["Amazon"]}, ctx)
    assert candidates
    assert ctx.observations[-1]["recall_mode"] == "provider_fallback"
    assert "recall failed" in ctx.observations[-1]["warnings"][0]

    summary = await build_summary("旅行收纳", [], ctx)
    assert any("provider fallback" in warning.lower() for warning in summary.warnings)
```

Reuse the explicit `submission_settings()` helper already present in `tests/test_tools.py`; do not monkeypatch `search_items()` internals.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_tools.py -k "three_tower_recall or provider_fallback" -q
```

Expected: FAIL because ItemSearch ignores `registry.recall`.

- [ ] **Step 3: Add candidate scores and user context**

Append optional fields to `ProductCandidate`:

```python
    ann_score: float | None = None
    rerank_score: float | None = None
```

Append to `ToolContext`:

```python
    user_profile: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Implement recall-first ItemSearch**

Replace `search_items()` with the same signature and this control flow:

```python
async def search_items(intent: ShoppingIntent, insight: dict, ctx: ToolContext) -> list[ProductCandidate]:
    query = f"{intent.category} {' '.join(intent.preferences)}".strip()
    if ctx.providers.recall is not None:
        try:
            result = await ctx.providers.recall.search(
                query=query,
                user_profile=ctx.user_profile,
                top_k=ctx.settings.recall_top_k,
                rerank_k=ctx.settings.rerank_top_k,
            )
            candidates = [
                hit.candidate.model_copy(
                    update={"ann_score": hit.ann_score, "rerank_score": hit.rerank_score}
                )
                for hit in result.data
            ]
            ctx.observations.append(
                {
                    "tool": "ItemSearch",
                    "provider": result.provider,
                    "provider_mode": result.provider_mode,
                    "latency_ms": result.latency_ms,
                    "warnings": result.warnings,
                    "recall_mode": "three_tower",
                    "candidate_count": len(candidates),
                }
            )
            return candidates
        except Exception as exc:
            if not ctx.settings.allow_product_fallback:
                raise
            recall_warning = f"recall failed; used Product Provider fallback: {exc}"
            recall_mode = "provider_fallback"
    else:
        recall_warning = "three-tower recall disabled; used configured Product Provider"
        recall_mode = "product_provider"

    platforms = insight.get("platforms", DEFAULT_PLATFORMS)
    result = await ctx.providers.product.search(query, platforms=platforms)
    ctx.observations.append(
        {
            "tool": "ItemSearch",
            "provider": result.provider,
            "provider_mode": result.provider_mode,
            "latency_ms": result.latency_ms,
            "warnings": [recall_warning, *result.warnings],
            "recall_mode": recall_mode,
            "candidate_count": len(result.data),
        }
    )
    return [ProductCandidate(**item) for item in result.data]
```

- [ ] **Step 5: Run tool and agent tests**

Before running tests, update `build_summary()` to aggregate unique observation warnings and recall modes:

```python
    observation_warnings = list(
        dict.fromkeys(
            str(warning)
            for observation in ctx.observations
            for warning in observation.get("warnings", [])
        )
    )
    recall_modes = sorted(
        {
            str(observation["recall_mode"])
            for observation in ctx.observations
            if observation.get("recall_mode")
        }
    )
    warnings.extend(observation_warnings)
    if recall_modes:
        warnings.append(f"recall modes: {', '.join(recall_modes)}")
```

Insert this after the existing provider-mode warning and before constructing `ShoppingSummary`.

Run:

```bash
uv run pytest tests/test_tools.py tests/test_agent_loop.py tests/test_schemas.py -q
```

Expected: all tests PASS; existing manual settings continue through the Product Provider path unless recall is configured.

- [ ] **Step 6: Commit**

```bash
git add app/tools/context.py app/tools/item_search.py app/tools/shopping_summary.py app/schemas.py tests/test_tools.py tests/test_agent_loop.py
git commit -m "feat: route item search through three tower recall"
```

### Task 7: Add Index Build, Benchmark, Documentation, And Phase Verification

**Files:**
- Create: `examples/build_faiss_index.py`
- Create: `examples/benchmark_faiss.py`
- Modify: `README.md`
- Replace: `app/recall/ann.py`
- Replace: `app/recall/tower_user.py`
- Replace: `app/recall/tower_query.py`
- Replace: `app/recall/tower_item.py`

**Interfaces:**
- Produces CLI: `uv run python examples/build_faiss_index.py --catalog tests/fixtures/recall_catalog.jsonl --output /tmp/omnimatch-faiss --embedding-base-url http://127.0.0.1:8100 --model-bundle-version fixture-v1 --dimension 8`
- Produces CLI: `uv run python examples/benchmark_faiss.py --index /tmp/omnimatch-faiss --queries /tmp/omnimatch-query-vectors.jsonl --iterations 1000`

- [ ] **Step 1: Write failing CLI contract tests**

Add to `tests/test_recall.py`:

```python
from examples.benchmark_faiss import build_parser as build_benchmark_parser
from examples.build_faiss_index import build_parser as build_index_parser


def test_faiss_cli_defaults_are_reproducible():
    index_args = build_index_parser().parse_args(
        [
            "--catalog", "catalog.jsonl",
            "--output", "index-dir",
            "--embedding-base-url", "http://127.0.0.1:8100",
            "--model-bundle-version", "bundle-v1",
            "--dimension", "8",
        ]
    )
    benchmark_args = build_benchmark_parser().parse_args(
        ["--index", "index-dir", "--queries", "queries.jsonl"]
    )

    assert index_args.batch_size == 128
    assert index_args.hnsw_m == 32
    assert benchmark_args.warmup == 100
    assert benchmark_args.iterations == 1000
    assert benchmark_args.top_k == 100
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
uv run pytest tests/test_recall.py -k faiss_cli_defaults -q
```

Expected: FAIL because the two example modules do not exist.

- [ ] **Step 3: Implement the index builder CLI**

Create `examples/build_faiss_index.py` with an async `build(args)` function that loads non-empty JSONL rows, validates each row as `ProductCandidate`, rejects duplicate IDs, calls `HttpEmbeddingProvider.encode_items()` in `--batch-size` chunks, and passes all vectors to `build_faiss_index()`. Required arguments are `--catalog`, `--output`, `--embedding-base-url`, `--model-bundle-version`, and `--dimension`; defaults are `--batch-size 128` and `--hnsw-m 32`. Read the secret only from `OMNIMATCH_EMBEDDING_API_KEY` and print only `manifest.model_dump_json(indent=2)`.

Expose a `build_parser()` used by this entrypoint:

```python
def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.batch_size < 1 or args.hnsw_m < 2:
        parser.error("batch-size must be >= 1 and hnsw-m must be >= 2")
    asyncio.run(build(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-base-url", required=True)
    parser.add_argument("--model-bundle-version", required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hnsw-m", type=int, default=32)
    return parser


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the benchmark CLI**

Create `examples/benchmark_faiss.py`. It loads `FaissANNProvider`, validates objects such as `{"vector": [1.0, 0.0, 0.0]}` with one vector per JSONL row, performs `--warmup 100` untimed searches, then round-robins queries for `--iterations 1000`. Measure each awaited `provider.search(vector, top_k=100)` with `perf_counter()` and calculate percentiles with `numpy.percentile`.

Print exactly these keys:

```python
report = {
    "machine": platform.platform(),
    "python": platform.python_version(),
    "faiss": getattr(faiss, "__version__", "unknown"),
    "index_manifest": provider.manifest.model_dump(),
    "query_count": len(vectors),
    "iterations": args.iterations,
    "p50_ms": float(np.percentile(latencies, 50)),
    "p95_ms": float(np.percentile(latencies, 95)),
    "p99_ms": float(np.percentile(latencies, 99)),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
if args.output is not None:
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
if args.max_p99_ms is not None and report["p99_ms"] > args.max_p99_ms:
    raise SystemExit(1)
```

Expose `build_parser()` with required `--index`/`--queries` plus defaults `--warmup 100`, `--iterations 1000`, and `--top-k 100`. Add optional `--output Path` and `--max-p99-ms float` arguments. Reject empty query files, dimensions unequal to `provider.manifest.embedding_dimension`, non-positive warmup/iteration values, and any vector containing a non-finite number.

- [ ] **Step 5: Remove misleading stub behavior and document operations**

Replace the four old stub modules with compatibility imports and deprecation docstrings:

```python
# app/recall/ann.py
"""Compatibility imports; ANN implementations live in recall providers."""
from app.recall.faiss_index import FaissANNProvider, build_faiss_index

__all__ = ["FaissANNProvider", "build_faiss_index"]
```

The three tower modules re-export `EmbeddingProvider` and state that tower execution belongs to the versioned embedding service. They must not retain input-length vectors.

Add README sections for model/index compatibility, Faiss build, benchmark, Product Provider fallback disclosure, and the distinction between Query-tower request encoding and Item-tower offline indexing.

- [ ] **Step 6: Run focused and full verification**

Run:

```bash
uv run pytest tests/test_recall.py tests/test_tools.py tests/test_providers.py tests/test_config.py -q
```

Expected: all focused tests PASS.

Run:

```bash
uv run pytest -q
```

Expected: the full backend suite passes with zero failures.

Run:

```bash
OMNIMATCH_PROFILE=submission uv run python examples/run_competition_agent.py
```

Expected: exit `0`, deterministic products, and placeholder/fallback disclosure.

- [ ] **Step 7: Commit**

```bash
git add app/recall examples/build_faiss_index.py examples/benchmark_faiss.py README.md tests/test_recall.py
git commit -m "docs: add faiss index operations and benchmarks"
```

## Phase Acceptance Checklist

- [ ] No input-length vector stub remains active.
- [ ] Semantic/personalization fusion is normalized, tested, and cold-start safe.
- [ ] Model and index manifests reject version, dimension, normalization, or metric mismatches.
- [ ] A real Faiss HNSW inner-product fixture returns catalog-backed candidates.
- [ ] ItemSearch uses Top-100 recall and Top-10 reranking when recall is configured.
- [ ] Reranker failure and Product Provider fallback are explicit in observations and final warnings.
- [ ] Query-side encoding and Item index encoding retain separate tower responsibilities.
- [ ] Benchmark output includes P50/P95/P99 and reproducibility metadata.
- [ ] Milvus and OpenSearch remain outside the product ANN implementation.
- [ ] Full backend and submission regression commands pass.
