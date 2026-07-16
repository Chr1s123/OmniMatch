# Retrieval Training, Evaluation, And Model Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible training and release pipeline for the User/Query/Item model bundle and BGE reranker, including false-negative filtering, staged training, offline metrics, release gates, index construction, and rollback metadata.

**Architecture:** Keep training code in a separate `app.training` package and optional dependency group so the API runtime stays lightweight. JSONL event contracts feed time-split datasets; three-tower CPT/SFT/DPO and reranker pointwise/pairwise/listwise/DPO stages emit immutable manifests and reports. A release gate promotes only artifacts whose measured reports meet every quality and latency threshold, then builds new Faiss/OpenSearch indices without mutating current aliases in place.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers/Datasets, PEFT LoRA, Accelerate, NumPy, safetensors, pytest.

## Global Constraints

- Training and AgentLoop are independent optimization chains connected only through versioned runtime contracts.
- Data splits are chronological and must prevent user-event or item-version leakage.
- Same-item SKU and cross-language duplicates are removed from negative samples.
- Training stages and learning rates are CPT `1e-5`, SFT `5e-6`, DPO `1e-6`.
- InfoNCE temperatures are cross-language `0.02` and same-language `0.05`.
- Hard Negative weight is `2.0`.
- The cross-language auxiliary loss is named `L_align`.
- Reranker stages are Pointwise BCE, Pairwise MarginRanking, Listwise ApproxNDCG, then market-aware DPO.
- Reranker base model is `BAAI/bge-reranker-v2-m3`; fine-tuning uses LoRA.
- Release gates are `Recall@100 >= 0.85`, `NDCG@10 >= 0.55`, cross-language Recall Gap `<= 0.05`, Cohen's Kappa `>= 0.75`, Faiss P99 `< 50ms`, and reranker P99 `<= 100ms`.
- Missing real logs or GPU results must never be represented as passing production metrics.
- `test` uses tiny local fixtures and does not download models or require GPU.
- This plan does not implement AgentLoop fork, OpenSearch serving, or runtime Faiss search; it consumes their contracts.

---

## Current State

- `app/eval` evaluates final Agent text with required/forbidden terms; it does not evaluate retrieval models.
- No event schema, dataset builder, false-negative filter, chronological split, or training fixture exists.
- No PyTorch, Transformers, Datasets, PEFT, Accelerate, or safetensors dependency exists.
- No three-tower model, personalization projection, InfoNCE, `L_align`, or DPO loss exists.
- No reranker training or GPU benchmark exists.
- No `Recall@100`, `NDCG@10`, language-gap, Kappa, or promotion-gate implementation exists.
- No immutable model Bundle or release report is produced.

## Implementation References

- [BAAI BGE Reranker v2 M3 model card](https://huggingface.co/BAAI/bge-reranker-v2-m3)

## File Structure

- Create: `app/training/schema.py`
  - Owns event, item, pair, preference, split, stage, report, and Bundle schemas.
- Create: `app/training/dataset.py`
  - Loads JSONL, validates chronology, filters false negatives, and writes deterministic splits.
- Create: `app/training/three_tower.py`
  - Owns the three encoders, personalization projection, pooling, and normalized outputs.
- Create: `app/training/losses.py`
  - Owns dynamic-temperature InfoNCE, hard-negative weighting, `L_align`, MarginMSE, and embedding DPO.
- Create: `app/training/stages.py`
  - Owns immutable stage configs and stage transition checks.
- Create: `app/training/train_three_tower.py`
  - Runs CPT, SFT, and DPO from explicit config.
- Create: `app/training/reranker.py`
  - Builds BGE reranker with LoRA and all four training losses.
- Create: `app/training/train_reranker.py`
  - Runs reranker stages and Hard Negative rounds.
- Create: `app/training/metrics.py`
  - Computes release quality metrics.
- Create: `app/training/release.py`
  - Validates reports, creates Bundle manifests, and controls promotion eligibility.
- Create: `examples/prepare_retrieval_dataset.py`
- Create: `examples/train_three_tower.py`
- Create: `examples/train_reranker.py`
- Create: `examples/evaluate_retrieval.py`
- Create: `examples/package_model_bundle.py`
- Create: `examples/promote_model_bundle.py`
- Create: `examples/benchmark_reranker.py`
- Create: `tests/fixtures/retrieval_events.jsonl`
- Create: `tests/fixtures/retrieval_items.jsonl`
- Create: `tests/test_training_dataset.py`
- Create: `tests/test_training_losses.py`
- Create: `tests/test_retrieval_metrics.py`
- Create: `tests/test_model_release.py`
- Modify: `pyproject.toml`, `uv.lock`, `README.md`

### Task 1: Add Optional ML Dependencies And Strict Data Contracts

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `app/training/__init__.py`
- Create: `app/training/schema.py`
- Create: `app/training/dataset.py`
- Create: `tests/fixtures/retrieval_events.jsonl`
- Create: `tests/fixtures/retrieval_items.jsonl`
- Create: `tests/test_training_dataset.py`

**Interfaces:**
- Produces: `InteractionEvent`, `CatalogItem`, `TrainingExample`, `DatasetSplit`
- Produces: `load_events(path: Path)`, `build_examples(events, items)`, and `chronological_split(events, train_ratio, validation_ratio)`

- [ ] **Step 1: Add concrete fixture rows**

Create `tests/fixtures/retrieval_items.jsonl`:

```jsonl
{"item_id":"sku-cn-1","item_group_id":"group-1","language":"zh","market":"CN","title":"帆布旅行收纳袋","category":"travel_bag","attributes":{"material":"canvas"},"version":"v1"}
{"item_id":"sku-en-1","item_group_id":"group-1","language":"en","market":"US","title":"Canvas travel organizer","category":"travel_bag","attributes":{"material":"canvas"},"version":"v1"}
{"item_id":"sku-cn-2","item_group_id":"group-2","language":"zh","market":"CN","title":"塑料玩具箱","category":"toy","attributes":{"material":"plastic"},"version":"v1"}
```

Create `tests/fixtures/retrieval_events.jsonl`:

```jsonl
{"user_id":"u1","query":"旅行收纳袋","item_id":"sku-cn-1","market":"CN","language":"zh","event_type":"purchase","impression_position":1,"ranker_score":0.9,"timestamp":"2026-06-01T00:00:00Z","item_group_id":"group-1"}
{"user_id":"u1","query":"旅行收纳袋","item_id":"sku-en-1","market":"US","language":"en","event_type":"impression","impression_position":2,"ranker_score":0.8,"timestamp":"2026-06-01T00:00:01Z","item_group_id":"group-1"}
{"user_id":"u1","query":"旅行收纳袋","item_id":"sku-cn-2","market":"CN","language":"zh","event_type":"impression","impression_position":3,"ranker_score":0.3,"timestamp":"2026-06-01T00:00:02Z","item_group_id":"group-2"}
{"user_id":"u2","query":"canvas organizer","item_id":"sku-en-1","market":"US","language":"en","event_type":"click","impression_position":1,"ranker_score":0.7,"timestamp":"2026-06-02T00:00:00Z","item_group_id":"group-1"}
{"user_id":"u3","query":"玩具箱","item_id":"sku-cn-2","market":"CN","language":"zh","event_type":"purchase","impression_position":1,"ranker_score":0.95,"timestamp":"2026-06-03T00:00:00Z","item_group_id":"group-2"}
```

- [ ] **Step 2: Write failing schema, false-negative, and split tests**

Create `tests/test_training_dataset.py`:

```python
from pathlib import Path

from app.training.dataset import build_examples, chronological_split, load_events, load_items


FIXTURES = Path(__file__).parent / "fixtures"


def test_same_group_cross_language_item_is_not_a_negative():
    events = load_events(FIXTURES / "retrieval_events.jsonl")
    items = load_items(FIXTURES / "retrieval_items.jsonl")
    examples = build_examples(events, items)
    purchase = next(example for example in examples if example.positive_item_id == "sku-cn-1")

    assert "sku-en-1" not in purchase.negative_item_ids
    assert "sku-cn-2" in purchase.negative_item_ids


def test_chronological_split_preserves_time_order():
    events = load_events(FIXTURES / "retrieval_events.jsonl")
    split = chronological_split(events, train_ratio=0.6, validation_ratio=0.2)

    assert max(event.timestamp for event in split.train) <= min(event.timestamp for event in split.validation)
    assert max(event.timestamp for event in split.validation) <= min(event.timestamp for event in split.test)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/test_training_dataset.py -q
```

Expected: FAIL because `app.training` does not exist.

- [ ] **Step 4: Implement strict schemas**

Create `app/training/schema.py`:

```python
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


EventType = Literal["impression", "click", "purchase"]


class InteractionEvent(BaseModel):
    user_id: str
    query: str
    item_id: str
    market: str
    language: str
    event_type: EventType
    impression_position: int = Field(ge=1)
    ranker_score: float
    timestamp: datetime
    item_group_id: str


class CatalogItem(BaseModel):
    item_id: str
    item_group_id: str
    language: str
    market: str
    title: str
    category: str
    attributes: dict[str, Any]
    version: str


class TrainingExample(BaseModel):
    user_id: str
    query: str
    query_language: str
    positive_item_id: str
    negative_item_ids: list[str]
    hard_negative_item_ids: list[str] = Field(default_factory=list)
    event_type: EventType


class DatasetSplit(BaseModel):
    train: list[InteractionEvent]
    validation: list[InteractionEvent]
    test: list[InteractionEvent]
```

- [ ] **Step 5: Implement loaders, false-negative filtering, and chronological split**

Create `app/training/dataset.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from app.training.schema import CatalogItem, DatasetSplit, InteractionEvent, TrainingExample


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_events(path: Path) -> list[InteractionEvent]:
    return [InteractionEvent.model_validate(row) for row in _load_jsonl(path)]


def load_items(path: Path) -> dict[str, CatalogItem]:
    items = [CatalogItem.model_validate(row) for row in _load_jsonl(path)]
    if len(items) != len({item.item_id for item in items}):
        raise ValueError("duplicate item_id in catalog")
    return {item.item_id: item for item in items}


def build_examples(events: list[InteractionEvent], items: dict[str, CatalogItem]) -> list[TrainingExample]:
    grouped: dict[tuple[str, str], list[InteractionEvent]] = {}
    for event in events:
        if event.item_id not in items:
            raise ValueError(f"event references missing item: {event.item_id}")
        grouped.setdefault((event.user_id, event.query), []).append(event)
    examples: list[TrainingExample] = []
    for query_events in grouped.values():
        positives = [event for event in query_events if event.event_type in {"click", "purchase"}]
        impressions = [event for event in query_events if event.event_type == "impression"]
        for positive in positives:
            positive_group = items[positive.item_id].item_group_id
            negatives = [
                event.item_id
                for event in impressions
                if items[event.item_id].item_group_id != positive_group
            ]
            examples.append(
                TrainingExample(
                    user_id=positive.user_id,
                    query=positive.query,
                    query_language=positive.language,
                    positive_item_id=positive.item_id,
                    negative_item_ids=list(dict.fromkeys(negatives)),
                    hard_negative_item_ids=[
                        event.item_id
                        for event in impressions
                        if items[event.item_id].item_group_id != positive_group
                        and event.ranker_score > 0.3
                    ],
                    event_type=positive.event_type,
                )
            )
        exposed_ids = {event.item_id for event in query_events}
        for exposure in impressions:
            exposure_group = items[exposure.item_id].item_group_id
            unexposed_negatives = [
                item_id
                for item_id, item in sorted(items.items())
                if item_id not in exposed_ids and item.item_group_id != exposure_group
            ][:100]
            if unexposed_negatives:
                examples.append(
                    TrainingExample(
                        user_id=exposure.user_id,
                        query=exposure.query,
                        query_language=exposure.language,
                        positive_item_id=exposure.item_id,
                        negative_item_ids=unexposed_negatives,
                        event_type="impression",
                    )
                )
    return examples


def chronological_split(events: list[InteractionEvent], train_ratio: float, validation_ratio: float) -> DatasetSplit:
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1 or train_ratio + validation_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than 1")
    ordered = sorted(events, key=lambda event: event.timestamp)
    train_end = max(1, int(len(ordered) * train_ratio))
    validation_end = max(train_end + 1, int(len(ordered) * (train_ratio + validation_ratio)))
    if validation_end >= len(ordered):
        validation_end = len(ordered) - 1
    return DatasetSplit(
        train=ordered[:train_end],
        validation=ordered[train_end:validation_end],
        test=ordered[validation_end:],
    )
```

- [ ] **Step 6: Add ML dependency group and run tests**

Run:

```bash
uv add --group ml torch transformers datasets peft accelerate safetensors numpy
```

Expected: dependencies appear only in the `ml` group and `uv.lock` updates.

Run:

```bash
uv run pytest tests/test_training_dataset.py -q
```

Expected: `2 passed` without model downloads.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/training tests/fixtures/retrieval_events.jsonl tests/fixtures/retrieval_items.jsonl tests/test_training_dataset.py
git commit -m "feat: add retrieval training data contracts"
```

### Task 2: Implement Three-Tower Model And Losses

**Files:**
- Create: `app/training/three_tower.py`
- Create: `app/training/losses.py`
- Create: `tests/test_training_losses.py`

**Interfaces:**
- Produces: `ThreeTowerModel`
- Produces: `dynamic_info_nce(query_vectors, item_vectors, query_languages, item_languages, hard_negative_mask, hard_negative_weight=2.0)`
- Produces: `cross_language_alignment_loss(left, right)`
- Produces: `embedding_dpo_loss(query, chosen, rejected, beta=0.1)`
- Produces: `margin_mse_distillation(student_positive, student_negative, teacher_positive, teacher_negative)`

- [ ] **Step 1: Write failing tensor-only loss tests**

Create `tests/test_training_losses.py`:

```python
import pytest

torch = pytest.importorskip("torch")

from app.training.losses import (
    cross_language_alignment_loss,
    dynamic_info_nce,
    embedding_dpo_loss,
)


def test_dynamic_info_nce_rewards_diagonal_pairs_and_hard_negative_weight():
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    items = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = dynamic_info_nce(
        queries,
        items,
        query_languages=["zh", "en"],
        item_languages=["zh", "en"],
        hard_negative_mask=torch.tensor([[False, True], [False, False]]),
        hard_negative_weight=2.0,
    )
    assert 0 <= loss.item() < 0.1


def test_cross_language_alignment_is_zero_for_equal_vectors():
    vectors = torch.tensor([[1.0, 0.0]])
    assert cross_language_alignment_loss(vectors, vectors).item() == pytest.approx(0.0)


def test_embedding_dpo_prefers_chosen_item():
    query = torch.tensor([[1.0, 0.0]])
    chosen = torch.tensor([[1.0, 0.0]])
    rejected = torch.tensor([[0.0, 1.0]])
    assert embedding_dpo_loss(query, chosen, rejected, beta=0.1).item() < 0.7
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -q
```

Expected: FAIL because model losses do not exist.

- [ ] **Step 3: Implement exact losses**

Create `app/training/losses.py`:

```python
import math

import torch
import torch.nn.functional as F


def dynamic_info_nce(query_vectors, item_vectors, query_languages, item_languages, hard_negative_mask, hard_negative_weight=2.0):
    query_vectors = F.normalize(query_vectors, dim=-1)
    item_vectors = F.normalize(item_vectors, dim=-1)
    similarities = query_vectors @ item_vectors.T
    temperatures = torch.tensor(
        [[0.05 if q_lang == i_lang else 0.02 for i_lang in item_languages] for q_lang in query_languages],
        device=similarities.device,
        dtype=similarities.dtype,
    )
    logits = similarities / temperatures
    logits = logits + hard_negative_mask.to(logits.dtype) * math.log(hard_negative_weight)
    targets = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, targets)


def cross_language_alignment_loss(left, right):
    return F.mse_loss(F.normalize(left, dim=-1), F.normalize(right, dim=-1))


def embedding_dpo_loss(query, chosen, rejected, beta=0.1):
    query = F.normalize(query, dim=-1)
    chosen_score = (query * F.normalize(chosen, dim=-1)).sum(dim=-1)
    rejected_score = (query * F.normalize(rejected, dim=-1)).sum(dim=-1)
    return -F.logsigmoid(beta * (chosen_score - rejected_score)).mean()


def margin_mse_distillation(student_positive, student_negative, teacher_positive, teacher_negative):
    student_margin = student_positive - student_negative
    teacher_margin = teacher_positive - teacher_negative
    return F.mse_loss(student_margin, teacher_margin)
```

- [ ] **Step 4: Implement the three-tower module**

Create `app/training/three_tower.py` with one `AutoModel` per tower, attention-mask mean pooling, `F.normalize`, and:

```python
class ThreeTowerModel(nn.Module):
    def __init__(self, base_model_name: str) -> None:
        super().__init__()
        self.user_tower = AutoModel.from_pretrained(base_model_name)
        self.query_tower = AutoModel.from_pretrained(base_model_name)
        self.item_tower = AutoModel.from_pretrained(base_model_name)
        hidden = self.query_tower.config.hidden_size
        self.personalization_projection = nn.Linear(hidden * 2, hidden)

    @staticmethod
    def _pool(output, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        summed = (output.last_hidden_state * mask).sum(dim=1)
        return F.normalize(summed / mask.sum(dim=1).clamp_min(1), dim=-1)

    def encode_query(self, input_ids, attention_mask):
        return self._pool(self.query_tower(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

    def encode_user(self, input_ids, attention_mask):
        return self._pool(self.user_tower(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

    def encode_item(self, input_ids, attention_mask):
        return self._pool(self.item_tower(input_ids=input_ids, attention_mask=attention_mask), attention_mask)

    def personalize(self, query_vector, user_vector):
        return F.normalize(self.personalization_projection(torch.cat([query_vector, user_vector], dim=-1)), dim=-1)
```

Tests must instantiate the module with a tiny local fake encoder or monkeypatch `AutoModel.from_pretrained`; unit tests must not download a model.

- [ ] **Step 5: Run loss tests and commit**

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -q
```

Expected: all tests PASS.

```bash
git add app/training/three_tower.py app/training/losses.py tests/test_training_losses.py
git commit -m "feat: add three tower model losses"
```

### Task 3: Implement CPT, SFT, And DPO Stage Orchestration

**Files:**
- Create: `app/training/stages.py`
- Create: `app/training/train_three_tower.py`
- Create: `examples/train_three_tower.py`
- Modify: `tests/test_training_losses.py`

**Interfaces:**
- Produces: immutable `StageConfig` values for CPT/SFT/DPO
- Produces CLI: `examples/train_three_tower.py --stage {cpt,sft,dpo}`

- [ ] **Step 1: Write failing stage tests**

Add tests:

```python
from app.training.stages import THREE_TOWER_STAGES, validate_stage_transition


def test_three_tower_stage_learning_rates_are_fixed():
    assert THREE_TOWER_STAGES["cpt"].learning_rate == 1e-5
    assert THREE_TOWER_STAGES["sft"].learning_rate == 5e-6
    assert THREE_TOWER_STAGES["dpo"].learning_rate == 1e-6


def test_stage_transition_requires_previous_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="SFT requires CPT"):
        validate_stage_transition("sft", tmp_path)
```

- [ ] **Step 2: Run stage tests to verify they fail**

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -k stage -q
```

Expected: FAIL because `app.training.stages` does not exist.

- [ ] **Step 3: Implement stage configuration and transition checks**

Create `app/training/stages.py`:

```python
from pathlib import Path
from typing import Literal

from pydantic import BaseModel


StageName = Literal["cpt", "sft", "dpo"]


class StageConfig(BaseModel, frozen=True):
    name: StageName
    learning_rate: float
    output_subdir: str


THREE_TOWER_STAGES = {
    "cpt": StageConfig(name="cpt", learning_rate=1e-5, output_subdir="01-cpt"),
    "sft": StageConfig(name="sft", learning_rate=5e-6, output_subdir="02-sft"),
    "dpo": StageConfig(name="dpo", learning_rate=1e-6, output_subdir="03-dpo"),
}


def validate_stage_transition(stage: StageName, run_dir: Path) -> Path | None:
    if stage == "cpt":
        return None
    previous = run_dir / ("01-cpt" if stage == "sft" else "02-sft") / "checkpoint-complete"
    if not previous.exists():
        label = "SFT requires CPT" if stage == "sft" else "DPO requires SFT"
        raise FileNotFoundError(f"{label}: {previous}")
    return previous.parent
```

- [ ] **Step 4: Implement stage trainers**

Add this configuration and dispatcher to `app/training/train_three_tower.py`:

```python
from pathlib import Path

from pydantic import BaseModel, Field

from app.training.stages import StageName, validate_stage_transition


class TrainConfig(BaseModel):
    stage: StageName
    base_model: str
    events: Path
    items: Path
    run_dir: Path
    seed: int = 42
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)


def run_stage(config: TrainConfig) -> Path:
    validate_stage_transition(config.stage, config.run_dir)
    runners = {"cpt": run_cpt, "sft": run_sft, "dpo": run_dpo}
    return runners[config.stage](config)
```

Implement `run_cpt()` with `AutoModelForMaskedLM`, `AutoTokenizer`, a text dataset built from Query and Item text, `DataCollatorForLanguageModeling(mlm_probability=0.15)`, and `Trainer` using the CPT stage learning rate. Implement `run_sft()` with `ThreeTowerModel` and a custom `Trainer.compute_loss()` that combines `dynamic_info_nce` with multi-task weights `purchase=1.0`, `click=0.5`, `exposure=0.2`, `rank_consistency=0.5`, `align=0.2`; sort curriculum batches by `len(hard_negative_item_ids)` before epoch sampling. Implement `run_dpo()` with chosen/rejected items and `embedding_dpo_loss(beta=0.1)`.

All three stage functions use this finalizer so a completion marker is written last:

```python
def finalize_stage(
    output_dir: Path,
    model,
    tokenizer,
    config: TrainConfig,
    metrics: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_config.json").write_text(
        config.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "checkpoint-complete").write_text("ok\n", encoding="utf-8")
    return output_dir
```

Each stage must write `training_config.json`, `metrics.json`, model weights, tokenizer files, and `checkpoint-complete` only after successful save. `training_config.json` must include dataset checksums, random seed, package versions, git commit, learning rate, temperatures, hard-negative weight, and all multi-task weights.

- [ ] **Step 5: Implement an explicit CLI**

`examples/train_three_tower.py` requires `--stage`, `--base-model`, `--events`, `--items`, `--run-dir`, `--seed`, `--epochs`, `--batch-size`, and `--gradient-accumulation-steps`. It calls `validate_stage_transition()` before importing or allocating a model and exits nonzero on missing prior stages.

- [ ] **Step 6: Run tensor/stage tests and commit**

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -q
```

Expected: all tests PASS without model downloads.

```bash
git add app/training/stages.py app/training/train_three_tower.py examples/train_three_tower.py tests/test_training_losses.py
git commit -m "feat: add staged three tower training"
```

### Task 4: Implement Reranker Training And Hard-Negative Rounds

**Files:**
- Create: `app/training/reranker.py`
- Create: `app/training/train_reranker.py`
- Create: `examples/train_reranker.py`
- Modify: `tests/test_training_losses.py`

**Interfaces:**
- Produces: `build_lora_reranker()`
- Produces losses: pointwise, pairwise, listwise ApproxNDCG, market-aware DPO
- Produces: `mine_round_two_hard_negatives(rows, threshold=0.3)`

- [ ] **Step 1: Write failing reranker loss and mining tests**

Add tests using small tensors and dictionaries:

```python
from app.training.reranker import approx_ndcg_loss, mine_round_two_hard_negatives


def test_round_two_mines_unpurchased_scores_above_point_three():
    rows = [
        {"item_id": "a", "score": 0.31, "purchased": False},
        {"item_id": "b", "score": 0.30, "purchased": False},
        {"item_id": "c", "score": 0.90, "purchased": True},
    ]
    assert [row["item_id"] for row in mine_round_two_hard_negatives(rows)] == ["a"]


def test_approx_ndcg_loss_prefers_correct_order():
    good = approx_ndcg_loss(torch.tensor([[3.0, 2.0, 1.0]]), torch.tensor([[3.0, 2.0, 0.0]]))
    bad = approx_ndcg_loss(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[3.0, 2.0, 0.0]]))
    assert good < bad
```

- [ ] **Step 2: Run reranker tests to verify they fail**

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -k "round_two or approx_ndcg" -q
```

Expected: FAIL because `app.training.reranker` does not exist.

- [ ] **Step 3: Implement reranker model and losses**

Use `AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3", num_labels=1)` and:

```python
def build_lora_reranker(base_model="BAAI/bge-reranker-v2-m3"):
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1)
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query", "key", "value"],
    )
    return get_peft_model(model, config)
```

Implement Pointwise with `binary_cross_entropy_with_logits`, Pairwise with `margin_ranking_loss(margin=0.2)`, differentiable ApproxNDCG with pairwise sigmoid rank approximation, and market-aware DPO by grouping chosen/rejected pairs within the same market.

Use these loss functions in `app/training/reranker.py`:

```python
def pointwise_loss(logits, labels):
    return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


def pairwise_loss(positive_scores, negative_scores):
    target = torch.ones_like(positive_scores)
    return F.margin_ranking_loss(positive_scores, negative_scores, target, margin=0.2)


def approx_ndcg_loss(scores, relevances, temperature=1.0):
    score_differences = scores.unsqueeze(2) - scores.unsqueeze(1)
    approximate_ranks = 1.0 + torch.sigmoid(
        -score_differences / temperature
    ).sum(dim=-1) - 0.5
    gains = torch.pow(2.0, relevances) - 1.0
    discounts = 1.0 / torch.log2(approximate_ranks + 1.0)
    dcg = (gains * discounts).sum(dim=-1)
    ideal_relevances = torch.sort(relevances, descending=True, dim=-1).values
    positions = torch.arange(
        1,
        relevances.shape[-1] + 1,
        device=relevances.device,
        dtype=relevances.dtype,
    )
    ideal_dcg = (
        (torch.pow(2.0, ideal_relevances) - 1.0)
        / torch.log2(positions + 1.0)
    ).sum(dim=-1).clamp_min(1e-8)
    return (1.0 - dcg / ideal_dcg).mean()


def market_dpo_loss(
    chosen_scores,
    rejected_scores,
    chosen_markets,
    rejected_markets,
    beta=0.1,
):
    if len(chosen_markets) != chosen_scores.shape[0]:
        raise ValueError("market count must match pair count")
    if list(chosen_markets) != list(rejected_markets):
        raise ValueError("DPO chosen/rejected pairs must come from the same market")
    return -F.logsigmoid(beta * (chosen_scores - rejected_scores)).mean()
```

Implement:

```python
def mine_round_two_hard_negatives(rows, threshold=0.3):
    return [row for row in rows if row["score"] > threshold and not row["purchased"]]
```

- [ ] **Step 4: Implement ordered reranker stages**

`train_reranker.py` enforces:

```text
01-pointwise -> 02-pairwise -> 03-listwise -> 04-dpo
```

Every stage writes the same reproducibility fields as three-tower training. After the first completed model, run candidate scoring and persist `hard-negatives-round-2.jsonl` using the strict `score > 0.3 and purchased == false` rule. Add bidirectional MarginMSE distillation rows containing teacher and embedding margins.

- [ ] **Step 5: Implement CLI and run tests**

`examples/train_reranker.py` requires explicit dataset/run/model arguments, defaults to FP16 only when CUDA is available, logs the effective batch size, warms up before any benchmark, and refuses to record a GPU latency result on CPU.

Run:

```bash
uv run --group ml pytest tests/test_training_losses.py -q
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app/training/reranker.py app/training/train_reranker.py examples/train_reranker.py tests/test_training_losses.py
git commit -m "feat: add staged reranker training"
```

### Task 5: Compute Offline Metrics And Enforce Release Gates

**Files:**
- Create: `app/training/metrics.py`
- Create: `app/training/release.py`
- Create: `tests/test_retrieval_metrics.py`
- Create: `tests/test_model_release.py`

**Interfaces:**
- Produces: `recall_at_k`, `ndcg_at_k`, `cross_language_recall_gap`, `cohens_kappa`
- Produces: `ReleaseReport`
- Produces: `evaluate_release(report) -> PromotionDecision`

- [ ] **Step 1: Write failing exact-metric tests**

Create tests with hand-computable rankings:

```python
def test_recall_and_ndcg_metrics():
    truth = {"q1": {"a", "b"}, "q2": {"c"}}
    ranked = {"q1": ["a", "x", "b"], "q2": ["x", "c"]}
    assert recall_at_k(truth, ranked, 2) == pytest.approx(0.75)
    assert ndcg_at_k(truth, ranked, 2) == pytest.approx(0.622038, rel=1e-5)


def test_release_gate_lists_every_failed_metric():
    report = ReleaseReport(
        model_bundle_version="candidate-v1",
        recall_at_100=0.84,
        ndcg_at_10=0.54,
        cross_language_recall_gap=0.06,
        cohens_kappa=0.74,
        faiss_p99_ms=50.0,
        reranker_p99_ms=101.0,
        real_log_evaluation=True,
        gpu_benchmark=True,
    )
    decision = evaluate_release(report)
    assert decision.approved is False
    assert len(decision.failures) == 6
```

- [ ] **Step 2: Run metric and gate tests to verify they fail**

Run:

```bash
uv run pytest tests/test_retrieval_metrics.py tests/test_model_release.py -q
```

Expected: FAIL because retrieval metrics and release models do not exist.

- [ ] **Step 3: Implement metric functions**

Create `app/training/metrics.py`:

```python
import math
from collections import Counter


def _validate_rankings(truth, ranked):
    if not truth:
        raise ValueError("truth must not be empty")
    missing = set(truth) - set(ranked)
    if missing:
        raise ValueError(f"missing rankings for queries: {sorted(missing)}")
    if any(not relevant for relevant in truth.values()):
        raise ValueError("each query must have at least one relevant item")


def recall_at_k(truth, ranked, k):
    _validate_rankings(truth, ranked)
    if k < 1:
        raise ValueError("k must be >= 1")
    values = [
        len(set(ranked[query_id][:k]) & relevant) / len(relevant)
        for query_id, relevant in truth.items()
    ]
    return sum(values) / len(values)


def ndcg_at_k(truth, ranked, k):
    _validate_rankings(truth, ranked)
    values = []
    for query_id, relevant in truth.items():
        gains = [1.0 if item_id in relevant else 0.0 for item_id in ranked[query_id][:k]]
        dcg = sum(gain / math.log2(position + 2) for position, gain in enumerate(gains))
        ideal_count = min(k, len(relevant))
        ideal = sum(1.0 / math.log2(position + 2) for position in range(ideal_count))
        values.append(dcg / ideal)
    return sum(values) / len(values)


def cross_language_recall_gap(language_recall):
    if len(language_recall) < 2:
        raise ValueError("at least two languages are required")
    values = list(language_recall.values())
    return max(values) - min(values)


def cohens_kappa(left, right):
    if len(left) != len(right) or not left:
        raise ValueError("annotation lists must be non-empty and equal length")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)
```

- [ ] **Step 4: Implement immutable release gates**

Create Pydantic models:

```python
class ReleaseReport(BaseModel):
    model_bundle_version: str
    recall_at_100: float
    ndcg_at_10: float
    cross_language_recall_gap: float
    cohens_kappa: float
    faiss_p99_ms: float
    reranker_p99_ms: float
    real_log_evaluation: bool
    gpu_benchmark: bool


class PromotionDecision(BaseModel):
    approved: bool
    failures: list[str]
```

`evaluate_release()` must add a separate failure for every missed threshold and must reject `real_log_evaluation=False` or `gpu_benchmark=False` even when numeric fixture metrics pass.

Implement the gate without rounding measured values:

```python
def evaluate_release(report: ReleaseReport) -> PromotionDecision:
    checks = [
        (report.recall_at_100 >= 0.85, "Recall@100 must be >= 0.85"),
        (report.ndcg_at_10 >= 0.55, "NDCG@10 must be >= 0.55"),
        (report.cross_language_recall_gap <= 0.05, "cross-language Recall Gap must be <= 0.05"),
        (report.cohens_kappa >= 0.75, "Cohen's Kappa must be >= 0.75"),
        (report.faiss_p99_ms < 50.0, "Faiss P99 must be < 50ms"),
        (report.reranker_p99_ms <= 100.0, "reranker P99 must be <= 100ms"),
        (report.real_log_evaluation, "real-log evaluation evidence is required"),
        (report.gpu_benchmark, "GPU benchmark evidence is required"),
    ]
    failures = [message for passed, message in checks if not passed]
    return PromotionDecision(approved=not failures, failures=failures)
```

- [ ] **Step 5: Run metric and gate tests**

Run:

```bash
uv run pytest tests/test_retrieval_metrics.py tests/test_model_release.py -q
```

Expected: all tests PASS without ML dependencies.

- [ ] **Step 6: Commit**

```bash
git add app/training/metrics.py app/training/release.py tests/test_retrieval_metrics.py tests/test_model_release.py
git commit -m "feat: gate retrieval model releases"
```

### Task 6: Package Immutable Bundles And Build Candidate Indices

**Files:**
- Modify: `app/training/schema.py`
- Modify: `app/training/release.py`
- Create: `examples/evaluate_retrieval.py`
- Create: `examples/package_model_bundle.py`
- Create: `examples/benchmark_reranker.py`
- Modify: `tests/test_model_release.py`

**Interfaces:**
- Produces: `ModelBundleManifest`
- Produces immutable Bundle directory and SHA-256 checksums
- Produces candidate Faiss/OpenSearch indices and an explicit promotion/rollback command

- [ ] **Step 1: Write a failing bundle-integrity test**

Create a temporary fake User/Query/Item/personalization/reranker artifact tree, package it, then assert:

```python
metadata = {
    "base_models": {
        "three_tower": "BAAI/bge-m3",
        "reranker": "BAAI/bge-reranker-v2-m3",
    },
    "embedding_dimension": 1024,
    "stage_configs": {"three_tower": "03-dpo/training_config.json"},
    "dataset_checksums": {"events": "abc123"},
    "git_commit": "0123456789abcdef",
    "created_at": "2026-07-16T00:00:00Z",
    "rollback_predecessor": "candidate-v0",
}
manifest = package_model_bundle(
    source_dir,
    bundle_dir,
    release_report,
    metadata,
)
assert manifest.model_bundle_version == release_report.model_bundle_version
assert set(manifest.components) == {"user_tower", "query_tower", "item_tower", "personalization_projection", "reranker"}
assert verify_model_bundle(bundle_dir) is True

(bundle_dir / "query_tower" / "model.safetensors").write_bytes(b"tampered")
assert verify_model_bundle(bundle_dir) is False
```

- [ ] **Step 2: Run the bundle test to verify it fails**

Run:

```bash
uv run pytest tests/test_model_release.py -k bundle -q
```

Expected: FAIL because Bundle packaging and verification do not exist.

- [ ] **Step 3: Add bundle schema and implementation**

Add this schema to `app/training/schema.py`:

```python
class ModelBundleManifest(BaseModel):
    model_bundle_version: str
    base_models: dict[str, str]
    embedding_dimension: int
    normalization: Literal["l2"] = "l2"
    distance_metric: Literal["inner_product"] = "inner_product"
    stage_configs: dict[str, str]
    dataset_checksums: dict[str, str]
    components: dict[str, str]
    component_checksums: dict[str, str]
    release_report_checksum: str
    git_commit: str
    created_at: datetime
    rollback_predecessor: str | None = None
```

Implement allowlisted packaging in `app/training/release.py`:

```python
ALLOWED_SUFFIXES = {".json", ".txt", ".model", ".safetensors"}
REQUIRED_COMPONENTS = {
    "user_tower",
    "query_tower",
    "item_tower",
    "personalization_projection",
    "reranker",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_model_bundle(source_dir, bundle_dir, release_report, metadata):
    decision = evaluate_release(release_report)
    if not decision.approved:
        raise ValueError("release report is not approved: " + "; ".join(decision.failures))
    if bundle_dir.exists():
        raise FileExistsError(bundle_dir)
    bundle_dir.mkdir(parents=True)
    component_paths = {}
    checksums = {}
    for component in sorted(REQUIRED_COMPONENTS):
        source = source_dir / component
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"invalid component directory: {component}")
        target = bundle_dir / component
        target.mkdir()
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"symlink is not allowed: {path}")
            if not path.is_file():
                continue
            if path.suffix not in ALLOWED_SUFFIXES:
                raise ValueError(f"file type is not allowlisted: {path.name}")
            relative = path.relative_to(source)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            checksums[str(destination.relative_to(bundle_dir))] = sha256_file(destination)
        component_paths[component] = component
    report_path = bundle_dir / "release-report.json"
    report_path.write_text(release_report.model_dump_json(indent=2), encoding="utf-8")
    manifest = ModelBundleManifest(
        model_bundle_version=release_report.model_bundle_version,
        components=component_paths,
        component_checksums=checksums,
        release_report_checksum=sha256_file(report_path),
        **metadata,
    )
    (bundle_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest


def verify_model_bundle(bundle_dir: Path) -> bool:
    manifest = ModelBundleManifest.model_validate_json(
        (bundle_dir / "manifest.json").read_text(encoding="utf-8")
    )
    for relative, expected in manifest.component_checksums.items():
        path = bundle_dir / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
            return False
    report_path = bundle_dir / "release-report.json"
    return report_path.is_file() and sha256_file(report_path) == manifest.release_report_checksum
```

The test passes explicit `metadata` for base models, dimensions, configs, checksums, git commit, creation time, and rollback predecessor. This keeps every manifest field concrete and auditable.

- [ ] **Step 4: Implement evaluation and benchmark CLIs**

- `evaluate_retrieval.py` reads ground-truth JSONL containing query language, ranking JSONL, annotation JSONL containing `judge_a`/`judge_b`, and the Faiss/Reranker benchmark reports. It calculates overall metrics, per-language Recall, Recall Gap, and Cohen's Kappa; it copies P99 only after matching model/index checksums, records `real_log_evaluation` only when an explicit signed dataset manifest is supplied, and writes `release-report.json`.
- `benchmark_reranker.py` performs warmup, measures batches of 100 candidates, reports P50/P95/P99, GPU type, CUDA/driver/Torch versions, FP16 flag, batch size, and model checksum. It refuses `gpu_benchmark=true` when CUDA is unavailable.
- `package_model_bundle.py` loads the report, calls `evaluate_release()`, exits `1` on any failure, and packages only an approved Bundle.

- [ ] **Step 5: Build candidate indices without switching current aliases**

After packaging, call the Faiss builder from the runtime plan with the new Item tower and write to `indices/faiss/{model_bundle_version}`. Generate OpenSearch bulk JSONL with Query-tower embeddings under versioned physical index names. Emit `promotion-plan.json` containing candidate paths/names, expected current predecessor, validation checksums, and rollback commands. Do not call the OpenSearch `_aliases` endpoint or update the runtime Faiss current pointer in this command.

- [ ] **Step 6: Implement guarded promotion and rollback**

Create `examples/promote_model_bundle.py`. It requires `--bundle`, `--promotion-plan`, `--opensearch-url`, `--faiss-current-link`, and an explicit `--apply` flag. Without `--apply`, print the validated actions and exit without mutation.

With `--apply`, execute this order:

1. Verify the Bundle, release report, candidate index checksums, physical OpenSearch indices, and expected predecessor.
2. Atomically update both OpenSearch aliases in one `POST /_aliases` request.
3. Create a temporary Faiss symlink beside `--faiss-current-link` and switch it with `os.replace()`.
4. If the Faiss switch fails, issue one compensating `_aliases` request restoring the predecessor and exit nonzero.
5. Write `active-release.json` through a temporary file and `os.replace()`.
6. Keep predecessor Bundle and indices untouched for rollback.

The `--rollback` mode reads `active-release.json`, validates the predecessor checksums, atomically restores the OpenSearch aliases and Faiss link, and records the rollback event. Tests use a fake OpenSearch transport and temporary symlinks to prove dry-run immutability, successful switch, compensation on Faiss failure, and rollback.

- [ ] **Step 7: Run release tests and commit**

Run:

```bash
uv run pytest tests/test_model_release.py tests/test_retrieval_metrics.py -q
```

Expected: all release tests PASS.

```bash
git add app/training examples/evaluate_retrieval.py examples/package_model_bundle.py examples/promote_model_bundle.py examples/benchmark_reranker.py tests/test_model_release.py
git commit -m "feat: package immutable retrieval model bundles"
```

### Task 7: Add Smoke Workflow, Documentation, And Full Verification

**Files:**
- Create: `examples/prepare_retrieval_dataset.py`
- Modify: `README.md`
- Modify: `tests/test_training_dataset.py`

**Interfaces:**
- Produces a CPU-only data/metric/release-gate smoke workflow
- Documents external GPU and real-log acceptance separately

- [ ] **Step 1: Write a failing CPU-only preparation smoke test**

Add to `tests/test_training_dataset.py`:

```python
from examples.prepare_retrieval_dataset import prepare_dataset


def test_prepare_dataset_writes_auditable_outputs(tmp_path):
    output = tmp_path / "prepared"
    manifest = prepare_dataset(
        events_path=FIXTURES / "retrieval_events.jsonl",
        items_path=FIXTURES / "retrieval_items.jsonl",
        output_dir=output,
        train_ratio=0.6,
        validation_ratio=0.2,
        seed=42,
    )

    assert set(path.name for path in output.iterdir()) == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "examples.jsonl",
        "dataset-manifest.json",
    }
    assert manifest.seed == 42
    assert manifest.source_checksums.keys() == {"events", "items"}
    assert manifest.real_log_dataset is False


def test_fixture_release_evidence_cannot_pass_production_gate():
    report = ReleaseReport(
        model_bundle_version="fixture-v1",
        recall_at_100=1.0,
        ndcg_at_10=1.0,
        cross_language_recall_gap=0.0,
        cohens_kappa=1.0,
        faiss_p99_ms=1.0,
        reranker_p99_ms=1.0,
        real_log_evaluation=False,
        gpu_benchmark=False,
    )
    decision = evaluate_release(report)
    assert decision.approved is False
    assert decision.failures == [
        "real-log evaluation evidence is required",
        "GPU benchmark evidence is required",
    ]
```

Add the required `ReleaseReport` and `evaluate_release` imports. No smoke test may call `from_pretrained` or access the network.

- [ ] **Step 2: Run the smoke test to verify it fails**

Run:

```bash
uv run pytest tests/test_training_dataset.py -k "prepare_dataset or fixture_release" -q
```

Expected: FAIL because `examples.prepare_retrieval_dataset` does not exist.

- [ ] **Step 3: Implement dataset preparation CLI**

Implement `prepare_dataset()` with the exact signature used by the test. It loads and validates fixtures through `load_events()`/`load_items()`, calls `chronological_split()` and `build_examples()`, refuses an existing output directory, and writes `train.jsonl`, `validation.jsonl`, `test.jsonl`, and `examples.jsonl` using `model_dump_json()`.

Add `DatasetManifest` to `app/training/schema.py` with source SHA-256 checksums, split timestamp bounds, row counts, false-negative removal count, language/market distributions, seed, and `real_log_dataset: bool`. The CLI writes this model to `dataset-manifest.json`; fixture paths force `real_log_dataset=False`.

The CLI requires events, items, output directory, train ratio, validation ratio, and seed. Its parser calls `prepare_dataset()` and prints the manifest path only.

- [ ] **Step 4: Document exact workflows and non-claims**

README must separate:

```bash
uv sync --group ml
uv run python examples/prepare_retrieval_dataset.py --events data/interactions.jsonl --items data/items.jsonl --output output/training/dataset-v1 --train-ratio 0.8 --validation-ratio 0.1 --seed 42
uv run --group ml python examples/train_three_tower.py --stage cpt --base-model BAAI/bge-m3 --events data/interactions.jsonl --items data/items.jsonl --run-dir output/training/run-v1 --seed 42 --epochs 1 --batch-size 8 --gradient-accumulation-steps 4
uv run --group ml python examples/train_three_tower.py --stage sft --base-model BAAI/bge-m3 --events data/interactions.jsonl --items data/items.jsonl --run-dir output/training/run-v1 --seed 42 --epochs 1 --batch-size 8 --gradient-accumulation-steps 4
uv run --group ml python examples/train_three_tower.py --stage dpo --base-model BAAI/bge-m3 --events data/interactions.jsonl --items data/items.jsonl --run-dir output/training/run-v1 --seed 42 --epochs 1 --batch-size 8 --gradient-accumulation-steps 4
uv run --group ml python examples/train_reranker.py --base-model BAAI/bge-reranker-v2-m3 --dataset output/training/dataset-v1 --run-dir output/training/reranker-v1 --seed 42 --epochs 1 --batch-size 8
uv run python examples/evaluate_retrieval.py --truth data/retrieval-truth.jsonl --rankings output/training/rankings.jsonl --annotations data/retrieval-annotations.jsonl --faiss-benchmark output/training/faiss-benchmark.json --reranker-benchmark output/training/reranker-benchmark.json --dataset-manifest data/signed-dataset-manifest.json --output output/training/release-report.json
uv run --group ml python examples/benchmark_reranker.py --model output/training/reranker-v1/04-dpo --queries data/reranker-benchmark.jsonl --batch-size 100 --warmup 20 --iterations 200 --output output/training/reranker-benchmark.json
uv run python examples/package_model_bundle.py --source output/training/run-v1 --reranker output/training/reranker-v1/04-dpo --release-report output/training/release-report.json --output output/bundles/omnimatch-retrieval-v1
```

State explicitly that fixture smoke results do not satisfy production gates and that only signed real-log evaluation plus a CUDA benchmark can approve a Bundle.

- [ ] **Step 5: Run verification**

Run lightweight tests:

```bash
uv run pytest tests/test_training_dataset.py tests/test_retrieval_metrics.py tests/test_model_release.py -q
```

Expected: all lightweight tests PASS without downloading models.

Run ML tensor tests:

```bash
uv run --group ml pytest tests/test_training_losses.py -q
```

Expected: all tensor-only tests PASS without downloading models.

Run the full backend suite:

```bash
uv run pytest -q
```

Expected: all non-ML backend tests PASS with zero failures.

- [ ] **Step 6: Commit**

```bash
git add examples/prepare_retrieval_dataset.py README.md tests/test_training_dataset.py
git commit -m "docs: add retrieval training and release workflow"
```

## Phase Acceptance Checklist

- [ ] Training data rejects missing items, duplicate catalog IDs, and invalid chronology.
- [ ] Same-group SKUs and cross-language duplicates cannot become negatives.
- [ ] CPT/SFT/DPO learning rates and ordering are enforced.
- [ ] Dynamic temperatures, Hard Negative weight `2.0`, and `L_align` are tested.
- [ ] Reranker training follows pointwise, pairwise, listwise, then market-aware DPO.
- [ ] Round-two mining uses strict `score > 0.3 and not purchased`.
- [ ] Release metrics and all six numeric thresholds are independently checked.
- [ ] Fixture or CPU results cannot claim real-log or GPU release evidence.
- [ ] Bundle components and reports are immutable and checksum-verified.
- [ ] Candidate index construction cannot silently switch production aliases.
- [ ] Real training and benchmark reports remain external acceptance evidence, not fabricated repository state.
