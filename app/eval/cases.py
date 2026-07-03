from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class EvalCase(BaseModel):
    id: str
    query: str
    required_terms: list[str] = []
    forbidden_terms: list[str] = []


class EvalResult(BaseModel):
    case_id: str
    score: float
    passed: bool
    notes: list[str]
    trace_dir: Path
