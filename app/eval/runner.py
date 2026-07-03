from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.agent.main_agent import CompetitionAgentLoop
from app.api.monitor import EventCollector
from app.config import OmniMatchSettings
from app.eval.cases import EvalCase, EvalResult
from app.providers.registry import ProviderRegistry


async def run_eval_cases(
    cases: list[EvalCase],
    settings: OmniMatchSettings,
    output_dir: Path,
) -> list[EvalResult]:
    results: list[EvalResult] = []
    providers = ProviderRegistry.from_settings(settings)
    for case in cases:
        thread_id = f"eval_{case.id}_{uuid4().hex[:6]}"
        trace_dir = output_dir / thread_id
        monitor = EventCollector(thread_id=thread_id)
        loop = CompetitionAgentLoop(
            thread_id=thread_id,
            session_dir=trace_dir,
            settings=settings,
            providers=providers,
            monitor=monitor,
        )
        summary = await loop.run(case.query)
        text = " ".join([summary.message, *[product.title for product in summary.products]])
        notes: list[str] = []
        required_hits = sum(1 for term in case.required_terms if term in text)
        forbidden_hits = [term for term in case.forbidden_terms if term.lower() in text.lower()]
        if forbidden_hits:
            notes.append(f"forbidden terms present: {', '.join(forbidden_hits)}")
        required_score = required_hits / max(1, len(case.required_terms))
        score = max(0.0, required_score - 0.5 * len(forbidden_hits))
        results.append(
            EvalResult(
                case_id=case.id,
                score=round(score, 2),
                passed=score >= 0.7,
                notes=notes,
                trace_dir=trace_dir,
            )
        )
    return results
