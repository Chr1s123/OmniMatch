import pytest

from app.config import OmniMatchSettings
from app.eval.cases import EvalCase
from app.eval.runner import run_eval_cases


@pytest.mark.asyncio
async def test_eval_runner_returns_scores_for_submission_profile(tmp_path):
    settings = OmniMatchSettings(
        profile="submission",
        llm_provider="placeholder",
        llm_model="placeholder-llm",
        product_provider="placeholder",
        web_search_provider="placeholder",
        shipping_provider="placeholder",
        memory_provider="placeholder",
        eval_provider="heuristic",
    )
    cases = [
        EvalCase(
            id="budget_no_plastic",
            query="旅行三件套，预算300，不要塑料",
            required_terms=["旅行", "塑料"],
            forbidden_terms=["无法推荐"],
        )
    ]

    results = await run_eval_cases(cases, settings=settings, output_dir=tmp_path)

    assert results[0].case_id == "budget_no_plastic"
    assert 0 <= results[0].score <= 1
    assert results[0].trace_dir.exists()
