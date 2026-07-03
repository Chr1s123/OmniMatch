from app.ranking.scorer import score_candidates
from app.schemas import ProductCandidate, ShoppingIntent


def test_ranking_penalizes_forbidden_material():
    intent = ShoppingIntent(
        original_query="旅行三件套，预算300，不要塑料",
        category="旅行三件套",
        budget=300,
        preferences=["耐用"],
        negative_constraints=["塑料"],
    )
    candidates = [
        ProductCandidate(
            id="plastic",
            platform="A",
            title="Plastic travel set",
            price=100,
            currency="CNY",
            url="https://example.com/plastic",
            material="plastic",
            evidence=["catalog"],
        ),
        ProductCandidate(
            id="canvas",
            platform="B",
            title="Canvas travel set",
            price=190,
            currency="CNY",
            url="https://example.com/canvas",
            material="canvas",
            evidence=["catalog", "guide"],
        ),
    ]

    scored = score_candidates(intent, candidates)

    assert scored[0].candidate.id == "canvas"
    assert any("negative constraint" in reason for reason in scored[1].score.rejection_reasons)


def test_ranking_marks_over_budget_candidate():
    intent = ShoppingIntent(
        original_query="预算100旅行三件套",
        category="旅行三件套",
        budget=100,
        preferences=[],
        negative_constraints=[],
    )
    candidate = ProductCandidate(
        id="expensive",
        platform="A",
        title="Expensive set",
        price=180,
        currency="CNY",
        shipping=30,
        tax=0,
        url="https://example.com/expensive",
        evidence=["catalog"],
    )

    scored = score_candidates(intent, [candidate])

    assert scored[0].score.total_landed_cost == 210
    assert "over budget" in scored[0].score.rejection_reasons
