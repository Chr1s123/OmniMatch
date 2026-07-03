from __future__ import annotations

from app.schemas import CandidateScore, ProductCandidate, ScoredProduct, ShoppingIntent


def score_candidates(
    intent: ShoppingIntent,
    candidates: list[ProductCandidate],
) -> list[ScoredProduct]:
    scored = [_score_one(intent, candidate) for candidate in candidates]
    return sorted(scored, key=lambda item: item.score.total, reverse=True)


def _score_one(intent: ShoppingIntent, candidate: ProductCandidate) -> ScoredProduct:
    reasons: list[str] = []
    constraint_score = 40.0
    if intent.budget is not None and candidate.total_landed_cost > intent.budget:
        constraint_score -= 25.0
        reasons.append("over budget")

    material = (candidate.material or candidate.title).lower()
    for forbidden in intent.negative_constraints:
        if forbidden.lower() in material or _material_matches_zh(forbidden, material):
            constraint_score -= 35.0
            reasons.append(f"negative constraint matched: {forbidden}")

    evidence_score = min(20.0, 8.0 * len(candidate.evidence))
    price_score = 20.0
    if intent.budget:
        price_score = max(0.0, 20.0 * (1 - candidate.total_landed_cost / (intent.budget * 1.5)))
    preference_score = 10.0 if any(pref.lower() in material for pref in intent.preferences) else 4.0
    risk_penalty = 10.0 if not candidate.url or not candidate.evidence else 0.0
    total = max(0.0, constraint_score + evidence_score + price_score + preference_score - risk_penalty)

    return ScoredProduct(
        candidate=candidate,
        score=CandidateScore(
            total=round(total, 2),
            constraint_score=round(constraint_score, 2),
            evidence_score=round(evidence_score, 2),
            price_score=round(price_score, 2),
            preference_score=round(preference_score, 2),
            risk_penalty=round(risk_penalty, 2),
            total_landed_cost=candidate.total_landed_cost,
            rejection_reasons=reasons,
        ),
    )


def _material_matches_zh(forbidden: str, material: str) -> bool:
    if forbidden == "塑料":
        return "plastic" in material
    return False
