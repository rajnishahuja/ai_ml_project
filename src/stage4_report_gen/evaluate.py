"""
Report quality evaluation for Stage 4.

Three evaluators:

  1. ROUGE — text overlap between generated explanations and gold references.
     Wraps the `rouge-score` package (soft dependency). Returns rouge1 / rouge2
     / rougeL F-measures.

  2. Structural completeness — schema-level checks on the assembled report
     (required fields populated, score in valid range, recommendations
     present for HIGH-risk clauses, etc.).

  3. Recommendation coverage — how many HIGH-risk clauses got a curated
     (non-fallback) recommendation versus the generic default.
"""

from __future__ import annotations

import logging
from typing import Any

from src.common.schema import RiskReport
from src.stage4_report_gen.recommender import DEFAULT_RECOMMENDATIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema-agnostic dict access
# ---------------------------------------------------------------------------

def _to_dict(report: RiskReport | dict) -> dict:
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report)


# ---------------------------------------------------------------------------
# ROUGE — text quality of generated explanations
# ---------------------------------------------------------------------------

def evaluate_explanations(
    generated: list[str],
    reference: list[str],
) -> dict[str, float]:
    """Compute rouge1 / rouge2 / rougeL F-measures.

    Args:
        generated: Model-generated explanation strings.
        reference: Reference (gold) explanation strings. Must be the same
            length as `generated`.

    Returns:
        Dict with keys `rouge1`, `rouge2`, `rougeL`, all in [0, 1]. Returns
        zeros for an empty input or when the rouge-score package is missing
        (with a warning log line — the rest of Stage 4 evaluation still runs).
    """
    if len(generated) != len(reference):
        raise ValueError(
            f"Length mismatch: {len(generated)} generated vs "
            f"{len(reference)} reference."
        )
    if not generated:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    try:
        from rouge_score import rouge_scorer   # type: ignore
    except ImportError:
        logger.warning(
            "rouge-score not installed — skipping ROUGE eval. "
            "Install with `pip install rouge-score`."
        )
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    sums = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for gen, ref in zip(generated, reference):
        scores = scorer.score(ref, gen)
        for key in sums:
            sums[key] += scores[key].fmeasure

    n = len(generated)
    return {key: round(value / n, 4) for key, value in sums.items()}


# ---------------------------------------------------------------------------
# Structural completeness
# ---------------------------------------------------------------------------

def evaluate_report_completeness(report: RiskReport | dict) -> dict[str, bool]:
    """Schema-level structural checks. Pure dict inspection.

    Checks (each maps to True/False in the result):
      - has_summary               : non-empty `summary` string
      - has_document_id           : non-empty `document_id`
      - score_in_range            : `overall_risk_score` is in [0, 10]
      - high_risk_have_recs       : every high-risk entry has a non-empty
                                    `recommendation`
      - medium_risk_have_recs     : every medium-risk entry has a non-empty
                                    `recommendation`
      - high_risk_have_explanations : every high-risk entry has a non-empty
                                      `explanation`
      - missing_protections_listed : `missing_protections` is a list (may be empty)
      - total_matches_buckets       : `total_clauses` ≥ |high| + |medium|
                                      (LOW bucket is summarized, not enumerated,
                                      so equality is not expected)
    """
    r = _to_dict(report)
    checks: dict[str, bool] = {}

    checks["has_summary"] = bool(r.get("summary"))
    checks["has_document_id"] = bool(r.get("document_id"))

    score = r.get("overall_risk_score", -1)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = -1.0
    checks["score_in_range"] = 0.0 <= score_f <= 10.0

    high = r.get("high_risk", []) or []
    medium = r.get("medium_risk", []) or []

    def _all_have(entries: list[dict], key: str) -> bool:
        return all(bool(e.get(key)) for e in entries) if entries else True

    checks["high_risk_have_recs"] = _all_have(high, "recommendation")
    checks["medium_risk_have_recs"] = _all_have(medium, "recommendation")
    checks["high_risk_have_explanations"] = _all_have(high, "explanation")
    checks["missing_protections_listed"] = isinstance(
        r.get("missing_protections"), list,
    )

    total = int(r.get("total_clauses", 0) or 0)
    checks["total_matches_buckets"] = total >= len(high) + len(medium)

    return checks


# ---------------------------------------------------------------------------
# Recommendation coverage
# ---------------------------------------------------------------------------

def recommendation_coverage(report: RiskReport | dict) -> dict[str, Any]:
    """How many HIGH-risk clauses got a curated (non-default) recommendation.

    Useful for tracking the lookup-table's coverage on real reports.

    Returns dict:
      - total_high           : int
      - curated              : int   (recommendation is not in DEFAULT_RECOMMENDATIONS)
      - default              : int   (recommendation matches DEFAULT_RECOMMENDATIONS["HIGH"])
      - coverage             : float in [0, 1] = curated / total_high
    """
    r = _to_dict(report)
    high = r.get("high_risk", []) or []
    total = len(high)
    if total == 0:
        return {"total_high": 0, "curated": 0, "default": 0, "coverage": 0.0}

    default_text = DEFAULT_RECOMMENDATIONS["HIGH"]
    default_count = sum(1 for e in high if e.get("recommendation") == default_text)
    curated = total - default_count

    return {
        "total_high": total,
        "curated": curated,
        "default": default_count,
        "coverage": round(curated / total, 3),
    }
