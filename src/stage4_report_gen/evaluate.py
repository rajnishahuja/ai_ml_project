"""
Report quality evaluation for Stage 4.

Two evaluators under the new design:

  1. ROUGE — text overlap between Mistral-generated `contract_summary` and
     a gold reference summary, when one is available. Wraps `rouge-score`
     (soft dependency).

  2. Structural completeness — schema-level checks on the assembled report:
     metadata block present, three risk-table buckets exist, conclusion
     populated, disclaimer non-empty, score in valid range.

`recommendation_coverage` is removed — the new design has no per-clause
recommendation lookup table to measure coverage of.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema-agnostic dict access
# ---------------------------------------------------------------------------

def _to_dict(report: Any) -> dict:
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report)


# ---------------------------------------------------------------------------
# ROUGE — text quality of generated summaries
# ---------------------------------------------------------------------------

def evaluate_summaries(
    generated: list[str],
    reference: list[str],
) -> dict[str, float]:
    """Compute rouge1 / rouge2 / rougeL F-measures over a batch.

    Args:
        generated: Model-generated summary strings (one per contract).
        reference: Reference (gold) summary strings.

    Returns:
        Dict with rouge1, rouge2, rougeL (all in [0, 1]). Returns zeros for
        an empty input or when `rouge-score` is missing.
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

def evaluate_report_completeness(report: Any) -> dict[str, bool]:
    """Schema-level structural checks.

    Validates the new Stage 4 schema. Returns a dict mapping check name
    to True/False — all True is a healthy report.
    """
    r = _to_dict(report)
    checks: dict[str, bool] = {}

    checks["has_document_id"]    = bool(r.get("document_id"))
    checks["has_metadata"]       = isinstance(r.get("metadata"), dict) and bool(r.get("metadata"))
    checks["has_summary"]        = bool(r.get("contract_summary"))
    checks["has_summary_header"] = bool(r.get("summary_header"))

    score = r.get("overall_risk_score", -1)
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        score_f = -1.0
    checks["score_in_range"] = 0.0 <= score_f <= 10.0

    risk_tables = r.get("risk_tables", {}) or {}
    checks["has_three_risk_tiers"] = (
        isinstance(risk_tables, dict)
        and set(risk_tables.keys()) >= {"HIGH", "MEDIUM", "LOW"}
    )
    checks["risk_rows_well_formed"] = all(
        isinstance(row, dict)
        and set(row.keys()) >= {"clause_type", "clause_text", "reasoning", "confidence"}
        for tier_rows in risk_tables.values()
        for row in tier_rows
    ) if risk_tables else True

    conclusion = r.get("conclusion") or {}
    checks["has_conclusion"] = (
        isinstance(conclusion, dict)
        and bool(conclusion.get("overall_assessment"))
    )

    checks["has_disclaimer"] = bool(r.get("disclaimer"))

    total = int(r.get("total_clauses", 0) or 0)
    rt_total = sum(len(v) for v in risk_tables.values()) if risk_tables else 0
    checks["totals_match"] = total == rt_total

    return checks
