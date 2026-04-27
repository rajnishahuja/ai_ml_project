"""
Report assembly for Stage 4.

Combines aggregation, explanations, and recommendations into a final
RiskReport. This module is the orchestration layer — it does not own any
ML logic itself; it composes:

    aggregator  →  group / score / detect missing protections
    explainer   →  plain-language explanations (FLAN-T5, optional)
    recommender →  curated remediation lookup table

Two output paths:
  - `build_report(...) -> RiskReport`         (dataclass — `src.common.schema`)
  - `build_report_dict(...) -> dict`          (plain dict — used by the
                                                LangGraph `nodes.py` and any
                                                pydantic consumer)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.common.schema import (
    ReportClause,
    ReportMetadata,
    RiskAssessedClause,
    RiskReport,
)
from src.common.utils import load_config

from src.stage4_report_gen.aggregator import (
    EXPECTED_PROTECTIONS,
    compute_contract_risk_score,
    find_missing_protections,
    get_top_risks,
    group_by_risk_level,
    low_risk_summary,
)
from src.stage4_report_gen.explainer import (
    ExplanationModel,
    generate_explanation,
)
from src.stage4_report_gen.recommender import get_recommendation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema-agnostic attribute reader (mirrors aggregator._get / explainer._get)
# ---------------------------------------------------------------------------

def _get(clause: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if hasattr(clause, name):
            value = getattr(clause, name)
            if value is not None:
                return value
        if isinstance(clause, dict) and name in clause:
            value = clause[name]
            if value is not None:
                return value
    return default


# ---------------------------------------------------------------------------
# Summary line
# ---------------------------------------------------------------------------

def _build_summary(
    total: int, n_high: int, n_medium: int, n_low: int, score: float,
) -> str:
    """One-paragraph natural-language summary for the report header."""
    return (
        f"This contract contains {total} assessed clause(s) with an overall "
        f"risk score of {score:.1f}/10. "
        f"{n_high} high-risk, {n_medium} medium-risk, and {n_low} low-risk "
        f"clause(s) were identified."
    )


# ---------------------------------------------------------------------------
# Per-clause report entry
# ---------------------------------------------------------------------------

def _build_report_entry(
    clause: Any,
    explanation_model: Optional[ExplanationModel],
    max_explanation_length: int,
) -> dict:
    """Convert one risk-assessed clause into a flat report-row dict.

    Dict-valued so it serializes cleanly into either:
      - `src.common.schema.ReportClause` (dataclass)
      - `app.schemas.domain.RiskReportRecommendation` (pydantic, has extra
        page_no / content_label fields)
    """
    clause_type = str(_get(clause, "clause_type", default="") or "")
    risk_level = str(_get(clause, "risk_level", default="LOW") or "LOW").upper()

    explanation = generate_explanation(
        clause,
        model=explanation_model,
        max_length=max_explanation_length,
    )
    recommendation = get_recommendation(clause_type, risk_level)

    return {
        "clause_id": str(_get(clause, "clause_id", default="") or ""),
        "clause_type": clause_type,
        "risk_level": risk_level,
        "explanation": explanation,
        "recommendation": recommendation,
        # Pydantic schema extras — included if present, ignored by the dataclass
        "page_no": _get(clause, "page_no"),
        "content_label": _get(clause, "content_label"),
    }


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_report_dict(
    clauses: list[Any],
    document_id: str,
    *,
    explanation_model: Optional[ExplanationModel] = None,
    max_explanation_length: int = 200,
    expected_protections: tuple[str, ...] = EXPECTED_PROTECTIONS,
    top_n: int = 5,
    models_used: Optional[dict[str, str]] = None,
) -> dict:
    """Build the report as a plain dict (no dataclass dependency).

    Use this from the LangGraph node (`stage4_report_gen/nodes.py`) and any
    pydantic-based consumer. The returned dict matches the JSON shape
    documented in ARCHITECTURE.md §"Stage 4 Output (Final Report)".

    Args:
        clauses: Stage 3 output (list of risk-assessed clauses, any schema).
        document_id: Source contract identifier.
        explanation_model: Optional loaded FLAN-T5 wrapper. If None, the
            existing `risk_reason` from Stage 3 is used verbatim.
        max_explanation_length: Max tokens for FLAN-T5 generation.
        expected_protections: Clause types treated as standard protections
            for the gap analysis.
        top_n: Cap on the number of clauses included per risk bucket. The
            Stage 4 report is meant to be human-readable, not exhaustive —
            past ~5–10 entries per bucket the report becomes noise.
        models_used: Optional model-version metadata for the report footer.

    Returns:
        Dict with keys: document_id, summary, high_risk, medium_risk,
        low_risk_summary, missing_protections, overall_risk_score,
        total_clauses, metadata.
    """
    if not clauses:
        logger.warning("build_report_dict: no clauses provided for %s.", document_id)

    grouped = group_by_risk_level(clauses)
    high, medium, low = grouped["HIGH"], grouped["MEDIUM"], grouped["LOW"]

    # Cap each non-LOW bucket at top_n by severity+confidence so the report
    # surfaces the most actionable items, not the full list.
    high_top = get_top_risks(high, n=top_n)
    medium_top = get_top_risks(medium, n=top_n)

    high_entries = [
        _build_report_entry(c, explanation_model, max_explanation_length)
        for c in high_top
    ]
    medium_entries = [
        _build_report_entry(c, explanation_model, max_explanation_length)
        for c in medium_top
    ]

    score = compute_contract_risk_score(clauses)
    missing = find_missing_protections(clauses, expected_types=expected_protections)
    summary = _build_summary(
        total=len(list(clauses)),
        n_high=len(high),
        n_medium=len(medium),
        n_low=len(low),
        score=score,
    )

    return {
        "document_id": document_id,
        "summary": summary,
        "high_risk": high_entries,
        "medium_risk": medium_entries,
        "low_risk_summary": low_risk_summary(low),
        "missing_protections": missing,
        "overall_risk_score": score,
        "total_clauses": len(list(clauses)),
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models_used": models_used or {},
        },
    }


def build_report(
    clauses: list[RiskAssessedClause],
    document_id: str,
    config_path: str = "configs/stage4_config.yaml",
    explanation_model: Optional[ExplanationModel] = None,
) -> RiskReport:
    """Build a typed `RiskReport` (dataclass version).

    Wraps `build_report_dict` and casts each report row into a `ReportClause`
    dataclass. Used by stand-alone scripts and unit tests that prefer the
    typed schema; the LangGraph node uses `build_report_dict` directly.

    Args:
        clauses: Stage 3 output.
        document_id: Source contract identifier.
        config_path: Path to `stage4_config.yaml`. Read for
            `max_explanation_length`. Missing file → defaults are used.
        explanation_model: Optional loaded FLAN-T5 wrapper.

    Returns:
        Fully-populated `RiskReport`.
    """
    # Optional config — fall back gracefully so unit tests don't need the file.
    max_explanation_length = 200
    try:
        cfg = load_config(config_path)
        max_explanation_length = int(
            cfg.get("max_explanation_length", max_explanation_length)
        )
    except FileNotFoundError:
        logger.info("Config %s not found — using defaults.", config_path)

    raw = build_report_dict(
        clauses,
        document_id,
        explanation_model=explanation_model,
        max_explanation_length=max_explanation_length,
    )

    def _to_report_clause(d: dict) -> ReportClause:
        return ReportClause(
            clause_id=d["clause_id"],
            clause_type=d["clause_type"],
            risk_level=d["risk_level"],
            explanation=d["explanation"],
            recommendation=d["recommendation"],
        )

    return RiskReport(
        document_id=raw["document_id"],
        summary=raw["summary"],
        high_risk=[_to_report_clause(d) for d in raw["high_risk"]],
        medium_risk=[_to_report_clause(d) for d in raw["medium_risk"]],
        low_risk_summary=raw["low_risk_summary"],
        missing_protections=raw["missing_protections"],
        overall_risk_score=raw["overall_risk_score"],
        total_clauses=raw["total_clauses"],
        metadata=ReportMetadata(
            generated_at=raw["metadata"]["generated_at"],
            models_used=raw["metadata"]["models_used"],
        ),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_report(report: RiskReport | dict, output_path: str) -> None:
    """Serialize and save a report to JSON.

    Accepts either the dataclass (`RiskReport`) or the dict form returned
    by `build_report_dict`. Creates parent directories as needed.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = report.to_dict() if hasattr(report, "to_dict") else report
    with open(output, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Saved report to %s", output)
