"""
Stage 4 report assembly.

Single public entry point: `assemble_report_dict(...)`. Combines:

  - The contract metadata block (`extract_metadata_block` from Stage 3).
  - The grouped risk-assessed clauses (from `aggregator.group_by_risk_level`).
  - The contract-level risk score.
  - The Mistral-generated summary + conclusion (already a dict).

Returns a flat dict matching the new Stage 4 schema (see plan):
metadata header, contract_summary, three risk_tables (all clauses, no top-N
cap), conclusion, fixed disclaimer, generation metadata.

The DOCX/PDF rendering and file persistence happen in
`docx_renderer.py` / `pdf_converter.py` and are wired by the LangGraph
`node_report_generation` — this module is the in-memory aggregation step.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DISCLAIMER_TEXT = (
    "This automated risk analysis is intended as a decision-support tool "
    "and does not constitute legal advice. All findings should be reviewed "
    "and confirmed by licensed legal counsel before relying on this "
    "document for any contract decision, negotiation, or signing."
)


# Order in which metadata fields appear in the report header. Matches the
# canonical order in `extract_metadata_block`.
_METADATA_ORDER = (
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
)


# ---------------------------------------------------------------------------
# Schema-agnostic attribute reader
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


def _get_reasoning(clause: Any) -> str:
    """Pull the Stage-3 reasoning text. Pydantic uses `risk_reason`,
    dataclass uses `risk_explanation` — try both."""
    for field in ("risk_reason", "risk_explanation"):
        value = _get(clause, field, default="")
        if value:
            return str(value)
    return ""


# ---------------------------------------------------------------------------
# Per-clause table row builder
# ---------------------------------------------------------------------------

def _build_table_row(clause: Any) -> dict[str, Any]:
    """One row in the HIGH / MEDIUM / LOW risk table.

    New schema (per plan): four columns — clause type, full clause text,
    Stage 3 reasoning, confidence score. NO recommendation, page_no, or
    content_label.
    """
    confidence_raw = _get(clause, "confidence", default=None)
    try:
        confidence = round(float(confidence_raw), 2) if confidence_raw is not None else None
    except (TypeError, ValueError):
        confidence = None

    return {
        "clause_type":  str(_get(clause, "clause_type", default="") or ""),
        "clause_text":  str(_get(clause, "clause_text", default="") or ""),
        "reasoning":    _get_reasoning(clause),
        "confidence":   confidence,
    }


# ---------------------------------------------------------------------------
# Header summary line
# ---------------------------------------------------------------------------

def _build_summary_header(
    total: int, n_high: int, n_medium: int, n_low: int, score: float,
) -> str:
    """Single-line header summary (separate from the Mistral-generated
    `contract_summary` paragraph). Used in the JSON response and as a
    backup if Mistral output is empty."""
    return (
        f"This contract contains {total} assessed clause(s) with an overall "
        f"risk score of {score:.1f}/10. "
        f"{n_high} high-risk, {n_medium} medium-risk, and {n_low} low-risk "
        f"clause(s) were identified."
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assemble_report_dict(
    document_id: str,
    metadata: dict[str, str],
    grouped: dict[str, Iterable[Any]],
    overall_risk_score: float,
    llm_output: dict[str, Any] | None = None,
    *,
    models_used: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the report payload (no DOCX/PDF — that's the renderer's job).

    Args:
        document_id: Source contract identifier.
        metadata: Output of `extract_metadata_block(...)`.
        grouped: Output of `group_by_risk_level(...)` — dict with keys
            HIGH / MEDIUM / LOW, each a list of risk-assessed clauses.
        overall_risk_score: Output of `compute_contract_risk_score(...)`.
        llm_output: Output of the Stage 4 Mistral call. Expected keys:
            `contract_summary`, `overall_assessment`, `high_priority_actions`,
            `medium_priority_actions`. May be None or partial — defaults are
            substituted so the report shape is stable.
        models_used: Optional model-version metadata for the report footer.

    Returns:
        The report dict. Caller (LangGraph node) attaches `docx_path` and
        `pdf_path` after rendering.
    """
    high = list(grouped.get("HIGH", []))
    medium = list(grouped.get("MEDIUM", []))
    low = list(grouped.get("LOW", []))
    total = len(high) + len(medium) + len(low)

    # Normalize metadata to canonical key order, with dashes for missing.
    meta_block = {key: metadata.get(key, "—") for key in _METADATA_ORDER}

    llm = llm_output or {}
    contract_summary = str(llm.get("contract_summary") or "").strip() or (
        _build_summary_header(total, len(high), len(medium), len(low), overall_risk_score)
    )
    overall_assessment = str(llm.get("overall_assessment") or "").strip() or (
        "No high-level assessment was generated."
    )
    high_actions = list(llm.get("high_priority_actions") or [])
    medium_actions = list(llm.get("medium_priority_actions") or [])

    return {
        "document_id":         document_id,
        "metadata":            meta_block,
        "contract_summary":    contract_summary,
        "overall_risk_score":  overall_risk_score,
        "total_clauses":       total,
        "summary_header":      _build_summary_header(
            total, len(high), len(medium), len(low), overall_risk_score,
        ),
        "risk_tables": {
            "HIGH":   [_build_table_row(c) for c in high],
            "MEDIUM": [_build_table_row(c) for c in medium],
            "LOW":    [_build_table_row(c) for c in low],
        },
        "conclusion": {
            "overall_assessment":      overall_assessment,
            "high_priority_actions":   high_actions,
            "medium_priority_actions": medium_actions,
        },
        "disclaimer":   DISCLAIMER_TEXT,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models_used":  models_used or {},
        # docx_path / pdf_path attached by the LangGraph node after rendering.
        "docx_path":    None,
        "pdf_path":     None,
    }
