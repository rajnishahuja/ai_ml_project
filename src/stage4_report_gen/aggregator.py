"""
Aggregation of risk-assessed clauses for report generation.

Pure Python — groups clauses by risk level, computes a contract-level
risk score, identifies the top-N risks, and detects missing standard
protections. No ML model needed here; the LLM-driven explanation work
lives in `explainer.py`.

Schema-agnostic by design: every helper reads attributes via `_get`, so
it accepts either the dataclass form (`src.common.schema.RiskAssessedClause`)
or the pydantic form (`app.schemas.domain.RiskAssessedClause`). The two
schemas use different field names for the explanation text — `risk_explanation`
vs `risk_reason` — and `_get` normalizes that.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Severity weights used for the contract-level score (HIGH ≫ MEDIUM ≫ LOW).
# Calibrated against ARCHITECTURE.md §"Stage 4 Output" example
# (overall_risk_score: 6.8 on a 0–10 scale → multiply by 10 at the end).
_RISK_WEIGHTS: dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.5,
    "LOW": 0.1,
}

# Severity ordering used for sorting top risks. Higher = more severe.
_SEVERITY_RANK: dict[str, int] = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}

# CUAD clause types that any well-formed commercial contract should contain.
# Used by `find_missing_protections`. List intentionally short — flagging too
# many "missing" clauses creates noise in the report. Source: review of CUAD
# v1 high-frequency types and the Atticus category descriptions.
EXPECTED_PROTECTIONS: tuple[str, ...] = (
    "Governing Law",
    "Cap On Liability",
    "Indemnification",
    "Termination For Convenience",
    "Insurance",
    "Warranty Duration",
)


def _get(clause: Any, *names: str, default: Any = None) -> Any:
    """Read the first attribute that exists on `clause`.

    Tries each name in order. Falls back through `__getitem__` for dicts
    and pydantic `model_dump`-style objects.
    """
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


def _normalize_risk(level: str | None) -> str:
    """Uppercase risk level; map empty/None to 'LOW' so unknowns don't crash."""
    if not level:
        return "LOW"
    return level.upper()


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def group_by_risk_level(clauses: Iterable[Any]) -> dict[str, list[Any]]:
    """Group clauses by risk level (HIGH / MEDIUM / LOW).

    Args:
        clauses: Stage 3 output clauses (any of the supported schema forms).

    Returns:
        Dict with exactly three keys: HIGH, MEDIUM, LOW. Each maps to a list
        of clauses. Order within a list preserves input order.
    """
    buckets: dict[str, list[Any]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for c in clauses:
        level = _normalize_risk(_get(c, "risk_level"))
        buckets.setdefault(level, []).append(c)
    return buckets


# ---------------------------------------------------------------------------
# Contract-level score
# ---------------------------------------------------------------------------

def compute_contract_risk_score(clauses: Iterable[Any]) -> float:
    """Compute an overall contract risk score in [0, 10].

    Confidence-weighted average of per-clause severity weights, scaled by 10
    for human readability (matches the example in ARCHITECTURE.md
    §"Stage 4 Output").

        score = 10 * Σ(severity[c] * confidence[c]) / Σ(confidence[c])

    Returns 0.0 for an empty input. Clauses with missing confidence default
    to 1.0 so they still contribute (this is the common case when the agent
    path does not emit a calibrated score).
    """
    clauses = list(clauses)
    if not clauses:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0
    for c in clauses:
        level = _normalize_risk(_get(c, "risk_level"))
        severity = _RISK_WEIGHTS.get(level, 0.0)
        # Default confidence=1.0 — agent-overridden clauses often carry None
        # (no calibrated score). Treat them as full weight.
        conf = float(_get(c, "confidence", default=1.0) or 1.0)
        weighted_sum += severity * conf
        total_weight += conf

    if total_weight == 0:
        return 0.0
    return round(10.0 * weighted_sum / total_weight, 2)


# ---------------------------------------------------------------------------
# Top risks
# ---------------------------------------------------------------------------

def get_top_risks(clauses: Iterable[Any], n: int = 5) -> list[Any]:
    """Return the top-N most severe clauses.

    Sort key (descending):
      1. severity rank (HIGH=3, MEDIUM=2, LOW=1)
      2. confidence (higher = more certain)

    Args:
        clauses: All risk-assessed clauses.
        n: Maximum number of clauses to return.

    Returns:
        List of up to `n` clauses, most-severe first. If the input has fewer
        than `n` clauses, returns all of them.
    """
    def sort_key(c: Any) -> tuple[int, float]:
        rank = _SEVERITY_RANK.get(_normalize_risk(_get(c, "risk_level")), 0)
        conf = float(_get(c, "confidence", default=0.0) or 0.0)
        return (rank, conf)

    return sorted(clauses, key=sort_key, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Missing-protection detection (gap analysis)
# ---------------------------------------------------------------------------

def find_missing_protections(
    clauses: Iterable[Any],
    expected_types: tuple[str, ...] = EXPECTED_PROTECTIONS,
) -> list[str]:
    """Identify standard protective clause types absent from the contract.

    Compares the set of clause types present in `clauses` against
    `expected_types` (case-insensitive). Returns the missing types in the
    original order of `expected_types`.

    Note: this is a lightweight gap analysis. It does not look at clause
    content — only presence by type. Stage 1+2 already filtered out
    low-confidence detections, so absence here means "DeBERTa did not find a
    matching span", which is a reasonable proxy for "the contract lacks this
    protection".
    """
    present = {(_get(c, "clause_type") or "").strip().lower() for c in clauses}
    return [t for t in expected_types if t.strip().lower() not in present]


# ---------------------------------------------------------------------------
# Convenience: low-risk summary text
# ---------------------------------------------------------------------------

def low_risk_summary(low_risk_clauses: list[Any]) -> str:
    """One-line natural-language summary of the LOW-risk bucket."""
    n = len(low_risk_clauses)
    if n == 0:
        return "No low-risk clauses identified."
    if n == 1:
        return "1 clause was assessed as standard / low risk."
    return f"{n} clauses were assessed as standard / low risk."
