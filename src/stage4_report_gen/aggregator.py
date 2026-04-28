"""
Aggregation helpers for Stage 4.

Pure Python — no ML model. Two responsibilities under the new design:

  - `group_by_risk_level(clauses)`         → bucket clauses for the report's
                                              HIGH / MEDIUM / LOW tables.
  - `compute_contract_risk_score(clauses)` → contract-level numeric score
                                              shown in the report header.

Schema-agnostic by design: every helper reads attributes via `_get`, so it
accepts the dataclass form (`src.common.schema.RiskAssessedClause`) or the
pydantic form (`app.schemas.domain.RiskAssessedClause`) interchangeably.

Removed under the new design (per plan): `find_missing_protections`,
`get_top_risks`, `low_risk_summary`, and the EXPECTED_PROTECTIONS constant —
the redesigned report does not surface "missing" / "top-N" sections, and
LOW clauses are now enumerated in a table rather than summarized.
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


def _get(clause: Any, *names: str, default: Any = None) -> Any:
    """Read the first attribute that exists on `clause`.

    Tolerates dataclass attrs, pydantic attrs, and dict keys. Treats `None`
    as missing so partially-populated objects still resolve through to the
    default sensibly.
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
    for human readability:

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
