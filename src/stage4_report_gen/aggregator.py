"""
Aggregation of risk-assessed clauses for report generation.

Pure Python — groups clauses by risk level, computes contract-level
risk score, and identifies top risks. No ML model needed.
"""

import logging

from src.common.schema import RiskAssessedClause

logger = logging.getLogger(__name__)

RISK_WEIGHTS = {"HIGH": 1.0, "MEDIUM": 0.5, "LOW": 0.1}
RISK_ORDER   = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def group_by_risk_level(
    clauses: list[RiskAssessedClause],
) -> dict[str, list[RiskAssessedClause]]:
    """Group clauses by risk level (HIGH / MEDIUM / LOW).

    Args:
        clauses: Stage 3 output clauses.

    Returns:
        Dict mapping risk_level → list of clauses.
    """
    groups: dict[str, list[RiskAssessedClause]] = {"HIGH": [], "MEDIUM": [], "LOW": []}
    for clause in clauses:
        level = clause.risk_level.upper()
        if level in groups:
            groups[level].append(clause)
        else:
            logger.warning("Unexpected risk_level %r for clause %s — skipping", level, clause.clause_id)
    return groups


def compute_contract_risk_score(clauses: list[RiskAssessedClause]) -> float:
    """Compute an overall contract risk score from individual clause risks.

    Weighted average: HIGH=1.0, MEDIUM=0.5, LOW=0.1, each scaled by
    DeBERTa confidence. Normalized to [0, 1].

    Args:
        clauses: All risk-assessed clauses for one contract.

    Returns:
        Float in [0, 1] representing overall contract risk.
    """
    if not clauses:
        return 0.0

    total = sum(
        RISK_WEIGHTS.get(c.risk_level.upper(), 0.0) * max(c.confidence, 0.0)
        for c in clauses
    )
    # Max possible score = all HIGH at confidence 1.0
    return round(total / len(clauses), 4)


def get_top_risks(
    clauses: list[RiskAssessedClause],
    n: int = 5,
) -> list[RiskAssessedClause]:
    """Return the top-N highest-risk clauses sorted by severity then confidence.

    Args:
        clauses: All risk-assessed clauses.
        n: Number of top risks to return.

    Returns:
        Top-N clauses sorted by risk level (HIGH first) then confidence descending.
    """
    return sorted(
        clauses,
        key=lambda c: (RISK_ORDER.get(c.risk_level.upper(), 99), -c.confidence),
    )[:n]
