"""
Aggregation of risk-assessed clauses for report generation.

Pure Python — groups clauses by risk level, computes contract-level
risk score, and identifies top risks. No ML model needed.
"""

import logging

from src.common.schema import RiskAssessedClause

logger = logging.getLogger(__name__)


def group_by_risk_level(
    clauses: list[RiskAssessedClause],
) -> dict[str, list[RiskAssessedClause]]:
    """Group clauses by risk level (HIGH / MEDIUM / LOW).

    Args:
        clauses: Stage 3 output clauses.

    Returns:
        Dict mapping risk_level → list of clauses.
    """
    raise NotImplementedError


def compute_contract_risk_score(clauses: list[RiskAssessedClause]) -> float:
    """Compute an overall contract risk score from individual clause risks.

    Weighted average: HIGH=1.0, MEDIUM=0.5, LOW=0.1. Weighted by
    confidence. Normalized to [0, 1].

    Args:
        clauses: All risk-assessed clauses for one contract.

    Returns:
        Float in [0, 1] representing overall contract risk.
    """
    raise NotImplementedError


def get_top_risks(
    clauses: list[RiskAssessedClause],
    n: int = 5,
) -> list[RiskAssessedClause]:
    """Return the top-N highest-risk clauses sorted by confidence.

    Args:
        clauses: All risk-assessed clauses.
        n: Number of top risks to return.

    Returns:
        Top-N clauses sorted by risk severity and confidence.
    """
    raise NotImplementedError
