"""
Recommendation lookup table.

Maps (clause_type, risk_pattern) → remediation text.
No ML model — curated dict with human-written recommendations.
"""

import logging

logger = logging.getLogger(__name__)

# Curated recommendation lookup table.
# Key: (clause_type, risk_level) → remediation text.
# Extend this dict as new clause types / risk patterns are identified.
RECOMMENDATION_TABLE: dict[tuple[str, str], str] = {
    ("Indemnification", "HIGH"): (
        "Negotiate mutual indemnification. Cap liability at contract value. "
        "Add carve-outs for gross negligence and willful misconduct."
    ),
    ("Termination For Convenience", "HIGH"): (
        "Add minimum notice period (60-90 days). Include wind-down obligations "
        "and payment for work completed prior to termination."
    ),
    ("Non-Compete", "HIGH"): (
        "Limit geographic scope and duration (max 1-2 years). Ensure the "
        "restriction is reasonable relative to the business relationship."
    ),
    ("Liability", "HIGH"): (
        "Cap total liability. Exclude consequential damages. Add mutual "
        "limitation of liability clauses."
    ),
    ("Ip Ownership Assignment", "HIGH"): (
        "Retain joint ownership or license-back rights for pre-existing IP. "
        "Clarify scope of assignment to project-specific deliverables."
    ),
    # Add more entries as needed during implementation.
}

# Default recommendation when no specific mapping exists.
DEFAULT_RECOMMENDATION = (
    "Review this clause with legal counsel. Consider negotiating terms "
    "to reduce risk exposure and ensure balanced obligations."
)


def get_recommendation(clause_type: str, risk_level: str) -> str:
    """Look up a remediation recommendation for the clause.

    Args:
        clause_type: CUAD clause type (e.g. "Indemnification").
        risk_level: Risk level string ("HIGH", "MEDIUM", "LOW").

    Returns:
        Remediation recommendation text.
    """
    raise NotImplementedError
