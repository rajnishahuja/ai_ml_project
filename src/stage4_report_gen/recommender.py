"""
Recommendation lookup table for Stage 4.

Maps (clause_type, risk_level) → human-readable remediation text. Pure
data + a thin lookup function — no ML model. The table is curated from
the CUAD clause descriptions and the Stage 3 label-review learnings
documented in ARCHITECTURE.md §"Labeling Review Learnings".

Lookup is case-insensitive on both keys. Missing entries fall through
to a per-risk-level default (`DEFAULT_RECOMMENDATIONS`) so HIGH-risk
clauses always carry actionable advice in the final report — even on
clause types the table doesn't cover yet.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# Curated (clause_type, risk_level) → recommendation. Keys are stored lowercase
# for case-insensitive lookup. Coverage focus:
#   - The 6 EXPECTED_PROTECTIONS from aggregator.py
#   - The high-impact CUAD types flagged in label review (license / IP / liability)
RECOMMENDATION_TABLE: dict[tuple[str, str], str] = {
    ("indemnification", "HIGH"): (
        "Negotiate mutual indemnification. Cap liability at contract value. "
        "Add carve-outs for gross negligence and willful misconduct."
    ),
    ("indemnification", "MEDIUM"): (
        "Confirm indemnification scope is balanced. Verify defense and "
        "settlement-control language is acceptable."
    ),

    ("termination for convenience", "HIGH"): (
        "Add a minimum notice period (60–90 days), wind-down obligations, "
        "and payment for work completed prior to termination."
    ),
    ("termination for convenience", "MEDIUM"): (
        "Confirm the notice period is workable and that payment for "
        "in-progress work is preserved."
    ),

    ("non-compete", "HIGH"): (
        "Limit geographic scope and duration (max 1–2 years). Ensure the "
        "restriction is reasonable relative to the business relationship."
    ),
    ("non-compete", "MEDIUM"): (
        "Tighten geographic / temporal scope. Add carve-outs for pre-existing "
        "business lines."
    ),

    ("cap on liability", "HIGH"): (
        "Cap total liability at a defined multiple of fees paid. Exclude "
        "consequential and indirect damages mutually."
    ),
    ("cap on liability", "MEDIUM"): (
        "Confirm the cap applies mutually and covers both direct and indirect "
        "claims under the contract."
    ),

    ("uncapped liability", "HIGH"): (
        "Convert to a capped liability clause with mutual exclusions for "
        "consequential damages. Reserve uncapped exposure only for IP "
        "infringement and gross negligence."
    ),
    ("uncapped liability", "MEDIUM"): (
        "Limit uncapped exposure to narrow, mutually agreed categories "
        "(e.g. confidentiality breach, IP indemnity)."
    ),

    ("liquidated damages", "HIGH"): (
        "Replace with actual-damages language or cap the liquidated amount. "
        "Confirm the figure is a reasonable forecast, not a penalty."
    ),

    ("ip ownership assignment", "HIGH"): (
        "Retain joint ownership or license-back rights for pre-existing IP. "
        "Clarify scope of assignment to project-specific deliverables."
    ),
    ("ip ownership assignment", "MEDIUM"): (
        "Confirm pre-existing IP is carved out from the assignment."
    ),

    ("license grant", "HIGH"): (
        "Narrow the license scope (field of use, territory, term). Avoid "
        "perpetual / irrevocable grants without compensating consideration."
    ),
    ("license grant", "MEDIUM"): (
        "Verify the license is revocable on material breach and bounded by "
        "field of use."
    ),

    ("irrevocable or perpetual license", "HIGH"): (
        "Replace with a fixed-term license with renewal terms. Add revocation "
        "rights on material breach or insolvency."
    ),

    ("non-transferable license", "MEDIUM"): (
        "Add affiliate / successor carve-outs to avoid blocking routine "
        "corporate restructuring."
    ),

    ("affiliate license-licensee", "HIGH"): (
        "Limit affiliate scope to wholly-owned entities. Add change-of-control "
        "termination rights."
    ),

    ("exclusivity", "HIGH"): (
        "Convert to non-exclusive or add minimum-volume off-ramps. Limit term "
        "and geographic scope."
    ),
    ("exclusivity", "MEDIUM"): (
        "Verify exclusivity is bounded by field of use and contains "
        "minimum-performance off-ramps."
    ),

    ("rofr/rofo/rofn", "HIGH"): (
        "Narrow trigger events. Add a market-test clause so the right does "
        "not block third-party offers indefinitely."
    ),

    ("anti-assignment", "HIGH"): (
        "Add carve-outs for assignment to affiliates, successors, and in "
        "connection with mergers / acquisitions."
    ),

    ("change of control", "HIGH"): (
        "Replace automatic termination with a notice-and-cure period. "
        "Restrict trigger events to genuine competitor acquisitions."
    ),

    ("most favored nation", "HIGH"): (
        "Limit to a defined product / customer scope. Add a measurement "
        "window so the obligation is auditable."
    ),

    ("minimum commitment", "HIGH"): (
        "Tie minimums to mutual performance obligations. Add force-majeure / "
        "market-condition adjustment mechanisms."
    ),

    ("price restrictions", "HIGH"): (
        "Replace fixed-price restrictions with index-linked or "
        "renegotiation triggers. Limit duration."
    ),

    ("volume restriction", "HIGH"): (
        "Convert hard caps to tiered pricing. Add headroom for forecast error."
    ),

    ("warranty duration", "HIGH"): (
        "Shorten warranty period to industry standard (12–24 months). Limit "
        "remedies to repair/replace at supplier's option."
    ),

    ("insurance", "HIGH"): (
        "Verify coverage limits match the contract's risk profile. Confirm "
        "additional-insured and waiver-of-subrogation language."
    ),

    ("governing law", "MEDIUM"): (
        "Confirm jurisdiction is acceptable for enforcement. Consider "
        "neutral forum if the parties are in different jurisdictions."
    ),

    ("audit rights", "HIGH"): (
        "Restrict audit frequency and scope. Add confidentiality protections "
        "and reasonable notice requirements."
    ),

    ("post-termination services", "HIGH"): (
        "Define a clear transition period and scope. Cap post-termination "
        "obligations to a reasonable wind-down window."
    ),

    ("covenant not to sue", "HIGH"): (
        "Limit to specific claims that are the subject of the contract. "
        "Avoid blanket releases of unrelated future claims."
    ),
}


# Per-risk-level fallback when no specific (clause_type, risk_level) entry
# matches. Always returns actionable text — never empty.
DEFAULT_RECOMMENDATIONS: dict[str, str] = {
    "HIGH": (
        "Review this clause with legal counsel before signing. Negotiate "
        "balanced obligations, cap exposure, and add mutual carve-outs."
    ),
    "MEDIUM": (
        "Have legal counsel review this clause. Confirm scope and "
        "obligations are commercially reasonable."
    ),
    "LOW": (
        "Standard / low-risk clause. No action required beyond routine review."
    ),
}


def get_recommendation(clause_type: str, risk_level: str) -> str:
    """Look up a remediation recommendation for a clause.

    Lookup is case-insensitive on both `clause_type` and `risk_level`.
    Falls back through:
        1. Exact (clause_type, risk_level) match
        2. Per-risk-level default (DEFAULT_RECOMMENDATIONS)
        3. Generic legal-review boilerplate

    Args:
        clause_type: CUAD clause type (e.g. "Indemnification", any case).
        risk_level: "HIGH" / "MEDIUM" / "LOW".

    Returns:
        Non-empty recommendation string.
    """
    if not clause_type:
        clause_type = ""
    risk_level_norm = (risk_level or "").upper()

    key = (clause_type.strip().lower(), risk_level_norm)
    if key in RECOMMENDATION_TABLE:
        return RECOMMENDATION_TABLE[key]

    if risk_level_norm in DEFAULT_RECOMMENDATIONS:
        return DEFAULT_RECOMMENDATIONS[risk_level_norm]

    return DEFAULT_RECOMMENDATIONS["MEDIUM"]
