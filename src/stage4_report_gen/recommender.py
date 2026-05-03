"""
Recommendation lookup table.

Maps (clause_type, risk_level) → remediation text.
No ML model — curated dict with human-written recommendations.
Covers all 36 risk-relevant CUAD clause types × 3 risk levels.
Falls back to DEFAULT_RECOMMENDATION for unmapped combinations.
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup table  (clause_type, risk_level) → recommendation
# Keys are lowercase for case-insensitive lookup.
# ---------------------------------------------------------------------------

RECOMMENDATION_TABLE: dict[tuple[str, str], str] = {

    # --- Renewal Term ---
    ("renewal term", "HIGH"):   "Automatic renewal locks you in long-term. Negotiate manual renewal only, or add an explicit opt-out window with sufficient notice (30–60 days).",
    ("renewal term", "MEDIUM"): "Renewal duration may exceed your planning horizon. Confirm notice requirements are trackable and add calendar reminders.",
    ("renewal term", "LOW"):    "Standard renewal terms — no action required.",

    # --- Notice Period to Terminate Renewal ---
    ("notice period to terminate renewal", "HIGH"):   "Notice period is too short or absent to act on before auto-renewal triggers. Negotiate for at least 30–60 days and add internal calendar alerts.",
    ("notice period to terminate renewal", "MEDIUM"): "Confirm the notice period aligns with internal approval timelines. Extend if sign-off process takes longer.",
    ("notice period to terminate renewal", "LOW"):    "Notice period is reasonable — no action required.",

    # --- Governing Law ---
    ("governing law", "HIGH"):   "Jurisdiction is foreign or adversarial, significantly increasing litigation cost and risk. Negotiate for your home state/country or a mutually neutral jurisdiction.",
    ("governing law", "MEDIUM"): "Non-home jurisdiction raises litigation cost. Consider negotiating for mutual choice or preferred venue clause.",
    ("governing law", "LOW"):    "Governing law is standard and non-adversarial — no action required.",

    # --- Most Favored Nation ---
    ("most favored nation", "HIGH"):   "MFN clause forces price matching across all customers, creating significant commercial exposure. Negotiate carve-outs for volume discounts, promotions, and special arrangements.",
    ("most favored nation", "MEDIUM"): "Scope is broad. Narrow to specific product/service categories and add exceptions for promotional pricing.",
    ("most favored nation", "LOW"):    "MFN is limited in scope and duration — acceptable as-is.",

    # --- Non-Compete ---
    ("non-compete", "HIGH"):   "Overly broad restriction in geography, duration, or business scope. Negotiate to narrow scope, shorten duration (max 1–2 years), and add specific carve-outs for existing activities.",
    ("non-compete", "MEDIUM"): "Geographic or time scope is broader than typical. Review against industry norms and negotiate reasonable limits.",
    ("non-compete", "LOW"):    "Reasonable non-compete with limited scope and duration — no action required.",

    # --- Exclusivity ---
    ("exclusivity", "HIGH"):   "Exclusive commitment prevents revenue diversification. Negotiate minimum revenue guarantees in exchange, or limit exclusivity to a specific product/geography/duration.",
    ("exclusivity", "MEDIUM"): "Exclusivity in a narrow segment — confirm alignment with business strategy and add a performance-based exit clause.",
    ("exclusivity", "LOW"):    "Limited exclusivity with clear boundaries — acceptable as-is.",

    # --- No-Solicit of Customers ---
    ("no-solicit of customers", "HIGH"):   "Broad post-termination restriction on customer solicitation. Negotiate to limit duration (max 1–2 years), geographic scope, and define 'customers' narrowly.",
    ("no-solicit of customers", "MEDIUM"): "Confirm the definition of 'customers' is narrow (active accounts only) and the restriction period is clearly limited.",
    ("no-solicit of customers", "LOW"):    "Standard no-solicit with reasonable scope — no action required.",

    # --- Competitive Restriction Exception ---
    ("competitive restriction exception", "HIGH"):   "Exceptions to competitive restrictions are narrow or absent. Negotiate to broaden carve-outs for existing customers, pre-existing activities, and affiliates.",
    ("competitive restriction exception", "MEDIUM"): "Carve-outs exist but may not cover all your business activities. Review against operational needs.",
    ("competitive restriction exception", "LOW"):    "Adequate exceptions in place — no action required.",

    # --- No-Solicit of Employees ---
    ("no-solicit of employees", "HIGH"):   "Overly broad restriction on hiring counterparty employees. Negotiate to limit to direct active solicitation only and add a 12-month sunset clause.",
    ("no-solicit of employees", "MEDIUM"): "Restriction may cover passive hiring. Negotiate to limit to active direct solicitation and exclude responses to public job postings.",
    ("no-solicit of employees", "LOW"):    "Standard no-solicit of employees with reasonable scope — no action required.",

    # --- Non-Disparagement ---
    ("non-disparagement", "HIGH"):   "One-sided non-disparagement without reciprocity. Negotiate for mutual obligations and limit scope to factually false statements.",
    ("non-disparagement", "MEDIUM"): "Scope of 'disparagement' is broad. Negotiate to limit to knowingly false statements and add a truth defence carve-out.",
    ("non-disparagement", "LOW"):    "Mutual non-disparagement with reasonable scope — no action required.",

    # --- Termination for Convenience ---
    ("termination for convenience", "HIGH"):   "Counterparty can terminate with minimal notice, leaving you with unrecovered investment. Negotiate for 60–90 days notice and wind-down payments for work in progress.",
    ("termination for convenience", "MEDIUM"): "Notice period may be insufficient for operational planning. Negotiate to extend and add transition assistance obligations.",
    ("termination for convenience", "LOW"):    "Reasonable termination for convenience with adequate notice — no action required.",

    # --- Rofr/Rofo/Rofn ---
    ("rofr/rofo/rofn", "HIGH"):   "Right of first refusal/offer significantly restricts exit options and M&A flexibility. Negotiate for a fixed exercise window (e.g., 10 business days) and limit to specific asset types.",
    ("rofr/rofo/rofn", "MEDIUM"): "Confirm timelines for exercising the right are workable for your transaction processes. Add a deemed-waiver if not exercised within the window.",
    ("rofr/rofo/rofn", "LOW"):    "Limited right of first refusal with clear procedures — no action required.",

    # --- Change of Control ---
    ("change of control", "HIGH"):   "Counterparty can terminate or impose conditions on merger/acquisition. Negotiate for notice-only triggers (not consent) and carve-outs for internal restructuring and affiliate transfers.",
    ("change of control", "MEDIUM"): "Consent requirement may delay M&A transactions. Ensure the clause applies symmetrically to both parties.",
    ("change of control", "LOW"):    "Standard change of control notification — no action required.",

    # --- Anti-Assignment ---
    ("anti-assignment", "HIGH"):   "Strict consent requirement for any assignment limits operational flexibility. Negotiate for notice-only on intra-group transfers and explicit carve-outs for affiliates and successors.",
    ("anti-assignment", "MEDIUM"): "Confirm affiliate transfers and corporate reorganisations are explicitly carved out from the consent requirement.",
    ("anti-assignment", "LOW"):    "Standard anti-assignment with affiliate carve-out — no action required.",

    # --- Revenue/Profit Sharing ---
    ("revenue/profit sharing", "HIGH"):   "Revenue share is disproportionate or uncapped. Negotiate for a percentage cap, a step-down structure as volume grows, and audit rights to verify calculations.",
    ("revenue/profit sharing", "MEDIUM"): "Ensure calculation methodology is clearly defined and audit rights are included.",
    ("revenue/profit sharing", "LOW"):    "Revenue sharing terms are clearly defined and auditable — no action required.",

    # --- Price Restrictions ---
    ("price restrictions", "HIGH"):   "Pricing constraints significantly limit revenue optimization. Negotiate for CPI inflation adjustments, periodic review rights, and volume-based exceptions.",
    ("price restrictions", "MEDIUM"): "Price restriction lacks an adjustment mechanism. Add a CPI or cost-of-living escalator and a periodic review right.",
    ("price restrictions", "LOW"):    "Price restrictions are narrow and time-limited — no action required.",

    # --- Minimum Commitment ---
    ("minimum commitment", "HIGH"):   "Minimum purchase commitment is significant relative to expected usage. Negotiate for volume step-ups tied to performance milestones and a force-majeure relief clause.",
    ("minimum commitment", "MEDIUM"): "Confirm minimum commitment is achievable under conservative demand scenarios. Add a catch-up or roll-forward mechanism.",
    ("minimum commitment", "LOW"):    "Minimum commitment is modest and achievable — no action required.",

    # --- Volume Restriction ---
    ("volume restriction", "HIGH"):   "Usage cap or fee escalation is triggered at a low threshold, limiting growth. Negotiate for higher thresholds or a fully uncapped usage tier.",
    ("volume restriction", "MEDIUM"): "Ensure thresholds are well above expected usage and fee escalation is capped at a reasonable multiple.",
    ("volume restriction", "LOW"):    "Volume thresholds are generous and well above expected usage — no action required.",

    # --- IP Ownership Assignment ---
    ("ip ownership assignment", "HIGH"):   "Broad assignment includes pre-existing or background IP. Negotiate to limit assignment to project-specific deliverables and retain a royalty-free license-back for internal use.",
    ("ip ownership assignment", "MEDIUM"): "Confirm IP assignment is limited to deliverables and does not capture background IP. Add explicit license-back rights for internal use.",
    ("ip ownership assignment", "LOW"):    "IP assignment is narrowly scoped to project deliverables — no action required.",

    # --- Joint IP Ownership ---
    ("joint ip ownership", "HIGH"):   "Joint ownership without consent requirements for commercialisation allows either party to exploit IP freely. Negotiate for mutual consent on third-party licensing and revenue sharing.",
    ("joint ip ownership", "MEDIUM"): "Ensure joint ownership terms include clear procedures for commercialisation decisions and revenue allocation.",
    ("joint ip ownership", "LOW"):    "Joint IP ownership with clear governance procedures — no action required.",

    # --- License Grant ---
    ("license grant", "HIGH"):   "License scope is overly broad (irrevocable, fully sublicensable, unlimited field-of-use). Negotiate to narrow field-of-use, add termination rights on material breach, and restrict sublicensing.",
    ("license grant", "MEDIUM"): "Confirm license scope is limited to the intended use case. Verify sublicensing rights and geographic limitations are explicit.",
    ("license grant", "LOW"):    "License is appropriately scoped and limited — no action required.",

    # --- Non-Transferable License ---
    ("non-transferable license", "HIGH"):   "Non-transferable restriction lacks affiliate or successor carve-outs, limiting operational flexibility. Negotiate for explicit carve-outs for affiliates and corporate reorganisations.",
    ("non-transferable license", "MEDIUM"): "Confirm that affiliate transfers and corporate reorganisations are explicitly excluded from the non-transfer restriction.",
    ("non-transferable license", "LOW"):    "Non-transferable restriction includes standard carve-outs — no action required.",

    # --- Affiliate License-Licensor ---
    ("affiliate license-licensor", "HIGH"):   "Licensor affiliate IP is included without ownership warranties. Negotiate for representations confirming the licensor holds rights to all included affiliate IP and indemnification for third-party claims.",
    ("affiliate license-licensor", "MEDIUM"): "Ensure indemnification covers affiliate IP claims and the scope of affiliate IP is clearly defined.",
    ("affiliate license-licensor", "LOW"):    "Affiliate licensor IP is clearly identified and warranted — no action required.",

    # --- Affiliate License-Licensee ---
    ("affiliate license-licensee", "HIGH"):   "Sublicense to affiliates extends to third parties, creating uncontrolled IP distribution. Negotiate to restrict sublicensing strictly to entities under common control.",
    ("affiliate license-licensee", "MEDIUM"): "Confirm affiliate sublicense is limited to entities under common control and does not extend to third-party or downstream sublicensing.",
    ("affiliate license-licensee", "LOW"):    "Affiliate licensee rights are appropriately scoped to controlled entities — no action required.",

    # --- Unlimited/All-You-Can-Eat-License ---
    ("unlimited/all-you-can-eat-license", "HIGH"):   "Unlimited license without a corresponding liability cap creates significant exposure. Ensure liability is capped commensurate with the unlimited usage rights granted.",
    ("unlimited/all-you-can-eat-license", "MEDIUM"): "Confirm 'unlimited' scope is clearly defined (users, deployments, geographies) to avoid future disputes over scope.",
    ("unlimited/all-you-can-eat-license", "LOW"):    "Unlimited license is well-defined with clear boundaries — no action required.",

    # --- Irrevocable or Perpetual License ---
    ("irrevocable or perpetual license", "HIGH"):   "Irrevocable perpetual license with no termination rights. Negotiate for termination rights on material breach, insolvency, or change of control.",
    ("irrevocable or perpetual license", "MEDIUM"): "Confirm perpetual license includes a clearly defined usage scope to prevent scope creep over time.",
    ("irrevocable or perpetual license", "LOW"):    "Perpetual license is narrowly scoped with adequate usage limitations — no action required.",

    # --- Source Code Escrow ---
    ("source code escrow", "HIGH"):   "Escrow release conditions are too broad — triggered by minor events. Negotiate to limit release triggers to bankruptcy and material uncured breach only.",
    ("source code escrow", "MEDIUM"): "Confirm the escrow agent is reputable and release procedures and verification rights are clearly defined.",
    ("source code escrow", "LOW"):    "Source code escrow is properly structured with limited release triggers — no action required.",

    # --- Post-Termination Services ---
    ("post-termination services", "HIGH"):   "Extensive post-termination obligations without defined end date or compensation. Negotiate for a fixed transition period, payment for effort, and clear scope limitations.",
    ("post-termination services", "MEDIUM"): "Ensure post-termination obligations are time-limited and compensated where significant effort is required.",
    ("post-termination services", "LOW"):    "Post-termination obligations are limited in scope and duration — no action required.",

    # --- Audit Rights ---
    ("audit rights", "HIGH"):   "Audit rights are one-sided and overly broad — unlimited access to books and systems. Negotiate for scope limitations, advance notice requirements, frequency caps, and cost allocation.",
    ("audit rights", "MEDIUM"): "Ensure audit frequency is capped (once per year), requires reasonable advance notice, and costs are allocated appropriately.",
    ("audit rights", "LOW"):    "Audit rights are appropriately scoped with standard procedural protections — no action required.",

    # --- Uncapped Liability ---
    ("uncapped liability", "HIGH"):   "Uncapped liability with no exclusion for consequential damages creates significant financial exposure. Negotiate for a mutual cap tied to contract value and exclusion of indirect/consequential damages.",
    ("uncapped liability", "MEDIUM"): "Uncapped liability is limited to specific breach types. Confirm scope is narrow and backed by adequate insurance coverage.",
    ("uncapped liability", "LOW"):    "Uncapped liability is limited to gross negligence or wilful misconduct — standard and acceptable.",

    # --- Cap on Liability ---
    ("cap on liability", "HIGH"):   "Liability cap is set below contract value, or applies asymmetrically to your obligations only. Negotiate for a mutual cap at minimum contract value with carve-outs for IP and confidentiality breaches.",
    ("cap on liability", "MEDIUM"): "Confirm the cap applies mutually and that carve-outs for serious breaches (IP, fraud, gross negligence) are present.",
    ("cap on liability", "LOW"):    "Liability cap is mutual, set at a reasonable level, and includes standard carve-outs — no action required.",

    # --- Liquidated Damages ---
    ("liquidated damages", "HIGH"):   "Liquidated damages appear punitive rather than compensatory, creating dispute risk. Negotiate to tie damages to actual harm, add a cap, and include a reasonableness clause.",
    ("liquidated damages", "MEDIUM"): "Confirm liquidated damages are proportionate to potential harm and clearly defined. Add a cap and a reasonableness/mitigation clause.",
    ("liquidated damages", "LOW"):    "Liquidated damages are proportionate and clearly defined — no action required.",

    # --- Warranty Duration ---
    ("warranty duration", "HIGH"):   "Warranty period is very short or liability for warranty breach is uncapped. Negotiate for a reasonable warranty period (min 12 months) and cap warranty liability at contract value.",
    ("warranty duration", "MEDIUM"): "Confirm warranty scope and available remedies (repair, replace, refund) are clearly defined and proportionate.",
    ("warranty duration", "LOW"):    "Warranty duration and scope are reasonable and clearly documented — no action required.",

    # --- Insurance ---
    ("insurance", "HIGH"):   "Insurance requirements are one-sided or set at unreasonably high coverage levels. Negotiate for mutual requirements and market-standard coverage amounts with named-insured status.",
    ("insurance", "MEDIUM"): "Confirm required coverage types and amounts are achievable and align with your existing policies. Add an obligation to notify of material policy changes.",
    ("insurance", "LOW"):    "Insurance requirements are standard and achievable — no action required.",

    # --- Covenant Not to Sue ---
    ("covenant not to sue", "HIGH"):   "Broad covenant waives rights to challenge IP validity across the entire portfolio. Negotiate to limit the covenant strictly to IP rights covered by this agreement.",
    ("covenant not to sue", "MEDIUM"): "Confirm the covenant is limited to the specific IP covered by the agreement and does not extend to the broader IP portfolio.",
    ("covenant not to sue", "LOW"):    "Covenant not to sue is narrowly scoped to the agreement's IP — no action required.",

    # --- Third Party Beneficiary ---
    ("third party beneficiary", "HIGH"):   "Broad third-party beneficiary rights could expose you to claims from unknown parties. Negotiate to explicitly exclude all third-party beneficiary rights, or name specific permitted beneficiaries.",
    ("third party beneficiary", "MEDIUM"): "Identify the specific third-party beneficiaries and confirm the scope of their rights is clearly limited.",
    ("third party beneficiary", "LOW"):    "Third-party beneficiary rights are clearly defined and limited in scope — no action required.",
}

# Default fallback when no specific mapping exists.
DEFAULT_RECOMMENDATION = (
    "Review this clause with legal counsel. Consider negotiating terms "
    "to reduce risk exposure and ensure obligations are balanced."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendation(clause_type: str, risk_level: str) -> str:
    """Return a remediation recommendation for the given clause and risk level.

    Lookup is case-insensitive. Falls back to DEFAULT_RECOMMENDATION if
    the combination is not in the table.

    Args:
        clause_type: CUAD clause type (e.g. "Cap on Liability").
        risk_level:  Risk level string ("HIGH", "MEDIUM", "LOW").

    Returns:
        Recommendation text string.
    """
    key = (clause_type.strip().lower(), risk_level.strip().upper())
    rec = RECOMMENDATION_TABLE.get(key)
    if rec is None:
        logger.debug("No recommendation for (%r, %r) — using default", clause_type, risk_level)
        return DEFAULT_RECOMMENDATION
    return rec
