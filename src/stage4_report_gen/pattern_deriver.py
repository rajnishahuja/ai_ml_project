"""Risk pattern derivation — tier-1 regex + tier-2 LLM fallback.

Maps a clause's free-form `risk_explanation` to one code from the controlled
vocabulary (~40 codes). Tier-1 is fast deterministic pattern matching;
tier-2 is a constrained Gemini call when tier-1 returns `unknown_pattern`.
Tier-2 is gated by `pattern_deriver.use_llm_tier2` (env override
STAGE4_PATTERN_DERIVE_USE_LLM); per docs/STAGE4_DECISION_LOG.md decision
#2, every tier-2 invocation is logged at INFO with clause_id and resulting
pattern code.

Inputs to tier-2 are clause-level only (clause_type + risk_explanation +
≤200-char excerpt). Full contract text never reaches the LLM.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

from src.stage4_report_gen.llm_client import GeminiClient

logger = logging.getLogger(__name__)


PATTERN_CODES = (
    "one_sided_against_signing_party",
    "one_sided_against_counterparty",
    "mutual_balanced",
    "perpetual_irrevocable_grant_to_counterparty",
    "perpetual_irrevocable_grant_to_signing_party",
    "uncapped_financial_exposure",
    "capped_financial_exposure",
    "automatic_renewal_short_notice",
    "automatic_renewal_balanced",
    "mutual_renewal_required",
    "unilateral_renewal_right_counterparty",
    "subjective_termination_trigger",
    "objective_termination_trigger",
    "vague_undefined_terms",
    "missing_carveouts",
    "void_ab_initio_consequence",
    "change_of_control_termination_against_signing_party",
    "change_of_control_termination_for_signing_party",
    "mfn_in_favor_of_signing_party",
    "mfn_against_signing_party",
    "non_compete_broad_and_long",
    "non_compete_narrow_or_short",
    "non_disparagement_perpetual_one_sided",
    "non_disparagement_mutual_with_carveouts",
    "confidentiality_unbalanced_or_perpetual",
    "confidentiality_standard",
    "assignment_restricted_only_for_signing_party",
    "auto_assignment_to_counterparty_affiliates",
    "assignment_requires_consent_both_parties",
    "audit_rights_excessive_against_signing_party",
    "audit_rights_reasonable",
    "governing_law_unfavorable_jurisdiction",
    "governing_law_neutral_or_favorable",
    "force_majeure_narrow",
    "force_majeure_standard",
    "warranty_disclaimer_broad",
    "warranty_protections_present",
    "insufficient_text_for_assessment",
    "unknown_pattern",
)

UNKNOWN = "unknown_pattern"


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern
    code: str
    requires_clause_type: Optional[str] = None  # if set, only fires for that type


def _r(text: str) -> re.Pattern:
    return re.compile(text, re.IGNORECASE)


# Order matters — earlier rules win. More specific rules go first.
_RULES: tuple[_Rule, ...] = (
    # --- Insufficient text ---------------------------------------------
    _Rule(
        _r(r"\b(insufficient|truncated|fragment(ed)?|too short|cannot determine|unable to assess)\b"),
        "insufficient_text_for_assessment",
    ),
    # --- Change of Control --------------------------------------------
    _Rule(
        _r(r"change of control.*(against|disadvantag(es?|ing)|hurts?|harms?)\s+(the\s+)?signing"),
        "change_of_control_termination_against_signing_party",
    ),
    _Rule(
        _r(r"change of control.*(in favor of|benefits?|favors?|protects?)\s+(the\s+)?signing"),
        "change_of_control_termination_for_signing_party",
    ),
    _Rule(
        _r(r"\bchange of control\b.*\b(terminat(e|ion)|trigger)"),
        "change_of_control_termination_against_signing_party",
        requires_clause_type="Change of Control",
    ),
    # --- MFN -----------------------------------------------------------
    _Rule(
        _r(r"\bmost favored nation\b.*\b(against|disadvantag|hurts?|harms?|burdens?)\s+(the\s+)?signing"),
        "mfn_against_signing_party",
    ),
    _Rule(
        _r(r"\bmost favored nation\b.*\b(in favor of|benefits?|favors?)\s+(the\s+)?signing"),
        "mfn_in_favor_of_signing_party",
    ),
    _Rule(
        _r(r"\bMFN\b.*\b(against|on|burdens?)\s+(the\s+)?signing"),
        "mfn_against_signing_party",
    ),
    # --- Perpetual / Irrevocable License ------------------------------
    _Rule(
        _r(r"\b(perpetual|irrevocable)\b.*\b(grant(ed)?|license).*\bto\s+(the\s+)?(counterparty|vendor|licensee)\b"),
        "perpetual_irrevocable_grant_to_counterparty",
    ),
    _Rule(
        _r(r"\b(perpetual|irrevocable)\b.*\b(grant(ed)?|license).*\bto\s+(the\s+)?signing\s+party\b"),
        "perpetual_irrevocable_grant_to_signing_party",
    ),
    _Rule(
        _r(r"\bperpetual\b.*\birrevocable\b"),
        "perpetual_irrevocable_grant_to_counterparty",
    ),
    # --- Uncapped / Capped liability ----------------------------------
    _Rule(
        _r(r"\b(uncapped|unlimited|unbounded|no cap)\b.*\b(liabilit|exposure|damages|financial)"),
        "uncapped_financial_exposure",
    ),
    _Rule(
        _r(r"\b(unlimited|uncapped)\s+(financial\s+)?(exposure|liability)"),
        "uncapped_financial_exposure",
    ),
    _Rule(
        _r(r"\b(capped|cap of|caps? at|limited to|maximum of)\b.*\b(liabilit|exposure|damages|fees)"),
        "capped_financial_exposure",
    ),
    # --- Renewal / Auto-renewal ---------------------------------------
    _Rule(
        _r(r"\bauto(matic)?[\s-]?renew(al|s|ing)?\b.*\b(short|narrow|tight|insufficient|inadequate)\s+notice\b"),
        "automatic_renewal_short_notice",
    ),
    _Rule(
        _r(r"\bauto(matic)?[\s-]?renew(al|s|ing)?\b.*\b\d{1,2}\s*(day|week)s?\s+notice\b"),
        "automatic_renewal_short_notice",
    ),
    _Rule(
        _r(r"\b(unilateral|sole|one[\s-]sided)\s+(option to renew|right to renew|renewal right)"),
        "unilateral_renewal_right_counterparty",
    ),
    _Rule(
        _r(r"\bmutual(ly)?\s+(agreed|consent(ed)?)\s+renew"),
        "mutual_renewal_required",
    ),
    _Rule(
        _r(r"\bauto(matic)?[\s-]?renew(al)?\b.*\b(60|90|reasonable|adequate|sufficient)\s*(day|days)?\s*notice\b"),
        "automatic_renewal_balanced",
    ),
    # --- Termination triggers -----------------------------------------
    _Rule(
        _r(r"\b(subjective|sole discretion|in (its|their) discretion|at will)\b.*\b(terminat(e|ion))"),
        "subjective_termination_trigger",
    ),
    _Rule(
        _r(r"\b(objective|defined|specific|enumerated|listed)\b.*\b(terminat(e|ion))"),
        "objective_termination_trigger",
    ),
    # --- Assignment ---------------------------------------------------
    _Rule(
        _r(r"\bassignment\b.*\brestricted\s+only\s+(for|on)\s+(the\s+)?signing"),
        "assignment_restricted_only_for_signing_party",
    ),
    _Rule(
        _r(r"\bone[\s-]sided\b.*\bassignment"),
        "assignment_restricted_only_for_signing_party",
    ),
    _Rule(
        _r(r"\bauto(matic(ally)?)?[\s-]?assignment\b.*\b(affiliate|subsidiar)"),
        "auto_assignment_to_counterparty_affiliates",
    ),
    _Rule(
        _r(r"\bassign(ment|s|able)\b.*?\bto\b.*?\baffiliate"),
        "auto_assignment_to_counterparty_affiliates",
    ),
    _Rule(
        _r(r"\bmutual(ly)?\s+consent\b.*\bassignment"),
        "assignment_requires_consent_both_parties",
    ),
    # --- Non-compete --------------------------------------------------
    _Rule(
        _r(r"\bnon[\s-]?compete\b.*\b(broad|long|wide|sweeping|extensive)"),
        "non_compete_broad_and_long",
    ),
    _Rule(
        _r(r"\bnon[\s-]?compete\b.*\b\d{1,2}\s*(year|years)\s*(\+|or more|plus)?\b"),
        "non_compete_broad_and_long",
    ),
    _Rule(
        _r(r"\bnon[\s-]?compete\b.*\b(narrow|short|limited|specific)"),
        "non_compete_narrow_or_short",
    ),
    # --- Non-disparagement --------------------------------------------
    _Rule(
        _r(r"\bnon[\s-]?disparag(e|ement)\b.*\b(perpetual|forever|indefinite|no\s+(time|duration)\s+limit|one[\s-]sided)"),
        "non_disparagement_perpetual_one_sided",
    ),
    _Rule(
        _r(r"\bnon[\s-]?disparag(e|ement)\b.*\b(mutual|reciprocal).*\b(carve[\s-]?out|exception)"),
        "non_disparagement_mutual_with_carveouts",
    ),
    # --- Confidentiality ----------------------------------------------
    _Rule(
        _r(r"\bconfidential(ity)?\b.*\b(unbalanced|one[\s-]sided|asymmetric|perpetual|forever)"),
        "confidentiality_unbalanced_or_perpetual",
    ),
    _Rule(
        _r(r"\bconfidential(ity)?\b.*\bstandard\b"),
        "confidentiality_standard",
    ),
    # --- Audit rights -------------------------------------------------
    _Rule(
        _r(r"\baudit\b.*\b(excessive|unrestricted|unlimited|burdensome|intrusive|on demand)"),
        "audit_rights_excessive_against_signing_party",
    ),
    _Rule(
        _r(r"\baudit\b.*\b(reasonable|standard|once per year|with notice)"),
        "audit_rights_reasonable",
    ),
    # --- Governing law ------------------------------------------------
    _Rule(
        _r(r"\b(governing law|jurisdiction|venue|choice of law)\b.*\b(unfavorable|hostile|distant|foreign|inconvenient)"),
        "governing_law_unfavorable_jurisdiction",
    ),
    _Rule(
        _r(r"\b(governing law|jurisdiction|venue)\b.*\b(neutral|favorable|home|local|delaware|new york|english law)"),
        "governing_law_neutral_or_favorable",
    ),
    # --- Force majeure ------------------------------------------------
    _Rule(
        _r(r"\bforce majeure\b.*\b(narrow|limited|restricted|excludes|enumerated only)"),
        "force_majeure_narrow",
    ),
    _Rule(
        _r(r"\bforce majeure\b.*\b(standard|broad|customary|industry-standard)"),
        "force_majeure_standard",
    ),
    # --- Warranty -----------------------------------------------------
    _Rule(
        _r(r"\bwarrant(y|ies)\b.*\b(disclaim(s|er|ed)|broad disclaimer|as[\s-]is|no warranties)"),
        "warranty_disclaimer_broad",
    ),
    _Rule(
        _r(r"\bwarrant(y|ies)\b.*\b(present|provided|express)"),
        "warranty_protections_present",
    ),
    # --- Specific risk constructs -------------------------------------
    _Rule(
        _r(r"\bvoid\s+ab\s+initio\b"),
        "void_ab_initio_consequence",
    ),
    _Rule(
        _r(r"\b(missing|lacks?|no|without)\s+(carve[\s-]?outs?|exceptions?|exclusions?)"),
        "missing_carveouts",
    ),
    _Rule(
        _r(r"\b(vague|undefined|ambiguous|imprecise|unclear)\s+(term|definition|language|scope)"),
        "vague_undefined_terms",
    ),
    # --- One-sided / Mutual fallbacks (keep last) ---------------------
    _Rule(
        _r(r"\bone[\s-]sided\b.*\b(against|burdens?|hurts?|harms?)\s+(the\s+)?signing"),
        "one_sided_against_signing_party",
    ),
    _Rule(
        _r(r"\bone[\s-]sided\b.*\b(against|burdens?)\s+(the\s+)?counterparty"),
        "one_sided_against_counterparty",
    ),
    _Rule(
        _r(r"\bone[\s-]sided\b"),
        "one_sided_against_signing_party",
    ),
    _Rule(
        _r(r"\bmutual(ly)?\s+balanced\b"),
        "mutual_balanced",
    ),
    _Rule(
        _r(r"\bmutual(ly)?\b.*\b(both parties|either party|reciprocal)"),
        "mutual_balanced",
    ),
)


def derive_tier1(clause_type: str, risk_explanation: str) -> str:
    """Return a controlled-vocab code, or `unknown_pattern` if no rule matches."""
    text = (risk_explanation or "").strip()
    if not text:
        return "insufficient_text_for_assessment"
    for rule in _RULES:
        if rule.requires_clause_type and rule.requires_clause_type != clause_type:
            continue
        if rule.pattern.search(text):
            return rule.code
    return UNKNOWN


def _is_tier2_enabled(config: dict) -> bool:
    env = os.environ.get("STAGE4_PATTERN_DERIVE_USE_LLM")
    if env is not None:
        return env.strip().lower() in {"1", "true", "yes", "on"}
    return bool(config.get("pattern_deriver", {}).get("use_llm_tier2", True))


_TIER2_PROMPT_TEMPLATE = """You are a legal-risk pattern classifier. Given a clause type and a free-form risk explanation, return EXACTLY ONE code from the controlled vocabulary below — nothing else, no preamble, no explanation, no quotes.

Controlled vocabulary:
{codes}

Clause type: {clause_type}
Risk explanation: {risk_explanation}
Clause text excerpt (≤200 chars, for grounding only): {excerpt}

Return: a single code from the list above."""


def derive_tier2(
    *,
    clause_id: str,
    clause_type: str,
    risk_explanation: str,
    clause_text_excerpt: str,
    client: GeminiClient,
) -> str:
    """LLM fallback. Returns one of PATTERN_CODES or `unknown_pattern`.

    Soft-fails to `unknown_pattern` on any failure (failure isolation —
    hard rule #7).
    """
    excerpt = (clause_text_excerpt or "")[:200]
    prompt = _TIER2_PROMPT_TEMPLATE.format(
        codes="\n".join(f"  - {c}" for c in PATTERN_CODES),
        clause_type=clause_type,
        risk_explanation=risk_explanation,
        excerpt=excerpt,
    )
    response = client.generate(prompt)
    if not response:
        return UNKNOWN
    candidate = response.strip().split()[0].strip(",.;:`'\"").lower()
    if candidate in PATTERN_CODES:
        return candidate
    for code in PATTERN_CODES:
        if code in response.lower():
            return code
    return UNKNOWN


def derive_pattern(
    *,
    clause_id: str,
    clause_type: str,
    risk_explanation: str,
    clause_text_excerpt: str = "",
    config: Optional[dict] = None,
    client: Optional[GeminiClient] = None,
) -> str:
    """Top-level entry point: tier-1 first, then tier-2 if enabled.

    Logs every tier-2 invocation at INFO with clause_id and resulting code
    (decision #2 in docs/STAGE4_DECISION_LOG.md).
    """
    code = derive_tier1(clause_type, risk_explanation)
    if code != UNKNOWN:
        return code

    cfg = config or {}
    if not _is_tier2_enabled(cfg):
        return UNKNOWN

    if client is None:
        from src.stage4_report_gen.llm_client import get_default_client
        client = get_default_client()

    result = derive_tier2(
        clause_id=clause_id,
        clause_type=clause_type,
        risk_explanation=risk_explanation,
        clause_text_excerpt=clause_text_excerpt,
        client=client,
    )
    logger.info(
        "pattern_deriver tier-2 invoked: clause_id=%s -> pattern=%s",
        clause_id, result,
    )
    return result
