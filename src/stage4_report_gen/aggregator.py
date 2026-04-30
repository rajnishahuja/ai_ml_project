"""Stage 4 deterministic aggregator — no LLM calls.

Takes Stage 3 output (a list of risk-assessed clauses) plus optional
contract metadata, and produces:

  - List[ClauseReport]                 — alias-normalized + pattern-derived
  - List[MissingProtection]            — checklist hits
  - ScoreBreakdown                     — base + missing boost + final
  - inferred contract_type             — when metadata absent
  - statistics dict                    — counts by risk level

ClauseReport.recommendation is left as a placeholder Recommendation here;
the recommender fills it in. ClauseReport.polished_explanation is left
empty; the explainer fills it in.

Per docs/STAGE4_DECISION_LOG.md decision #1, when total_clauses == 0 the
ScoreBreakdown carries note="no clauses assessed" and final_score=0.0.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.common.schema import (
    ClauseReport,
    ExecutiveSummaryDigest,
    MissingProtection,
    Recommendation,
    ScoreBreakdown,
)
from src.stage4_report_gen import pattern_deriver
from src.stage4_report_gen.llm_client import GeminiClient

logger = logging.getLogger(__name__)


_DATA_DIR = Path(__file__).parent
METADATA_CLAUSE_TYPES: frozenset[str] = frozenset({
    "Document Name", "Parties", "Agreement Date",
    "Effective Date", "Expiration Date",
})

# Importance levels that contribute to missing-protection score boost.
_BOOST_IMPORTANCE: frozenset[str] = frozenset({"critical", "important"})

_VALID_RISK_LEVELS: frozenset[str] = frozenset({"HIGH", "MEDIUM", "LOW"})


# ---------------------------------------------------------------------------
# YAML loaders (cached on module import)
# ---------------------------------------------------------------------------

def _load_yaml(filename: str) -> Any:
    path = _DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


_ALIASES: dict[str, str] = _load_yaml("clause_type_aliases.yaml")
_MISSING_PROTECTIONS: dict[str, list[dict]] = _load_yaml("missing_protections.yaml")
_ALIAS_INDEX: dict[str, str] = {
    key.lower().strip(): value for key, value in _ALIASES.items()
}


# ---------------------------------------------------------------------------
# Schema-agnostic clause field reader
# ---------------------------------------------------------------------------

def _get(clause: Any, *names: str, default: Any = None) -> Any:
    """Read first attribute or dict key that exists and is not None."""
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
# Clause type normalization
# ---------------------------------------------------------------------------

def normalize_clause_type(raw: str) -> str:
    """Map a free-form Stage 3 clause_type to the canonical CUAD type.

    Returns the original string verbatim if no alias matches — the
    aggregator still passes such clauses through; recommender/missing-
    protection logic will use the unmapped string.
    """
    if not raw:
        return ""
    key = raw.strip().lower()
    return _ALIAS_INDEX.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Risk level normalization
# ---------------------------------------------------------------------------

def _normalize_risk_level(raw: Any) -> str:
    if not raw:
        return "LOW"
    upper = str(raw).upper().strip()
    return upper if upper in _VALID_RISK_LEVELS else "LOW"


# ---------------------------------------------------------------------------
# Contract-type inference
# ---------------------------------------------------------------------------

# Heuristic: clause-type signatures that suggest a contract category.
_CONTRACT_TYPE_SIGNATURES: tuple[tuple[str, frozenset[str]], ...] = (
    ("license", frozenset({
        "License Grant", "Irrevocable Or Perpetual License",
        "Non-Transferable License", "Affiliate License-Licensee",
        "Affiliate License-Licensor", "Source Code Escrow",
    })),
    ("distribution", frozenset({
        "Exclusivity", "Minimum Commitment", "Volume Restriction",
        "Most Favored Nation", "Price Restrictions",
    })),
    ("services", frozenset({
        "Post-Termination Services", "Audit Rights",
        "IP Ownership Assignment",
    })),
    ("vendor", frozenset({
        "Audit Rights", "Insurance", "Warranty Duration",
    })),
    ("nda", frozenset({
        "Non-Disparagement",
    })),
    ("employment", frozenset({
        "Non-Compete", "Non-Solicit Of Employees",
    })),
)


def infer_contract_type(present_clause_types: set[str]) -> str:
    """Return the contract category whose signature most overlaps with
    the present clause types. Defaults to 'universal' (the always-applied
    checklist) when no category dominates.
    """
    best_label = "universal"
    best_score = 0
    for label, signature in _CONTRACT_TYPE_SIGNATURES:
        overlap = len(signature & present_clause_types)
        if overlap > best_score:
            best_score = overlap
            best_label = label
    return best_label if best_score > 0 else "universal"


# ---------------------------------------------------------------------------
# Missing-protection detection
# ---------------------------------------------------------------------------

def find_missing_protections(
    present_clause_types: Iterable[str],
    contract_type: str,
) -> list[MissingProtection]:
    """Apply the universal checklist + the contract-type checklist and
    return one MissingProtection per item not in present_clause_types."""
    present = set(present_clause_types)
    missing: list[MissingProtection] = []
    seen: set[str] = set()

    for category in ("universal", contract_type):
        for item in _MISSING_PROTECTIONS.get(category, []) or []:
            ct = item["clause_type"]
            if ct in present or ct in seen:
                continue
            seen.add(ct)
            missing.append(MissingProtection(
                clause_type=ct,
                importance=item["importance"],
                rationale=str(item["rationale"]).strip(),
            ))
    return missing


# ---------------------------------------------------------------------------
# Score breakdown
# ---------------------------------------------------------------------------

def compute_score_breakdown(
    high_count: int,
    medium_count: int,
    low_count: int,
    missing: list[MissingProtection],
    *,
    high_weight: int = 3,
    medium_weight: int = 2,
    low_weight: int = 1,
    scale: float = 10.0 / 3.0,
    boost_per: float = 0.5,
    cap: float = 10.0,
) -> ScoreBreakdown:
    total = high_count + medium_count + low_count
    boost_count = sum(1 for m in missing if m.importance in _BOOST_IMPORTANCE)
    if total == 0:
        return ScoreBreakdown(
            high_count=0,
            medium_count=0,
            low_count=0,
            base_score=0.0,
            missing_critical_or_important=boost_count,
            missing_boost=0.0,
            final_score=0.0,
            note="no clauses assessed",
        )
    base = (
        high_count * high_weight
        + medium_count * medium_weight
        + low_count * low_weight
    ) / total * scale
    boost = boost_count * boost_per
    final = min(round(base + boost, 2), cap)
    return ScoreBreakdown(
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        base_score=round(base, 2),
        missing_critical_or_important=boost_count,
        missing_boost=round(boost, 2),
        final_score=final,
        note="",
    )


# ---------------------------------------------------------------------------
# Per-clause normalization → ClauseReport
# ---------------------------------------------------------------------------

def _placeholder_recommendation() -> Recommendation:
    """Recommender fills these in for HIGH/MEDIUM. LOW gets the placeholder."""
    return Recommendation(
        text="",
        market_standard="",
        fallback_position="",
        priority="LOW",
        match_level="universal",
    )


def _similar_to_dict(s: Any) -> dict:
    if isinstance(s, dict):
        return s
    return {
        "text": _get(s, "text", default=""),
        "risk_level": _get(s, "risk_level", default=""),
        "similarity": _get(s, "similarity", default=0.0),
    }


def to_clause_report(
    raw_clause: Any,
    *,
    config: dict,
    client: GeminiClient | None = None,
    similar_max: int = 5,
) -> ClauseReport:
    """Build a ClauseReport from one Stage 3 risk-assessed clause."""
    clause_id = str(_get(raw_clause, "clause_id", default="") or "")
    document_id = str(_get(raw_clause, "document_id", default="") or "")
    clause_text = str(_get(raw_clause, "clause_text", default="") or "")
    clause_type_original = str(_get(raw_clause, "clause_type", default="") or "")
    clause_type = normalize_clause_type(clause_type_original)
    risk_level = _normalize_risk_level(_get(raw_clause, "risk_level"))
    risk_explanation = str(
        _get(raw_clause, "risk_explanation", "risk_reason", default="") or ""
    )
    confidence_raw = _get(raw_clause, "confidence", default=0.0)
    try:
        confidence = float(confidence_raw) if confidence_raw is not None else 0.0
    except (TypeError, ValueError):
        confidence = 0.0
    overridden = bool(_get(raw_clause, "overridden", default=False))

    similar_raw = _get(raw_clause, "similar_clauses", default=[]) or []
    similar = [_similar_to_dict(s) for s in similar_raw[:similar_max]]

    excerpt = clause_text[:200]
    risk_pattern = pattern_deriver.derive_pattern(
        clause_id=clause_id,
        clause_type=clause_type,
        risk_explanation=risk_explanation,
        clause_text_excerpt=excerpt,
        config=config,
        client=client,
    )

    return ClauseReport(
        clause_id=clause_id,
        document_id=document_id,
        clause_text=clause_text,
        clause_type=clause_type,
        clause_type_original=clause_type_original,
        risk_level=risk_level,
        risk_pattern=risk_pattern,
        risk_explanation=risk_explanation,
        polished_explanation="",
        recommendation=_placeholder_recommendation(),
        confidence=confidence,
        overridden=overridden,
        similar_clauses=similar,
    )


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

def aggregate(
    raw_clauses: Iterable[Any],
    metadata: dict[str, str] | None = None,
    *,
    config: dict | None = None,
    client: GeminiClient | None = None,
) -> dict:
    """Run the full aggregation pipeline.

    Returns a dict with:
      reports                — list[ClauseReport] (excluding metadata clause types)
      metadata_block         — dict[str, str] from input + any metadata clauses we saw
      contract_type          — str (inferred or from metadata)
      contract_type_inferred — bool
      missing_protections    — list[MissingProtection]
      score_breakdown        — ScoreBreakdown
      statistics             — dict[str, int] (counts by risk level + total)
    """
    cfg = config or {}
    aggregator_cfg = cfg.get("aggregator", {})
    similar_max = int(aggregator_cfg.get("similar_clauses_max_passthrough", 5))
    inference_enabled = bool(aggregator_cfg.get("contract_type_inference", True))

    rscfg = cfg.get("risk_score", {})
    score_kwargs = {
        "high_weight": int(rscfg.get("high_weight", 3)),
        "medium_weight": int(rscfg.get("medium_weight", 2)),
        "low_weight": int(rscfg.get("low_weight", 1)),
        "scale": float(rscfg.get("scale", 10.0 / 3.0)),
        "boost_per": float(rscfg.get("missing_protection_boost_per", 0.5)),
        "cap": float(rscfg.get("cap", 10.0)),
    }

    seen_metadata: dict[str, str] = {}
    reports: list[ClauseReport] = []

    for raw in raw_clauses:
        ct_original = str(_get(raw, "clause_type", default="") or "")
        ct_canonical = normalize_clause_type(ct_original)
        if ct_canonical in METADATA_CLAUSE_TYPES:
            text_value = str(_get(raw, "clause_text", default="") or "").strip()
            if text_value and ct_canonical not in seen_metadata:
                seen_metadata[ct_canonical] = text_value
            continue
        reports.append(to_clause_report(
            raw, config=cfg, client=client, similar_max=similar_max,
        ))

    incoming_metadata = dict(metadata or {})
    metadata_provided = bool(incoming_metadata)
    metadata_block: dict[str, str] = {}
    for key in ("Document Name", "Parties", "Agreement Date",
                "Effective Date", "Expiration Date"):
        metadata_block[key] = (
            incoming_metadata.get(key)
            or seen_metadata.get(key)
            or "—"
        )

    present_canonical = {r.clause_type for r in reports}

    contract_type_from_metadata = (
        incoming_metadata.get("contract_type")
        or incoming_metadata.get("Contract Type")
    )
    if contract_type_from_metadata:
        contract_type = str(contract_type_from_metadata).lower()
        contract_type_inferred = False
    elif inference_enabled:
        contract_type = infer_contract_type(present_canonical)
        contract_type_inferred = True
    else:
        contract_type = "universal"
        contract_type_inferred = True

    missing = find_missing_protections(present_canonical, contract_type)

    counts = Counter(r.risk_level for r in reports)
    score = compute_score_breakdown(
        high_count=counts.get("HIGH", 0),
        medium_count=counts.get("MEDIUM", 0),
        low_count=counts.get("LOW", 0),
        missing=missing,
        **score_kwargs,
    )

    statistics = {
        "high": counts.get("HIGH", 0),
        "medium": counts.get("MEDIUM", 0),
        "low": counts.get("LOW", 0),
        "total": len(reports),
    }

    return {
        "reports": reports,
        "metadata_block": metadata_block,
        "metadata_provided": metadata_provided,
        "contract_type": contract_type,
        "contract_type_inferred": contract_type_inferred,
        "missing_protections": missing,
        "score_breakdown": score,
        "statistics": statistics,
    }


# ---------------------------------------------------------------------------
# Executive-summary digest builder
# ---------------------------------------------------------------------------

def build_executive_digest(
    aggregation: dict,
    *,
    top_n: int = 5,
    excerpt_max: int = 200,
) -> ExecutiveSummaryDigest:
    """Hard rule #1 boundary — this is the ONLY thing the LLM sees for the
    contract-level summary. Excerpts are truncated to excerpt_max chars."""
    reports: list[ClauseReport] = aggregation["reports"]
    score: ScoreBreakdown = aggregation["score_breakdown"]
    statistics = dict(aggregation["statistics"])
    statistics["overall_risk_score"] = score.final_score
    statistics["base_score"] = score.base_score
    statistics["missing_boost"] = score.missing_boost

    high = [r for r in reports if r.risk_level == "HIGH"]
    high_sorted = sorted(high, key=lambda r: r.confidence, reverse=True)
    top_high_risk = [
        {
            "canonical_clause_type": r.clause_type,
            "risk_pattern": r.risk_pattern,
            "risk_explanation": r.risk_explanation,
            "clause_text_excerpt": (r.clause_text or "")[:excerpt_max],
        }
        for r in high_sorted[:top_n]
    ]

    return ExecutiveSummaryDigest(
        metadata=dict(aggregation["metadata_block"]),
        statistics=statistics,
        top_high_risk=top_high_risk,
        missing_protections=[m.clause_type for m in aggregation["missing_protections"]],
        metadata_provided=bool(aggregation["metadata_provided"]),
    )
