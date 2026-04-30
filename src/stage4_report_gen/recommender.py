"""Recommendation lookup — no LLM.

Consumes recommendations_data.yaml and resolves a Recommendation for each
HIGH or MEDIUM ClauseReport via a four-tier match:

  1. exact       — entries[(canonical_clause_type, risk_pattern)]
  2. type        — entries[(canonical_clause_type, "*")] from type_fallbacks
  3. risk_level  — generic.HIGH or generic.MEDIUM
  4. universal   — generic.UNIVERSAL

LOW-risk clauses skip recommendation lookup; they retain the placeholder
Recommendation set by the aggregator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.common.schema import ClauseReport, Recommendation

logger = logging.getLogger(__name__)


_DATA_FILE = Path(__file__).parent / "recommendations_data.yaml"


def _load() -> dict[str, Any]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_DATA = _load()
_ENTRIES: list[dict] = list(_DATA.get("entries", []) or [])
_TYPE_FALLBACKS: dict[str, dict] = dict(_DATA.get("type_fallbacks", {}) or {})
_GENERIC: dict[str, dict] = dict(_DATA.get("generic", {}) or {})
_EXACT_INDEX: dict[tuple[str, str], dict] = {
    (entry["clause_type"], entry["risk_pattern"]): entry
    for entry in _ENTRIES
    if "clause_type" in entry and "risk_pattern" in entry
}


def _from_entry(entry: dict, *, match_level: str) -> Recommendation:
    return Recommendation(
        text=str(entry.get("recommendation", "")).strip(),
        market_standard=str(entry.get("market_standard", "")).strip(),
        fallback_position=str(entry.get("fallback_position", "")).strip(),
        priority=str(entry.get("priority", "MEDIUM")).strip(),
        match_level=match_level,
    )


def lookup(clause_type: str, risk_pattern: str, risk_level: str) -> Recommendation:
    """Resolve a Recommendation for one (clause_type, risk_pattern, risk_level).

    The risk_level is consulted only for the tier-3 generic fallback.
    """
    exact = _EXACT_INDEX.get((clause_type, risk_pattern))
    if exact is not None:
        return _from_entry(exact, match_level="exact")

    type_entry = _TYPE_FALLBACKS.get(clause_type)
    if type_entry is not None:
        return _from_entry(type_entry, match_level="type")

    risk_key = (risk_level or "").upper()
    risk_entry = _GENERIC.get(risk_key)
    if risk_entry is not None:
        return _from_entry(risk_entry, match_level="risk_level")

    universal = _GENERIC.get("UNIVERSAL", {})
    return _from_entry(universal, match_level="universal")


def attach_recommendations(reports: Iterable[ClauseReport]) -> list[ClauseReport]:
    """Mutate-and-return: fill `recommendation` on every HIGH and MEDIUM
    ClauseReport. LOW-risk reports are left untouched.
    """
    results: list[ClauseReport] = []
    coverage = {"exact": 0, "type": 0, "risk_level": 0, "universal": 0, "skipped": 0}
    for report in reports:
        if report.risk_level in ("HIGH", "MEDIUM"):
            rec = lookup(report.clause_type, report.risk_pattern, report.risk_level)
            report.recommendation = rec
            coverage[rec.match_level] += 1
        else:
            coverage["skipped"] += 1
        results.append(report)
    logger.info("recommender coverage: %s", coverage)
    return results
