"""
LangGraph-compatible tools for the Stage 3 risk assessment agent.

Tools:
  1. precedent_search  — vector RAG over the labeled corpus (FAISS).
                          Implementation lives in `embeddings.py`; this file
                          only owns the agent-facing wrapper signature.
  2. contract_search   — structured same-document lookup, filtered by a
                          static `clause_type → related_types` map. Returns
                          only those clauses from the same contract whose
                          type is *legally related* to the target's type.
                          No embeddings, no RAG, no LLM calls.

Per ARCHITECTURE.md (decision 2026-04-23): contract_search is a structured
lookup, not a similarity search. It feeds the Mistral-7B reasoning agent on
the low-confidence path so it can resolve same-contract cross-references
(e.g. "the indemnification clause is uncapped, but the cap-on-liability
clause already limits exposure to 12 months of fees").

The static relations file (`data/reference/clause_type_relations.json`) is
built by `scripts/build_clause_type_relations.py`. It maps each of the 41
CUAD clause types to a curated list of related types so that the agent
sees focused, legally-relevant evidence instead of a 13-clause sibling dump.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# Metadata clause types are excluded from contract_search output. They route
# to the Stage 4 report header, not the risk agent. Source: ARCHITECTURE.md
# §"Stage 3 Training Data Pipeline" / "Metadata routing".
METADATA_CLAUSE_TYPES: frozenset[str] = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

DEFAULT_INDEX_PATH = "data/processed/contract_clause_index.json"
DEFAULT_SPANS_PATH = "data/processed/all_positive_spans.json"
DEFAULT_RELATIONS_PATH = "data/reference/clause_type_relations.json"


@lru_cache(maxsize=2)
def _load_relations(relations_path: str) -> dict[str, frozenset[str]]:
    """Load and cache the static clause_type → related_types map.

    Each value is converted to a frozenset for O(1) membership checks at
    filter time. Built by `scripts/build_clause_type_relations.py`.
    """
    path = Path(relations_path)
    if not path.exists():
        raise FileNotFoundError(f"Relations file not found: {relations_path}")
    with open(path) as f:
        raw = json.load(f)
    return {key: frozenset(value) for key, value in raw.items()}


@lru_cache(maxsize=2)
def _load_index(index_path: str) -> dict:
    """Load and cache the pre-computed contract → clauses index.

    The index is a dict keyed by `document_id`; each value carries the
    ordered list of clauses for that contract (metadata included, with an
    `is_metadata` flag). Built by `scripts/build_contract_clause_index.py`.
    """
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=2)
def _load_corpus(spans_path: str) -> tuple[dict, ...]:
    """Fallback loader: read the raw spans file and normalize on the fly.

    Used only when the pre-built index is unavailable. Returned tuple is
    hashable so it can sit inside lru_cache. Each entry uses normalized
    keys (`document_id` / `clause_id` / `start_pos`).
    """
    path = Path(spans_path)
    if not path.exists():
        raise FileNotFoundError(f"Spans file not found: {spans_path}")

    with open(path) as f:
        raw = json.load(f)

    normalized = []
    for span in raw:
        clause_type = span.get("clause_type", "")
        normalized.append({
            "clause_id": span.get("clause_id") or span.get("id", ""),
            "document_id": span.get("document_id") or span.get("contract", ""),
            "clause_type": clause_type,
            "clause_text": span.get("clause_text", ""),
            "start_pos": span.get("start_pos") or span.get("answer_start", 0),
            "is_metadata": clause_type in METADATA_CLAUSE_TYPES,
        })
    return tuple(normalized)


def _coerce_clause(clause: Any) -> dict:
    """Convert a clause object (dataclass / pydantic / dict) into a flat dict."""
    if isinstance(clause, dict):
        src = clause
    elif hasattr(clause, "model_dump"):  # pydantic v2
        src = clause.model_dump()
    elif hasattr(clause, "to_dict"):
        src = clause.to_dict()
    else:
        src = {k: getattr(clause, k) for k in (
            "clause_id", "document_id", "clause_type", "clause_text",
        ) if hasattr(clause, k)}

    return {
        "clause_id": src.get("clause_id") or src.get("id", ""),
        "document_id": src.get("document_id") or src.get("contract", ""),
        "clause_type": src.get("clause_type", ""),
        "clause_text": src.get("clause_text", ""),
        "start_pos": src.get("start_pos") or src.get("answer_start", 0),
    }


def contract_search(
    document_id: str,
    clause_type: str | None = None,
    *,
    all_clauses: Iterable[Any] | None = None,
    index_path: str = DEFAULT_INDEX_PATH,
    spans_path: str = DEFAULT_SPANS_PATH,
    relations_path: str = DEFAULT_RELATIONS_PATH,
    include_metadata: bool = False,
) -> list[dict]:
    """Return clauses from the same contract whose type is *related* to the
    target clause's type, per the static `clause_type_relations.json` map.

    Filtering pipeline:

        1. Pick the contract's clauses (in-memory list, pre-built index, or
           raw spans fallback).
        2. If `clause_type` is supplied, look up its related types in the
           static relations file and keep only clauses of those types.
           Clauses of the *same* type as the target are also included
           (relevant when a contract has multiple instances of the same
           clause type — e.g. two indemnification provisions for different
           breach categories).
        3. Drop metadata types (Document Name, Parties, etc.) unless
           `include_metadata=True`.

    If `clause_type` is `None` or unknown to the relations file, the type
    filter is skipped — the function returns *all* non-metadata siblings,
    matching the pre-relations behavior. This keeps backward compat with
    callers / agents that don't have a clause_type to pass.

    Resolution paths for the contract data, tried in order:

      1. **In-memory mode** (preferred at runtime): `all_clauses` keyword.
      2. **Index mode** (default offline): pre-built clause index file.
      3. **Corpus fallback**: raw spans file if the index is missing.

    Args:
        document_id: Contract identifier (matches the `contract` field in
            `all_positive_spans.json`).
        clause_type: Optional CUAD clause type of the clause being assessed
            (e.g. "Indemnification" — but note CUAD's 41 types do not
            include that one; see `data/reference/cuad_category_descriptions.csv`).
            When provided, only clauses whose type is in
            `relations[clause_type] ∪ {clause_type}` are returned.
        all_clauses: Optional in-memory list (LangGraph runtime).
        index_path: Path to the pre-built clause index.
        spans_path: Path to the raw spans file (fallback only).
        relations_path: Path to the clause-type relations map.
        include_metadata: Include metadata clause types if True.

    Returns:
        List of dicts: clause_id, document_id, clause_type, clause_text,
        start_pos, is_metadata. Empty list on unknown document_id, missing
        relations data, or no matching siblings. Never raises.
    """
    if not document_id:
        logger.warning("contract_search: empty document_id; returning [].")
        return []

    # ----- Step 1: pick the contract's clauses -----
    contract_clauses = _resolve_contract_clauses(
        document_id=document_id,
        all_clauses=all_clauses,
        index_path=index_path,
        spans_path=spans_path,
        include_metadata=include_metadata,
    )
    if not contract_clauses:
        return []

    # ----- Step 2: apply clause_type relations filter -----
    if clause_type:
        allowed_types = _allowed_types_for(clause_type, relations_path)
        if allowed_types is None:
            logger.debug(
                "contract_search: clause_type=%r not in relations file; "
                "returning all non-metadata siblings.",
                clause_type,
            )
        else:
            before = len(contract_clauses)
            contract_clauses = [
                c for c in contract_clauses
                if c["clause_type"] in allowed_types
            ]
            logger.debug(
                "contract_search: filtered %d → %d clauses by relations[%r].",
                before, len(contract_clauses), clause_type,
            )

    return contract_clauses


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_contract_clauses(
    *,
    document_id: str,
    all_clauses: Iterable[Any] | None,
    index_path: str,
    spans_path: str,
    include_metadata: bool,
) -> list[dict]:
    """Return all clauses for `document_id` (no clause_type filter applied)."""
    # 1. In-memory mode
    if all_clauses is not None:
        return [
            c for c in (_coerce_clause(x) for x in all_clauses)
            if c["document_id"] == document_id
            and (include_metadata or c["clause_type"] not in METADATA_CLAUSE_TYPES)
        ]

    # 2. Index mode
    try:
        index = _load_index(index_path)
    except FileNotFoundError:
        logger.info(
            "Index %s not found — falling back to spans file %s.",
            index_path, spans_path,
        )
        index = None

    if index is not None:
        entry = index.get(document_id)
        if entry is None:
            logger.debug("contract_search: %r not in index.", document_id)
            return []
        return [
            {**c, "document_id": document_id}
            for c in entry["clauses"]
            if include_metadata or not c.get("is_metadata", False)
        ]

    # 3. Corpus fallback
    candidates = _load_corpus(spans_path)
    return [
        c for c in candidates
        if c["document_id"] == document_id
        and (include_metadata or not c.get("is_metadata", False))
    ]


def _allowed_types_for(
    clause_type: str,
    relations_path: str,
) -> frozenset[str] | None:
    """Resolve `clause_type` → set of allowed types (related ∪ self).

    Returns None if the relations file is missing or the clause_type is not
    present in it — caller treats that as "no filter, return everything".
    """
    try:
        relations = _load_relations(relations_path)
    except FileNotFoundError:
        logger.warning(
            "Relations file %s not found — clause_type filter disabled. "
            "Run scripts/build_clause_type_relations.py to generate it.",
            relations_path,
        )
        return None

    if clause_type not in relations:
        return None

    return relations[clause_type] | {clause_type}


# ---------------------------------------------------------------------------
# precedent_search — placeholder. Real implementation lives in embeddings.py
# (FAISS index over the 4,410 risk-relevant labeled clauses).
# ---------------------------------------------------------------------------

def precedent_search(
    clause_text: str,
    index_path: str,
    top_k: int = 5,
) -> list[dict]:
    """Retrieve top-K similar labeled clauses from the FAISS index.

    Args:
        clause_text: Query clause text.
        index_path: Path to the built FAISS index file.
        top_k: Number of similar clauses to return.

    Returns:
        List of dicts: {clause_text, clause_type, risk_level, risk_reason,
        similarity}. See ARCHITECTURE.md §"Tool Definitions".

    Implemented in: `src/stage3_risk_agent/embeddings.py`. Not in scope for
    this branch — only `contract_search` is required here.
    """
    raise NotImplementedError(
        "precedent_search lives in embeddings.py — not implemented in this branch."
    )
