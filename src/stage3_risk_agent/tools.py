"""
LangGraph-compatible tools for the Stage 3 risk assessment agent.

Tools:
  1. precedent_search  — vector RAG over the labeled corpus (FAISS).
                          Implementation lives in `embeddings.py`; this file
                          only owns the agent-facing wrapper signature.
  2. contract_search   — structured same-document lookup. Returns every
                          non-metadata clause Stage 1+2 extracted for a given
                          contract. No embeddings, no RAG, no LLM calls.

Per ARCHITECTURE.md (decision 2026-04-23): contract_search is a structured
lookup, not a similarity search. The output feeds the Mistral-7B reasoning
agent on the low-confidence path so it can resolve same-contract
cross-references (e.g. "IP was already assigned in another clause").
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
    all_clauses: Iterable[Any] | None = None,
    index_path: str = DEFAULT_INDEX_PATH,
    spans_path: str = DEFAULT_SPANS_PATH,
    include_metadata: bool = False,
) -> list[dict]:
    """Return every non-metadata clause from the same contract.

    Three resolution paths, tried in order:

      1. **In-memory mode** (preferred at agent runtime): pass `all_clauses`.
         Typically the LangGraph state's `extracted_clauses` list. Avoids
         touching disk inside the tool call.

      2. **Index mode** (default for offline / batch use): load the
         pre-computed `contract_clause_index.json` (built by
         `scripts/build_contract_clause_index.py`) and do an O(1) dict
         lookup by `document_id`. Each per-contract entry already carries
         the `is_metadata` flag, so filtering is cheap.

      3. **Corpus fallback**: if the index file is missing, load the raw
         `all_positive_spans.json` and filter on the fly. Slower, but keeps
         the tool functional in environments where the index hasn't been
         generated yet.

    Args:
        document_id: Contract identifier (e.g. "contract_042" or the CUAD
            slug used as the `contract` field in `all_positive_spans.json`).
        all_clauses: Optional in-memory list of clause objects (dicts,
            dataclasses, or pydantic models). When provided, the function
            does NOT touch disk.
        index_path: Path to the pre-computed clause index. Default points
            at `data/processed/contract_clause_index.json`.
        spans_path: Path to the raw spans file (fallback only).
        include_metadata: If True, include the 5 metadata clause types in
            the result. Default False — they route to the Stage 4 header.

    Returns:
        List of dicts with keys:
            - clause_id    : str
            - document_id  : str
            - clause_type  : str
            - clause_text  : str
            - start_pos    : int
            - is_metadata  : bool   (only present when loaded from the index)

        Empty list if no matches are found. Never raises on a missing
        document — the agent treats `[]` as "no evidence" and falls back.
    """
    if not document_id:
        logger.warning("contract_search: empty document_id; returning [].")
        return []

    # 1. In-memory mode — used by the LangGraph agent at runtime.
    if all_clauses is not None:
        siblings = [
            c for c in (_coerce_clause(x) for x in all_clauses)
            if c["document_id"] == document_id
            and (include_metadata or c["clause_type"] not in METADATA_CLAUSE_TYPES)
        ]
        if not siblings:
            logger.debug(
                "contract_search: no clauses for %r in supplied list.",
                document_id,
            )
        return siblings

    # 2. Index mode — fast dict lookup against the pre-built index.
    try:
        index = _load_index(index_path)
    except FileNotFoundError:
        logger.info(
            "Index %s not found — falling back to spans file %s. "
            "Run scripts/build_contract_clause_index.py to build it.",
            index_path, spans_path,
        )
        index = None

    if index is not None:
        entry = index.get(document_id)
        if entry is None:
            logger.debug("contract_search: %r not in index.", document_id)
            return []
        # Add document_id to each row (the per-clause objects in the index
        # don't carry it — it's the dict key) and filter metadata.
        return [
            {**c, "document_id": document_id}
            for c in entry["clauses"]
            if include_metadata or not c.get("is_metadata", False)
        ]

    # 3. Corpus fallback — filter the raw spans corpus on the fly.
    candidates = _load_corpus(spans_path)
    siblings = [
        c for c in candidates
        if c["document_id"] == document_id
        and (include_metadata or not c.get("is_metadata", False))
    ]
    if not siblings:
        logger.debug(
            "contract_search: no clauses for %r in spans corpus.",
            document_id,
        )
    return siblings


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
