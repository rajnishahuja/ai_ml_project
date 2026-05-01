"""
Stage 3 risk classifier — module-level interface for the agent pipeline.

Wraps scripts/infer.py's RiskClassifier (Ens-F: CE + CORN ensemble).
Adds extract_signing_party() to resolve the signing party from the clause batch
before each predict() call.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.infer import RiskClassifier  # noqa: E402 — re-export for pipeline imports
from src.common.schema import ClauseObject

logger = logging.getLogger(__name__)

__all__ = ["RiskClassifier", "extract_signing_party"]


def extract_signing_party(document_id: str, clauses: list[ClauseObject]) -> str:
    """Find the Parties clause for this contract and return its text.

    Stage 1+2 extracts a 'Parties' clause for most contracts. Its text (e.g.
    'AT&T Inc. and Jane Smith Consulting') is passed to RiskClassifier.predict()
    so DeBERTa can resolve signing-party direction — the #1 driver of HIGH/LOW
    label flips identified in manual review.

    Args:
        document_id: The contract's document_id.
        clauses:     All ClauseObjects extracted from this contract by Stage 1+2.

    Returns:
        Parties clause text, or empty string if not found.
    """
    for clause in clauses:
        if clause.document_id == document_id and clause.clause_type == "Parties":
            return clause.clause_text

    logger.warning(
        "No 'Parties' clause found for document_id=%s — signing_party will be empty",
        document_id,
    )
    return ""
