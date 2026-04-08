"""
FAISS index builder and query interface for clause embeddings.

Uses all-MiniLM-L6-v2 for encoding clause text into dense vectors.
"""

import logging

from src.common.schema import SimilarClause

logger = logging.getLogger(__name__)


def build_faiss_index(
    clauses: list[dict],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    output_path: str = "data/faiss_index/clauses.index",
) -> None:
    """Build a FAISS index from labeled clause data.

    Each clause is embedded using the sentence-transformer model.
    The index and metadata are saved to disk.

    Args:
        clauses: List of clause dicts with 'clause_text' and 'risk_level' keys.
        embedding_model: HuggingFace sentence-transformer model name.
        output_path: Path to save the FAISS index file.
    """
    raise NotImplementedError


def query_similar(
    clause_text: str,
    index_path: str = "data/faiss_index/clauses.index",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
) -> list[SimilarClause]:
    """Query the FAISS index for clauses similar to the input.

    Args:
        clause_text: Text of the clause to search for.
        index_path: Path to the built FAISS index.
        embedding_model: Model for encoding the query.
        top_k: Number of results to return.

    Returns:
        List of SimilarClause with text, risk_level, and similarity.
    """
    raise NotImplementedError
