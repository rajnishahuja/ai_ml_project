"""
Clause embeddings — FAISS index builder and query interface.

Two public functions:
  build_index()   — offline, called once by scripts/build_faiss_index.py
  query_index()   — runtime, called by tools.py precedent_search

Index type: IndexFlatIP over L2-normalised vectors = exact cosine similarity.
At 4,276 vectors this is microseconds per query — no ANN approximation needed.

Metadata is stored as a parallel JSON array (data/faiss_index/clauses_meta.json).
Position i in the JSON corresponds to vector i in the FAISS index.
"""

import json
import logging
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.common.schema import SimilarClause

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model_cache: dict[str, SentenceTransformer] = {}   # model_name → SentenceTransformer
_faiss_cache: dict[str, tuple] = {}                 # index_path → (faiss.Index, list[dict])


def _get_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info("Loading embedding model: %s", model_name)
        try:
            _model_cache[model_name] = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def _get_index(index_path: str) -> tuple:
    """Load FAISS index + metadata once, cache in memory for subsequent calls."""
    global _faiss_cache
    if index_path not in _faiss_cache:
        logger.info("Loading FAISS index from %s", index_path)
        index = faiss.read_index(index_path)
        with open(_meta_path(index_path)) as f:
            metadata = json.load(f)
        _faiss_cache[index_path] = (index, metadata)
    return _faiss_cache[index_path]


def _meta_path(index_path: str) -> Path:
    return Path(index_path).with_suffix(".json")


# ---------------------------------------------------------------------------
# Build (offline, run once)
# ---------------------------------------------------------------------------

def build_index(training_data_path: str, index_path: str,
                splits_path: str | None = None,
                model_name: str = DEFAULT_MODEL) -> None:
    """Embed labeled clauses from training_dataset.json and write FAISS index.

    Only indexes the train split when splits_path is provided — prevents test
    clauses from appearing in FAISS results during evaluation (data leakage).

    Skips rows with label=None.
    Writes two files:
      <index_path>          — FAISS IndexFlatIP binary
      <index_path>.json     — parallel metadata array

    Args:
        training_data_path: Path to data/processed/training_dataset.json.
        index_path: Destination path for the FAISS index file.
        splits_path: Path to splits.json; if given, only train-split rows are indexed.
    """
    logger.info("Loading training data from %s", training_data_path)
    with open(training_data_path) as f:
        rows = json.load(f)

    if splits_path:
        with open(splits_path) as f:
            splits = json.load(f)
        train_nums = set(splits["train"])
        before = len(rows)
        rows = [r for r in rows if r.get("row_num") in train_nums]
        logger.info("Filtered to train split: %d → %d rows", before, len(rows))

    # Drop rows with no resolved label
    skipped = sum(1 for r in rows if r.get("label") is None)
    rows = [r for r in rows if r.get("label") is not None]
    logger.info("Indexing %d clauses (skipped %d None-label rows)", len(rows), skipped)

    texts = [r["clause_text"] for r in rows]
    metadata = [
        {
            "clause_text": r["clause_text"],
            "clause_type": r["clause_type"],
            "risk_level": r["label"],
        }
        for r in rows
    ]

    logger.info("Encoding %d clauses with %s ...", len(texts), model_name)
    model = _get_model(model_name)
    vectors = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)
    vectors = vectors.astype(np.float32)

    dim = vectors.shape[1]
    logger.info("Building FAISS IndexFlatIP (dim=%d) ...", dim)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    logger.info("FAISS index saved → %s (%d vectors)", index_path, index.ntotal)

    meta_path = _meta_path(index_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    logger.info("Metadata saved → %s", meta_path)


# ---------------------------------------------------------------------------
# Query (runtime, called per clause)
# ---------------------------------------------------------------------------

def query_index(clause_text: str, index_path: str, k: int = 5,
                model_name: str = DEFAULT_MODEL) -> list[SimilarClause]:
    """Retrieve the top-k most similar clauses from the FAISS index.

    Args:
        clause_text: The clause text to search for.
        index_path: Path to the FAISS index file.
        k: Number of results to return.

    Returns:
        List of SimilarClause ordered by descending similarity.
    """
    model = _get_model(model_name)
    vector = model.encode([clause_text], convert_to_numpy=True,
                          normalize_embeddings=True).astype(np.float32)

    index, metadata = _get_index(index_path)
    k = min(k, index.ntotal)
    scores, indices = index.search(vector, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        results.append(
            SimilarClause(
                text=m["clause_text"],
                clause_type=m["clause_type"],
                risk_level=m["risk_level"],
                similarity=float(score),
            )
        )
    return results
