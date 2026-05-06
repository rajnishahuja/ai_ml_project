from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="FAISS index info")
async def faiss_index_info():
    """Returns metadata about the FAISS precedent index."""
    return {
        "description": "Static FAISS precedent index (training corpus)",
        "vectors": 3398,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "index_type": "IndexFlatIP",
        "note": (
            "This index contains labeled training clauses used for precedent search. "
            "It is not a runtime document store."
        ),
    }
