from fastapi import APIRouter, HTTPException

# Import the core storage parsing utilities
from src.stage3_risk_agent.embeddings import get_all_document_ids, get_document_chunks

router = APIRouter()

@router.get("/")
async def list_all_documents():
    """
    Returns a list of all unique UUID Documents currently saved 
    in the global FAISS index.
    """
    try:
        doc_ids = get_all_document_ids()
        return {
            "status": "success",
            "total_documents": len(doc_ids),
            "document_ids": doc_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{document_id}")
async def get_document(document_id: str):
    """
    Retreives physical Text Chunks and properties associated with
    a specific UUID to prove the vector store captured the data accurately.
    """
    try:
        chunks = get_document_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found in index.")
            
        return {
            "status": "success",
            "document_id": document_id,
            "total_chunks": len(chunks),
            "data": chunks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
