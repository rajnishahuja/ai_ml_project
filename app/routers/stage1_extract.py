import os
import shutil
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

# Import the LangGraph Orchestrator and Preprocessing utils
from src.workflow.graph import risk_graph
from src.stage1_extract_classify.preprocessing import preprocess_contract

router = APIRouter()


@router.post("/analyze")
async def analyze_document_with_agent(file: UploadFile = File(...)):
    """
    Upload a contract (PDF, DOCX, TXT) to execute the Full Risk Agent Pipeline.
    This triggers LangGraph which natively handles DeBERTa extraction and FAISS
    encoding completely in parallel!
    """
    import uuid

    # Setup paths cleanly using pathlib
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate a cryptographically secure UUID for multi-tenant safety
    doc_id = str(uuid.uuid4())
    # Prepend UUID to the temp path to prevent concurrent buffer overwriting
    temp_path = data_dir / f"{doc_id}_{file.filename}"

    # Save uploaded file to disk securely
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    try:
        # Preprocess text off the main thread to keep FastAPI completely responsive
        loop = asyncio.get_running_loop()
        contract_text = await loop.run_in_executor(
            None, preprocess_contract, str(temp_path), doc_id
        )

        # Kick off the Parallel LangGraph Execution Graph!
        initial_state = {
            "contract_text": contract_text,
            "document_id": doc_id,
            "extracted_clauses": [],
            "faiss_status": "Pending",
            "flagged_clauses": [],
            "risk_assessed_clauses": [],
            "final_report": {}
        }

        # Use ainvoke! LangGraph handles the Node multiprocessing automatically.
        final_state = await risk_graph.ainvoke(initial_state)

        return {
            "status": "success",
            "filename": file.filename,
            "document_id": doc_id,
            "faiss_sync_status": final_state.get("faiss_status"),
            "extracted_clauses": [
                c.model_dump() for c in final_state.get("extracted_clauses", [])
            ],
            "risk_assessed_clauses": [
                c.model_dump() for c in final_state.get("risk_assessed_clauses", [])
            ],
            "final_report": final_state.get("final_report", {})
        }

    except Exception as e:
        print(f"❌ LangGraph Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the local storage after returning the JSON report
        if temp_path.exists():
            temp_path.unlink()
