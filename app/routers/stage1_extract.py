import os
import shutil
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

# Import the LangGraph Orchestrator and Preprocessing utils
from src.workflow.graph import risk_graph
from src.stage1_extract_classify.preprocessing import preprocess_contract

router = APIRouter()


# Stage 4 writes reports here; the download endpoints read from the same path.
REPORTS_DIR = Path("data/reports")

DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)
PDF_MEDIA_TYPE = "application/pdf"


@router.post("/analyze")
async def analyze_document_with_agent(file: UploadFile = File(...)):
    """
    Upload a contract (PDF, DOCX, TXT) to execute the Full Risk Agent Pipeline.
    This triggers LangGraph which natively handles DeBERTa extraction and FAISS
    encoding completely in parallel!

    Returns the assembled JSON report plus URLs to download the .docx and
    (when available) .pdf renderings written by Stage 4.
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

        report = final_state.get("final_report", {}) or {}

        # Build download URLs for the rendered report files. Only surface
        # the PDF link if Stage 4 actually produced one.
        download_links = {"docx": f"/report/{doc_id}/docx"}
        if report.get("pdf_path"):
            download_links["pdf"] = f"/report/{doc_id}/pdf"

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
            "final_report": report,
            "report_download": download_links,
        }

    except Exception as e:
        print(f"❌ LangGraph Pipeline Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the local storage after returning the JSON report
        if temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------------
# Report download endpoints
# ---------------------------------------------------------------------------

def _resolve_report_path(document_id: str, suffix: str) -> Path:
    """Build and validate the absolute path of a report file.

    Defends against directory traversal: rejects any document_id containing
    path separators or `..`.
    """
    if "/" in document_id or "\\" in document_id or ".." in document_id:
        raise HTTPException(status_code=400, detail="Invalid document_id.")
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / REPORTS_DIR / f"{document_id}{suffix}"


@router.get("/report/{document_id}/docx")
async def download_docx(document_id: str):
    """Download the Stage 4 .docx report for a previously-analyzed contract."""
    path = _resolve_report_path(document_id, ".docx")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"DOCX report not found for document_id={document_id!r}.",
        )
    return FileResponse(
        path=str(path),
        media_type=DOCX_MEDIA_TYPE,
        filename=f"risk_report_{document_id}.docx",
    )


@router.get("/report/{document_id}/pdf")
async def download_pdf(document_id: str):
    """Download the Stage 4 .pdf report. Returns 404 if PDF rendering was skipped."""
    path = _resolve_report_path(document_id, ".pdf")
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"PDF report not available for document_id={document_id!r}. "
                f"Use /report/{document_id}/docx instead."
            ),
        )
    return FileResponse(
        path=str(path),
        media_type=PDF_MEDIA_TYPE,
        filename=f"risk_report_{document_id}.pdf",
    )
