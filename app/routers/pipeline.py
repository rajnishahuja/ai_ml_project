import asyncio
import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.stage1_extract_svc import run_full_pipeline

router = APIRouter()


@router.post(
    "/analyze",
    summary="Full pipeline — extract clauses, assess risk, generate report",
    description=(
        "Upload a contract (PDF, DOCX, or TXT). "
        "Runs Stage 1 (DeBERTa clause extraction) → Stage 3 (Ens-F + LangGraph risk agent) "
        "→ Stage 4 (report generation). "
        "Returns a complete risk report with executive summary and per-clause breakdown. "
        "Typical processing time: 5–15 minutes depending on contract length."
    ),
)
async def analyze_contract(file: UploadFile = File(...)):
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename.")
    
    filename = file.filename
    doc_id = Path(filename).stem
    temp_path = data_dir / filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")

    try:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, run_full_pipeline, str(temp_path), doc_id
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path.exists():
            temp_path.unlink()
