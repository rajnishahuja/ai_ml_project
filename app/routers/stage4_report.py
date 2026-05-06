import asyncio

from fastapi import APIRouter, HTTPException

from app.schemas.requests import Stage4Request
from app.services.stage4_report_svc import run_stage4

router = APIRouter()


@router.post(
    "/report",
    summary="Stage 4 — generate risk report from assessed clauses",
    description=(
        "Accepts a list of risk-assessed clauses (output of Stage 3). "
        "Generates executive summary via LLM, attaches per-clause recommendations, "
        "and returns a complete RiskReport with overall risk score."
    ),
)
async def generate_report(request: Stage4Request):
    if not request.assessed_clauses:
        raise HTTPException(status_code=422, detail="assessed_clauses list must not be empty")
    try:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None, run_stage4, request.document_id, request.assessed_clauses
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
