import asyncio

from fastapi import APIRouter, HTTPException

from app.schemas.requests import Stage3Request
from app.services.stage3_agent_svc import run_stage3

router = APIRouter()


@router.post(
    "/assess",
    summary="Stage 3 — assess risk for a list of clauses",
    description=(
        "Accepts pre-extracted clauses (from Stage 1 or manually provided). "
        "Runs DeBERTa Ens-F classification + LangGraph ReAct agent per clause. "
        "Returns risk level (LOW/MEDIUM/HIGH), explanation, and agent trace for each clause."
    ),
)
async def assess_risk(request: Stage3Request):
    if not request.clauses:
        raise HTTPException(status_code=422, detail="clauses list must not be empty")
    try:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, run_stage3, request.clauses, request.use_contract_search
        )
        return {"assessed_clauses": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
