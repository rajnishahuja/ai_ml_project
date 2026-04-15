from app.schemas.domain import ExtractedClause
from app.services.stage1_extract_svc import get_extraction_service
from src.workflow.state import RiskAnalysisState


def node_extract_clauses(state: RiskAnalysisState):
    """
    STAGE 1 & STAGE 2 — Node A
    Uses the locally hosted DeBERTa QA model to extract and classify
    legal clauses from the raw contract text.
    Output: List[ExtractedClause] -> passed to Node C (risk classifier)
    """
    print(f"🔄 [NODE A | Stage 1+2] DeBERTa extracting clauses for doc: {state['document_id']}")

    extraction_service = get_extraction_service()
    raw_clauses = extraction_service.infer_from_text(
        text=state["contract_text"],
        doc_id=state["document_id"]
    )
    extracted_models = [ExtractedClause(**clause) for clause in raw_clauses]

    print(f"✅ [NODE A | Stage 1+2] Extracted {len(extracted_models)} clauses.")
    return {"extracted_clauses": extracted_models}
