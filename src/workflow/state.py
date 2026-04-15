from typing import TypedDict, List, Dict, Any
from app.schemas.domain import ExtractedClause, RiskAssessedClause


class RiskAnalysisState(TypedDict):
    """
    Central Nervous System of the E2E pipeline.
    Carries data sequentially from Stage 1 through Stage 4.
    Each node reads from here and writes its outputs back here.
    """
    contract_text: str
    document_id: str

    # [Node A] Stage 1+2 output: DeBERTa extracted clauses
    extracted_clauses: List[ExtractedClause]

    # [Node B] Stage 3 output: FAISS vector DB sync status
    faiss_status: str

    # [Node C] Stage 3 output: DeBERTa risk flags (High/Med/Low) per clause
    flagged_clauses: List[Dict[str, Any]]

    # [Node D] Stage 3 output: Mistral RAG explanations per flagged clause
    risk_assessed_clauses: List[RiskAssessedClause]

    # [Node E] Stage 4 output: Final compiled risk report
    final_report: Dict[str, Any]
