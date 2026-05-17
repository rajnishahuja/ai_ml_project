import os
from pathlib import Path

from src.common.schema import ClauseObject
from src.stage1_extract_classify.model import ClauseExtractorClassifier
from src.stage1_extract_classify.preprocessing import preprocess_contract
from src.stage3_risk_agent.agent import assess_clauses
from src.stage4_report_gen.report_builder import build_report

BASE_DIR = Path(__file__).resolve().parent.parent.parent
STAGE1_MODEL_PATH = "rajnishahuja/cuad-stage1-deberta"
DEFAULT_CE_MODEL   = "rajnishahuja/cuad-risk-deberta-ce-parties"
DEFAULT_CORN_MODEL = "rajnishahuja/cuad-risk-deberta-corn-parties"

_extractor: ClauseExtractorClassifier | None = None


def _get_extractor() -> ClauseExtractorClassifier:
    global _extractor
    if _extractor is None:
        _extractor = ClauseExtractorClassifier(STAGE1_MODEL_PATH)
    return _extractor


def _to_schema_clause(c, document_id: str) -> ClauseObject:
    return ClauseObject(
        clause_id=c.clause_id,
        document_id=document_id,
        clause_text=c.clause_text,
        clause_type=c.clause_type,
        start_pos=c.start_pos,
        end_pos=c.end_pos,
        confidence=c.confidence,
    )


from src.common.pipeline_service import run_end_to_end_pipeline

def run_full_pipeline(file_path: str, doc_id: str) -> dict:
    """Run Stage 1 → Stage 3 → Stage 4 synchronously via unified service."""
    report = run_end_to_end_pipeline(
        contract_path=file_path,
        doc_id=doc_id,
        stage1_model=STAGE1_MODEL_PATH,
        ce_model_path=DEFAULT_CE_MODEL,
        corn_model_path=DEFAULT_CORN_MODEL,
    )
    return report.to_dict()

    """
    # ------------------------------------------------------------------
    # OLD: Local implementation (commented out for review)
    # ------------------------------------------------------------------
    contract_text = preprocess_contract(file_path, doc_id)
    raw_clauses = _get_extractor().extract(contract_text, doc_id=doc_id)
    if not raw_clauses:
        return {"document_id": doc_id, "error": "No clauses extracted from document"}

    schema_clauses = [_to_schema_clause(c, doc_id) for c in raw_clauses]

    ce_path = DEFAULT_CE_MODEL if os.path.exists(DEFAULT_CE_MODEL) else None
    corn_path = DEFAULT_CORN_MODEL if os.path.exists(DEFAULT_CORN_MODEL) else None
    assessed = assess_clauses(
        clauses=schema_clauses,
        ce_model_path=ce_path,
        corn_model_path=corn_path,
    )

    report = build_report(clauses=assessed, document_id=doc_id)
    return report.to_dict()
    """
