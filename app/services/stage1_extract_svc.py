import os
from pathlib import Path

from src.common.schema import ClauseObject
from src.stage1_extract_classify.model import ClauseExtractorClassifier
from src.stage1_extract_classify.preprocessing import preprocess_contract
from src.stage3_risk_agent.agent import assess_clauses
from src.stage4_report_gen.report_builder import build_report

BASE_DIR           = Path(__file__).resolve().parent.parent.parent
DEFAULT_STAGE1     = str(BASE_DIR / "models" / "stage1_2_deberta")
DEFAULT_CE_MODEL   = str(BASE_DIR / "models" / "stage3_risk_deberta_v3_run22_parties" / "final")
DEFAULT_CORN_MODEL = str(BASE_DIR / "models" / "stage3_risk_deberta_v3_run23_corn_parties" / "final")

STAGE1_MODEL_PATH = os.getenv("STAGE1_MODEL_PATH", DEFAULT_STAGE1)

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


def run_full_pipeline(file_path: str, doc_id: str) -> dict:
    """Run Stage 1 → Stage 3 → Stage 4 synchronously.

    Called via asyncio.run_in_executor() so it never blocks the FastAPI event loop.
    Returns report.to_dict() ready for JSON serialisation.
    """
    contract_text = preprocess_contract(file_path, doc_id)
    raw_clauses = _get_extractor().extract(contract_text, doc_id=doc_id)
    if not raw_clauses:
        return {"document_id": doc_id, "error": "No clauses extracted from document"}

    schema_clauses = [_to_schema_clause(c, doc_id) for c in raw_clauses]

    ce_path   = DEFAULT_CE_MODEL   if os.path.exists(DEFAULT_CE_MODEL)   else None
    corn_path = DEFAULT_CORN_MODEL if os.path.exists(DEFAULT_CORN_MODEL) else None
    assessed = assess_clauses(
        clauses=schema_clauses,
        ce_model_path=ce_path,
        corn_model_path=corn_path,
    )

    report = build_report(clauses=assessed, document_id=doc_id)
    return report.to_dict()
