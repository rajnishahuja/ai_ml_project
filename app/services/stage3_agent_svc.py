import os
from dataclasses import asdict
from pathlib import Path

from app.schemas.requests import ClauseInput
from src.common.schema import ClauseObject
from src.stage3_risk_agent.agent import assess_clauses

BASE_DIR           = Path(__file__).resolve().parent.parent.parent
DEFAULT_CE_MODEL   = str(BASE_DIR / "models" / "stage3_risk_deberta_v3_run22_parties" / "final")
DEFAULT_CORN_MODEL = str(BASE_DIR / "models" / "stage3_risk_deberta_v3_run23_corn_parties" / "final")


def _to_schema_clause(c: ClauseInput) -> ClauseObject:
    return ClauseObject(
        clause_id=c.clause_id,
        document_id=c.document_id,
        clause_text=c.clause_text,
        clause_type=c.clause_type,
        start_pos=c.start_pos,
        end_pos=c.end_pos,
        confidence=c.confidence,
    )


def run_stage3(clauses: list[ClauseInput], use_contract_search: bool = True) -> list[dict]:
    """Assess risk for a list of clauses. Returns list of RiskAssessedClause dicts."""
    schema_clauses = [_to_schema_clause(c) for c in clauses]

    ce_path   = DEFAULT_CE_MODEL   if os.path.exists(DEFAULT_CE_MODEL)   else None
    corn_path = DEFAULT_CORN_MODEL if os.path.exists(DEFAULT_CORN_MODEL) else None

    assessed = assess_clauses(
        clauses=schema_clauses,
        ce_model_path=ce_path,
        corn_model_path=corn_path,
        use_contract_search=use_contract_search,
    )

    return [asdict(r) for r in assessed]
