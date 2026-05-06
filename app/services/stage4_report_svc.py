from app.schemas.requests import AssessedClauseInput
from src.common.schema import RiskAssessedClause
from src.stage4_report_gen.report_builder import build_report


def _to_internal(c: AssessedClauseInput) -> RiskAssessedClause:
    return RiskAssessedClause(
        clause_id=c.clause_id,
        document_id=c.document_id,
        clause_text=c.clause_text,
        clause_type=c.clause_type,
        risk_level=c.risk_level,
        risk_explanation=c.risk_explanation,
        confidence=c.confidence,
    )


def run_stage4(document_id: str, assessed_clauses: list[AssessedClauseInput]) -> dict:
    """Generate a risk report from already-assessed clauses."""
    internal = [_to_internal(c) for c in assessed_clauses]
    report = build_report(clauses=internal, document_id=document_id)
    return report.to_dict()
