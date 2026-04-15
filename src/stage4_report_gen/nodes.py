from src.workflow.state import RiskAnalysisState


def node_report_generation(state: RiskAnalysisState):
    """
    STAGE 4 — Node E
    Waits for Node D (fully risk-assessed clauses). Aggregates and categorises
    clauses by risk tier, computes an overall contract risk score, identifies
    missing standard protections, and packages the structured JSON report
    ready for the frontend UI / client.
    Output: final_report -> returned to the API response
    """
    assessed_clauses = state.get("risk_assessed_clauses", [])
    print(f"🔄 [NODE E | Stage 4 - Report] Aggregating {len(assessed_clauses)} assessed clauses...")

    high_risk   = [c for c in assessed_clauses if c.risk_level == "HIGH"]
    medium_risk = [c for c in assessed_clauses if c.risk_level == "MEDIUM"]
    low_risk    = [c for c in assessed_clauses if c.risk_level == "LOW"]

    # TODO: Replace mock score/summary with FLAN-T5 or deterministic scoring logic
    report = {
        "summary": (
            f"This contract contains {len(assessed_clauses)} assessed clause(s). "
            f"{len(high_risk)} HIGH, {len(medium_risk)} MEDIUM, {len(low_risk)} LOW risk."
        ),
        "high_risk": [
            {
                "clause_id": c.clause_id,
                "clause_type": c.clause_type,
                "explanation": c.risk_reason,
                "recommendation": "Renegotiate to achieve mutual and balanced protections."
            }
            for c in high_risk
        ],
        "medium_risk": [
            {
                "clause_id": c.clause_id,
                "clause_type": c.clause_type,
                "explanation": c.risk_reason,
                "recommendation": "Review and clarify scope."
            }
            for c in medium_risk
        ],
        "low_risk_summary": f"{len(low_risk)} clause(s) assessed as standard / low risk.",
        "missing_protections": [],   # TODO: populate with gap analysis logic
        "overall_risk_score": round(len(high_risk) * 2.5 + len(medium_risk) * 1.0, 2),
        "total_clauses": len(assessed_clauses)
    }

    print(f"✅ [NODE E | Stage 4 - Report] Final report compiled. Risk score: {report['overall_risk_score']}")
    return {"final_report": report}
