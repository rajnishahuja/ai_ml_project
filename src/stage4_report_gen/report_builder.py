"""
Report assembly: combines aggregation and recommendations into a RiskReport.

build_report() is the main entry point. It:
  1. Groups clauses by risk level via aggregator.
  2. Attaches per-clause recommendations via recommender.
  3. Calls Qwen once to generate the executive summary.
  4. Assembles and returns a RiskReport.
"""

import json
import logging
import os
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from src.common.schema import (
    ClauseObject,
    ReportClause,
    ReportMetadata,
    RiskAssessedClause,
    RiskReport,
)
from src.common.utils import load_config
from src.stage4_report_gen.aggregator import (
    compute_contract_risk_score,
    group_by_risk_level,
)
from src.stage4_report_gen.recommender import get_recommendation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executive summary (single LLM call)
# ---------------------------------------------------------------------------

def _generate_summary(
    document_id: str,
    high_clauses: list[RiskAssessedClause],
    medium_clauses: list[RiskAssessedClause],
    low_count: int,
    overall_score: float,
    llm: ChatOpenAI,
) -> str:
    high_types  = ", ".join(c.clause_type for c in high_clauses[:5]) or "none"
    medium_types = ", ".join(c.clause_type for c in medium_clauses[:5]) or "none"

    prompt = (
        f"You are a legal risk analyst. Write a concise executive summary "
        f"(2–3 sentences) for the following contract risk assessment.\n\n"
        f"Contract: {document_id}\n"
        f"Overall risk score: {overall_score:.2f} / 1.00\n"
        f"High-risk clauses ({len(high_clauses)}): {high_types}\n"
        f"Medium-risk clauses ({len(medium_clauses)}): {medium_types}\n"
        f"Low-risk clauses: {low_count}\n\n"
        f"Summary should highlight the most significant risks and give the "
        f"signing party a clear sense of overall exposure. Be direct and factual."
    )
    response = llm.invoke(prompt)
    return response.content.strip() if hasattr(response, "content") else str(response)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_report(
    clauses: list[RiskAssessedClause],
    document_id: str,
    config_path: str = "configs/stage4_config.yaml",
) -> RiskReport:
    """Build the final risk report for a contract.

    Steps:
        1. Aggregate clauses (group by risk level, compute score).
        2. Attach per-clause recommendations from lookup table.
        3. Generate executive summary via Qwen (single LLM call).
        4. Assemble into RiskReport.

    Args:
        clauses:     Stage 3 output (list of RiskAssessedClause).
        document_id: Identifier for the source contract.
        config_path: Path to stage4_config.yaml.

    Returns:
        Complete RiskReport ready for serialisation.
    """
    cfg = load_config(config_path)

    llm = ChatOpenAI(
        model=cfg["agent_model"],
        base_url=cfg["agent_base_url"],
        api_key=cfg.get("agent_api_key", "none"),
        temperature=0,
    )

    # 1. Aggregate
    groups = group_by_risk_level(clauses)
    score  = compute_contract_risk_score(clauses)
    logger.info(
        "Contract %s — HIGH=%d MEDIUM=%d LOW=%d score=%.3f",
        document_id,
        len(groups["HIGH"]), len(groups["MEDIUM"]), len(groups["LOW"]),
        score,
    )

    # 2. Build ReportClause for HIGH and MEDIUM
    def to_report_clause(c: RiskAssessedClause) -> ReportClause:
        return ReportClause(
            clause_id=c.clause_id,
            clause_type=c.clause_type,
            risk_level=c.risk_level,
            explanation=c.risk_explanation,
            recommendation=get_recommendation(c.clause_type, c.risk_level),
        )

    high_report   = [to_report_clause(c) for c in groups["HIGH"]]
    medium_report = [to_report_clause(c) for c in groups["MEDIUM"]]

    # 3. Low-risk summary (counts + types — no LLM needed)
    low_types = sorted({c.clause_type for c in groups["LOW"]})
    if low_types:
        low_summary = (
            f"{len(groups['LOW'])} clause(s) assessed as low risk: "
            + ", ".join(low_types) + "."
        )
    else:
        low_summary = "No low-risk clauses identified."

    # 4. Executive summary (one LLM call)
    logger.info("Generating executive summary via LLM ...")
    summary = _generate_summary(
        document_id=document_id,
        high_clauses=groups["HIGH"],
        medium_clauses=groups["MEDIUM"],
        low_count=len(groups["LOW"]),
        overall_score=score,
        llm=llm,
    )

    # 5. Assemble
    return RiskReport(
        document_id=document_id,
        summary=summary,
        high_risk=high_report,
        medium_risk=medium_report,
        low_risk_summary=low_summary,
        missing_protections=[],
        overall_risk_score=score,
        total_clauses=len(clauses),
        metadata=ReportMetadata(
            generated_at=datetime.now(timezone.utc).isoformat(),
            models_used={
                "risk_classification": "Ens-F (DeBERTa-v3-base CE+CORN)",
                "risk_explanation":    cfg["agent_model"],
                "report_summary":      cfg["agent_model"],
            },
        ),
    )


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def save_report(report: RiskReport, output_path: str) -> None:
    """Serialise and save a RiskReport to JSON.

    Args:
        report:      The assembled report.
        output_path: File path to write the JSON output.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Report saved to %s", output_path)
