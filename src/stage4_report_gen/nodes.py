"""
Stage 4 LangGraph node — STUB.

This is a temporary stub introduced during the Stage 4 rewrite (per the
new hierarchical Gemini-Flash design). It exists only so that
`src/workflow/graph.py` continues to import `node_report_generation`
while the rewrite is in progress.

Group I of the rewrite plan will replace this file with the real node
that delegates to `report_builder.generate_report`. Until then, calling
this node populates `final_report` with a placeholder so the graph still
runs end-to-end during Stage 1+2 / Stage 3 testing.
"""

from __future__ import annotations

import logging

from src.workflow.state import RiskAnalysisState

logger = logging.getLogger(__name__)


def node_report_generation(state: RiskAnalysisState) -> dict:
    """STAGE 4 — Node E (stub).

    Placeholder while the new Stage 4 (Gemini-Flash hierarchical design) is
    being built. Returns a minimal `final_report` so the LangGraph DAG
    completes; downstream consumers should treat the report as "not yet
    rendered" and look for `report.status == "stage4_stub"` until the real
    node lands.
    """
    document_id = state.get("document_id", "") or ""
    n_clauses = len(state.get("risk_assessed_clauses", []) or [])

    logger.info(
        "[Stage 4 stub] report generation skipped for document_id=%r "
        "(%d risk-assessed clauses) — real node arrives in Group I.",
        document_id, n_clauses,
    )

    return {
        "final_report": {
            "status": "stage4_stub",
            "document_id": document_id,
            "total_clauses": n_clauses,
            "note": (
                "Stage 4 rewrite in progress — this is a stub. The real "
                "node_report_generation will produce a ContractReport "
                "rendered to JSON / Markdown / PDF / DOCX."
            ),
        }
    }
