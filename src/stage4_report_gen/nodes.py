"""
LangGraph node for Stage 4 — wraps the modular `report_builder` so it can
be plugged into `src/workflow/graph.py`.

Reads `risk_assessed_clauses` from the graph state, calls
`build_report_dict`, and writes `final_report` back. The heavy lifting
(grouping, scoring, gap analysis, recommendations, explanations) lives in
the sibling modules — this file is the integration seam.
"""

from __future__ import annotations

import logging

from src.stage4_report_gen.report_builder import build_report_dict
from src.workflow.state import RiskAnalysisState

logger = logging.getLogger(__name__)


def node_report_generation(state: RiskAnalysisState) -> dict:
    """Stage 4 — Node E: Aggregate, score, recommend, package the report.

    Waits for Node D's fully risk-assessed clauses. Produces the final JSON
    report payload (see ARCHITECTURE.md §"Stage 4 Output (Final Report)").

    The FLAN-T5 explainer is intentionally not loaded here — clauses arrive
    with a Stage 3 `risk_reason` already, and `build_report_dict` falls back
    to that reason when no `explanation_model` is supplied. To enable
    FLAN-T5 generation, load it once at app startup (e.g. in
    `app/dependencies/`) and pass via a partial:

        from src.stage4_report_gen.explainer import load_explanation_model
        explainer = load_explanation_model(device=0)
        node = lambda s: node_report_generation(s, explainer=explainer)
    """
    assessed_clauses = state.get("risk_assessed_clauses", []) or []
    document_id = state.get("document_id", "") or ""

    print(
        f"🔄 [NODE E | Stage 4 - Report] Aggregating {len(assessed_clauses)} "
        f"assessed clause(s) for document_id={document_id!r}..."
    )

    report = build_report_dict(
        clauses=assessed_clauses,
        document_id=document_id,
        explanation_model=None,        # Stage 3 reasons used verbatim
        max_explanation_length=200,
    )

    print(
        f"✅ [NODE E | Stage 4 - Report] Report compiled. "
        f"Risk score: {report['overall_risk_score']:.2f}/10, "
        f"HIGH={len(report['high_risk'])}, "
        f"MEDIUM={len(report['medium_risk'])}, "
        f"missing_protections={len(report['missing_protections'])}."
    )
    return {"final_report": report}
