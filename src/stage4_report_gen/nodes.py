"""
Stage 4 LangGraph node — orchestrates the Mistral summary call, the report
assembly, and the DOCX/PDF rendering.

Reads from state:
  - risk_assessed_clauses : Stage 3 output (list of RiskAssessedClause)
  - extracted_clauses     : Stage 1+2 output (used to build the metadata
                            block for the report header)
  - contract_text         : Full contract text (passed to Mistral as
                            input for the summary + conclusion)
  - document_id           : Contract identifier (UUID from /analyze)

Writes to state:
  - final_report : dict with all report fields, including `docx_path` and
                   `pdf_path` for the FastAPI download endpoints.

The Mistral wrapper is pluggable — defaults to the deterministic Mock so
the pipeline runs end-to-end before real Mistral is wired. To swap, set
`src.stage4_report_gen.nodes.LLM_CLIENT` to a real `LLMClient` instance.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.common.llm_client import DEFAULT_CLIENT, LLMClient, MockLLMClient
from src.stage3_risk_agent.tools import extract_metadata_block
from src.stage4_report_gen.aggregator import (
    compute_contract_risk_score,
    group_by_risk_level,
)
from src.stage4_report_gen.docx_renderer import render_docx
from src.stage4_report_gen.pdf_converter import convert_to_pdf
from src.stage4_report_gen.prompts import build_summary_and_conclusion_prompt
from src.stage4_report_gen.report_builder import assemble_report_dict
from src.workflow.state import RiskAnalysisState

logger = logging.getLogger(__name__)


# Module-level swappable client. Tests / dev use the Mock; production will
# replace this once the Mistral integration lands.
LLM_CLIENT: LLMClient = DEFAULT_CLIENT

# Reports land in this directory by default. Each report file is keyed by
# document_id (UUID), so concurrent /analyze calls don't collide.
DEFAULT_REPORTS_DIR = Path("data/reports")


def _invoke_llm_safely(client: LLMClient, system_prompt: str, user_prompt: str) -> dict:
    """Call the LLM, fall back to the Mock if anything goes wrong.

    The pipeline must always produce a report. If a real client raises
    (e.g. NotImplementedError on the HuggingFace stub, OOM, JSON parse
    failure), we log and substitute Mock output so the rest of Stage 4
    (DOCX, PDF, FastAPI download) still completes.
    """
    try:
        return client.generate_json(system_prompt, user_prompt)
    except Exception as e:                                            # noqa: BLE001
        logger.warning(
            "Stage 4 LLM call failed (%s) — falling back to MockLLMClient.", e,
        )
        return MockLLMClient().generate_json(system_prompt, user_prompt)


def node_report_generation(state: RiskAnalysisState) -> dict:
    """STAGE 4 — Node E: assemble + render the final risk report.

    Map-Reduce fan-in node. Runs ONCE per contract after every Stage 3
    worker has produced its risk-assessed clause.
    """
    clauses       = state.get("risk_assessed_clauses", []) or []
    extracted     = state.get("extracted_clauses", []) or []
    contract_text = state.get("contract_text", "") or ""
    document_id   = state.get("document_id", "") or ""

    print(
        f"🔄 [NODE E | Stage 4 - Report] Assembling report for "
        f"document_id={document_id!r} from {len(clauses)} risk-assessed "
        f"clause(s)..."
    )

    # --- Phase 1: deterministic aggregation ------------------------------
    metadata = extract_metadata_block(extracted)
    grouped  = group_by_risk_level(clauses)
    score    = compute_contract_risk_score(clauses)

    # --- Phase 2: Mistral call (summary + conclusion) --------------------
    sys_prompt, usr_prompt = build_summary_and_conclusion_prompt(
        contract_text=contract_text,
        grouped_clauses=grouped,
    )
    llm_out = _invoke_llm_safely(LLM_CLIENT, sys_prompt, usr_prompt)

    models_used = {
        "summary": getattr(LLM_CLIENT, "name", LLM_CLIENT.__class__.__name__),
    }

    # --- Phase 3: assemble the report dict -------------------------------
    report = assemble_report_dict(
        document_id=document_id,
        metadata=metadata,
        grouped=grouped,
        overall_risk_score=score,
        llm_output=llm_out,
        models_used=models_used,
    )

    # --- Phase 4: render DOCX, then convert to PDF (best-effort) --------
    docx_path = render_docx(report, output_dir=DEFAULT_REPORTS_DIR)
    pdf_path  = convert_to_pdf(docx_path)

    report["docx_path"] = str(docx_path)
    report["pdf_path"]  = str(pdf_path) if pdf_path else None

    print(
        f"✅ [NODE E] Report ready. "
        f"score={report['overall_risk_score']:.2f}/10, "
        f"HIGH={len(report['risk_tables']['HIGH'])}, "
        f"MEDIUM={len(report['risk_tables']['MEDIUM'])}, "
        f"LOW={len(report['risk_tables']['LOW'])}, "
        f"docx={docx_path.name}, "
        f"pdf={'yes' if pdf_path else 'skipped'}"
    )
    return {"final_report": report}
