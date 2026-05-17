"""
End-to-end contract risk analysis pipeline (CLI).

Usage:
    python scripts/run_pipeline.py --contract path/to/contract.pdf
    python scripts/run_pipeline.py --contract path/to/contract.pdf --output report.json
    python scripts/run_pipeline.py --contract path/to/contract.pdf --no-contract-search

Stages:
    1. Preprocess + extract clauses  (Stage 1 DeBERTa QA model)
    2. Assess risk per clause        (Stage 3 Ens-F + LangGraph agent)
    3. Generate report               (Stage 4 aggregator + Qwen summary)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_STAGE1_MODEL = "rajnishahuja/cuad-stage1-deberta"
DEFAULT_STAGE3_CONFIG = "configs/stage3_config.yaml"
DEFAULT_STAGE4_CONFIG = "configs/stage4_config.yaml"
DEFAULT_CE_MODEL = "rajnishahuja/cuad-risk-deberta-ce-parties"
DEFAULT_CORN_MODEL = "rajnishahuja/cuad-risk-deberta-corn-parties"


def _to_schema_clause(c, document_id: str):
    """Convert Stage 1 ClauseObject dataclass → common schema ClauseObject."""
    from src.common.schema import ClauseObject

    return ClauseObject(
        clause_id=c.clause_id,
        document_id=document_id,
        clause_text=c.clause_text,
        clause_type=c.clause_type,
        start_pos=c.start_pos,
        end_pos=c.end_pos,
        confidence=c.confidence,
        confidence_logit=getattr(c, "confidence_logit", None),
        page_no=getattr(c, "page_no", None),
        content_label=getattr(c, "content_label", None),
    )


from src.common.pipeline_service import run_end_to_end_pipeline
from src.stage4_report_gen.report_builder import save_report


from src.common.pipeline_service import run_end_to_end_pipeline


def run(
    contract_path: str,
    output_path: str,
    stage1_model: str,
    stage3_config: str,
    stage4_config: str,
    ce_model_path: str,
    corn_model_path: str,
    use_contract_search: bool,
    persist_db_path: str,
) -> None:
    # ------------------------------------------------------------------
    # NEW: Unified Pipeline Service
    # ------------------------------------------------------------------
    doc_id = Path(contract_path).stem
    report = run_end_to_end_pipeline(
        contract_path=contract_path,
        doc_id=doc_id,
        stage1_model=stage1_model,
        stage3_config=stage3_config,
        stage4_config=stage4_config,
        ce_model_path=ce_model_path,
        corn_model_path=corn_model_path,
        use_contract_search=use_contract_search,
        persist_db_path=persist_db_path,
    )

    from src.stage4_report_gen.report_builder import save_report
    save_report(report, output_path)
    
    # Commented out Markdown output - focusing on JSON-only outputs for now
    # from src.stage4_report_gen.report_builder import save_markdown_report
    # md_output = str(Path(output_path).with_suffix(".md"))
    # save_markdown_report(report, md_output)
    # logger.info("Human-readable report saved to %s", md_output)

    # Calculate counts for display
    high_count = len(report.high_risk)
    med_count = len(report.medium_risk)
    low_count = report.total_clauses - high_count - med_count

    print(f"\n{'='*60}")
    print(f"  Contract  : {doc_id}")
    print(f"  Risk score: {report.overall_risk_score:.2f} / 1.00")
    print(f"  HIGH      : {high_count} clause(s)")
    print(f"  MEDIUM    : {med_count} clause(s)")
    print(f"  LOW       : {low_count} clause(s)")
    print(f"  Summary   : {report.summary[:200]}...")
    print(f"{'='*60}\n")

    """
    # ------------------------------------------------------------------
    # OLD: Local implementation (commented out for review)
    # ------------------------------------------------------------------
    from src.stage1_extract_classify.model import ClauseExtractorClassifier
    from src.stage1_extract_classify.preprocessing import preprocess_contract
    from src.stage3_risk_agent.agent import assess_clauses
    from src.stage4_report_gen.report_builder import build_report, save_report

    contract_path = os.path.abspath(contract_path)
    doc_id = Path(contract_path).stem

    # Stage 1: extract clauses
    logger.info("Stage 1 — preprocessing %s", contract_path)
    contract_text = preprocess_contract(contract_path, doc_id)
    extractor = ClauseExtractorClassifier(stage1_model)
    raw_clauses = extractor.extract(contract_text, doc_id=doc_id)

    if not raw_clauses:
        logger.warning("No clauses extracted — check the contract file and model.")
        sys.exit(1)

    schema_clauses = [_to_schema_clause(c, doc_id) for c in raw_clauses]

    # Stage 3: assess risk
    assessed = assess_clauses(
        clauses=schema_clauses,
        config_path=stage3_config,
        ce_model_path=ce_model_path if os.path.exists(ce_model_path) else None,
        corn_model_path=corn_model_path if os.path.exists(corn_model_path) else None,
        use_contract_search=use_contract_search,
    )

    # Stage 4: generate report
    report = build_report(
        clauses=assessed,
        document_id=doc_id,
        config_path=stage4_config,
    )
    save_report(report, output_path)
    """


def main():
    parser = argparse.ArgumentParser(
        description="Legal Contract Risk Analyzer — end-to-end pipeline"
    )
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to contract file (PDF, DOCX, or TXT)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: <contract_stem>_report.json)",
    )
    parser.add_argument(
        "--stage1-model",
        default=DEFAULT_STAGE1_MODEL,
        help=f"Path to Stage 1 DeBERTa model (default: {DEFAULT_STAGE1_MODEL})",
    )
    parser.add_argument(
        "--stage3-config",
        default=DEFAULT_STAGE3_CONFIG,
    )
    parser.add_argument(
        "--stage4-config",
        default=DEFAULT_STAGE4_CONFIG,
    )
    parser.add_argument(
        "--ce-model",
        default=DEFAULT_CE_MODEL,
        help="Path to CE DeBERTa model (falls back to HF Hub if path missing)",
    )
    parser.add_argument(
        "--corn-model",
        default=DEFAULT_CORN_MODEL,
        help="Path to CORN DeBERTa model (falls back to HF Hub if path missing)",
    )
    parser.add_argument(
        "--no-contract-search",
        action="store_true",
        help="Disable contract_search tool on agent path (faster, slightly lower F1)",
    )
    parser.add_argument(
        "--persist-db",
        default="data/checkpoints/agent_state.db",
        help="Path to SQLite database for persistent agent checkpointing",
    )
    args = parser.parse_args()

    if not os.path.exists(args.contract):
        parser.error(f"Contract file not found: {args.contract}")

    output = args.output or f"{Path(args.contract).stem}_report.json"

    run(
        contract_path=args.contract,
        output_path=output,
        stage1_model=args.stage1_model,
        stage3_config=args.stage3_config,
        stage4_config=args.stage4_config,
        ce_model_path=args.ce_model,
        corn_model_path=args.corn_model,
        use_contract_search=not args.no_contract_search,
        persist_db_path=args.persist_db,
    )


if __name__ == "__main__":
    main()
