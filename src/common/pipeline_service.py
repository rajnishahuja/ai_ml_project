import logging
import os
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from src.common.schema import ClauseObject, RiskReport, RiskAssessedClause, SimilarClause, AgentTraceEntry
from src.stage1_extract_classify.model import ClauseExtractorClassifier
from src.stage1_extract_classify.preprocessing import preprocess_contract
from src.stage3_risk_agent.agent import assess_clauses
from src.stage4_report_gen.report_builder import build_report

logger = logging.getLogger(__name__)

def _to_schema_clause(c, document_id: str) -> ClauseObject:
    """Convert Stage 1 ClauseObject dataclass → common schema ClauseObject."""
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

DEFAULT_STAGE1_MODEL = "rajnishahuja/cuad-stage1-deberta"

def run_end_to_end_pipeline(
    contract_path: str,
    doc_id: Optional[str] = None,
    stage1_model: str = DEFAULT_STAGE1_MODEL,
    stage3_config: str = "configs/stage3_config.yaml",
    stage4_config: str = "configs/stage4_config.yaml",
    ce_model_path: Optional[str] = None,
    corn_model_path: Optional[str] = None,
    use_contract_search: bool = True,
    persist_db_path: str = "data/checkpoints/agent_state.db",
) -> RiskReport:
    """Unified end-to-end pipeline logic used by CLI and API."""
    
    pipeline_start = time.perf_counter()
    
    if doc_id is None:
        doc_id = Path(contract_path).stem
        
    # Stage 1: Document Structure & Layout Parsing
    logger.info("Pipeline: Stage 1 — Preprocessing & Layout Parsing %s", contract_path)
    stage1_start = time.perf_counter()
    contract_text = preprocess_contract(contract_path, doc_id)
    stage1_duration = time.perf_counter() - stage1_start

    # Stage 2: Semantic Clause Span Extraction (With Cache Detection)
    final_dir = os.path.join("data", "output", "final", doc_id)
    os.makedirs(final_dir, exist_ok=True)
    stage2_path = os.path.join(final_dir, "stage2_output.json")
    
    stage2_start = time.perf_counter()
    
    if os.path.exists(stage2_path):
        logger.info("Pipeline: Stage 2 — Found cached stage2_output.json at %s. Skipping DeBERTa span extraction!", stage2_path)
        with open(stage2_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        schema_clauses = [
            ClauseObject(
                clause_id=c.get("clause_id", ""),
                document_id=doc_id,
                clause_text=c.get("clause_text", ""),
                clause_type=c.get("clause_type", ""),
                start_pos=c.get("start_pos", 0),
                end_pos=c.get("end_pos", 0),
                confidence=c.get("confidence", 0.0),
                confidence_logit=c.get("confidence_logit"),
                page_no=c.get("page_no"),
                content_label=c.get("content_label"),
            )
            for c in cached_data
        ]
        stage2_duration = time.perf_counter() - stage2_start
    else:
        logger.info("Pipeline: Stage 2 — Running DeBERTa extraction")
        extractor = ClauseExtractorClassifier(stage1_model)
        raw_clauses = extractor.extract(contract_text, doc_id=doc_id)
        
        if not raw_clauses:
            raise ValueError(f"No clauses extracted from document {doc_id}")
            
        schema_clauses = [_to_schema_clause(c, doc_id) for c in raw_clauses]
        
        # Save Stage 2 intermediate output
        with open(stage2_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() if hasattr(c, "to_dict") else asdict(c) for c in schema_clauses], 
                f, 
                indent=2
            )
        logger.info("Saved Stage 2 intermediate output to %s", stage2_path)
        stage2_duration = time.perf_counter() - stage2_start
    
    # Stage 3: Risk Assessment
    stage3_path = os.path.join(final_dir, "stage3_output.json")
    stage3_start = time.perf_counter()
    if os.path.exists(stage3_path):
        logger.info("Pipeline: Stage 3 — Found cached Stage 3 risk output, resuming from cache")
        with open(stage3_path, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
        assessed = []
        for c in cached_data:
            # Reconstruct SimilarClause items
            sims = []
            if "similar_clauses" in c:
                for s in c["similar_clauses"]:
                    sims.append(SimilarClause(**s))
            c["similar_clauses"] = sims
            
            # Reconstruct AgentTraceEntry items
            trace = []
            if "agent_trace" in c:
                for t in c["agent_trace"]:
                    trace.append(AgentTraceEntry(**t))
            c["agent_trace"] = trace
            
            assessed.append(RiskAssessedClause(**c))
        stage3_duration = time.perf_counter() - stage3_start
    else:
        logger.info("Pipeline: Stage 3 — Assessing risk for %d clauses", len(schema_clauses))
        assessed = assess_clauses(
            clauses=schema_clauses,
            config_path=stage3_config,
            ce_model_path=ce_model_path,
            corn_model_path=corn_model_path,
            use_contract_search=use_contract_search,
            persist_db_path=persist_db_path,
        )
        
        # Save Stage 3 intermediate output
        with open(stage3_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() if hasattr(c, "to_dict") else asdict(c) for c in assessed], 
                f, 
                indent=2
            )
        logger.info("Saved Stage 3 intermediate output to %s", stage3_path)
        stage3_duration = time.perf_counter() - stage3_start
    
    # Stage 4: Report Generation
    logger.info("Pipeline: Stage 4 — Generating report")
    stage4_start = time.perf_counter()
    report = build_report(
        clauses=assessed,
        document_id=doc_id,
        config_path=stage4_config,
    )
    
    # Save Stage 4 final report
    stage4_path = os.path.join(final_dir, "final_report.json")
    with open(stage4_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    logger.info("Saved Stage 4 final report output to %s", stage4_path)
    
    # Save Stage 4 beautiful HTML report
    stage4_html_path = os.path.join(final_dir, "report.html")
    try:
        from src.stage4_report_gen.renderers.html_renderer import render_html_report
        render_html_report(report, stage4_html_path)
        logger.info("Automatically generated beautiful HTML report at %s", stage4_html_path)
    except Exception as e:
        logger.error("Failed to generate HTML report: %s", e)
        
    stage4_duration = time.perf_counter() - stage4_start
    
    total_duration = time.perf_counter() - pipeline_start
    
    # Save latency telemetry metrics
    latency_metrics = {
        "document_id": doc_id,
        "stage1_layout_parsing_seconds": round(stage1_duration, 4),
        "stage2_span_extraction_seconds": round(stage2_duration, 4),
        "stage3_risk_assessment_seconds": round(stage3_duration, 4),
        "stage4_report_generation_seconds": round(stage4_duration, 4),
        "total_pipeline_execution_seconds": round(total_duration, 4)
    }
    
    latency_path = os.path.join(final_dir, "latency_metrics.json")
    with open(latency_path, "w", encoding="utf-8") as f:
        json.dump(latency_metrics, f, indent=2)
        
    # Log duration report dashboard
    logger.info("\n" + "=" * 60 +
                "\n           PIPELINE EXECUTION LATENCY METRICS" +
                "\n" + "=" * 60 +
                f"\n  Document ID:  {doc_id}" +
                f"\n  Stage 1:      {stage1_duration:>6.2f}s  (Document Layout Parsing)" +
                f"\n  Stage 2:      {stage2_duration:>6.2f}s  (Semantic Clause Span Extraction)" +
                f"\n  Stage 3:      {stage3_duration:>6.2f}s  (Agentic Risk Assessment)" +
                f"\n  Stage 4:      {stage4_duration:>6.2f}s  (Executive Report Generation)" +
                "\n  " + "-" * 56 +
                f"\n  Total Time:   {total_duration:>6.2f}s" +
                "\n" + "=" * 60)
    
    return report
