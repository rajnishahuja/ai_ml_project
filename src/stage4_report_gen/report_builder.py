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

from src.common.schema import (
    ClauseObject,
    ReportClause,
    ReportMetadata,
    RiskAssessedClause,
    RiskReport,
)
from src.common.utils import load_config, make_llm
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
    llm,
) -> str:
    high_types = ", ".join(c.clause_type for c in high_clauses[:5]) or "none"
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

    llm = make_llm(cfg)

    # 1. Aggregate
    groups = group_by_risk_level(clauses)
    score = compute_contract_risk_score(clauses)
    logger.info(
        "Contract %s — HIGH=%d MEDIUM=%d LOW=%d score=%.3f",
        document_id,
        len(groups["HIGH"]),
        len(groups["MEDIUM"]),
        len(groups["LOW"]),
        score,
    )

    # 2. Clean Explanation Helper for Stage 4 Reports
    def clean_explanation(text: str) -> str:
        if not text:
            return text
        import re

        # Strip agent step headers: ### Step 1: Precedent Search Results etc.
        text = re.sub(r"#+\s*Step\s*\d+[:\.].*", "", text, flags=re.IGNORECASE)
        # Strip other markdown headers (##, ###)
        text = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
        # Strip leading "Analysis:" prefix
        text = re.sub(r"^Analysis:\s*", "", text.strip(), flags=re.IGNORECASE)
        # Strip blockquote clause text lines (> "clause text here")
        text = re.sub(r"^\s*>.*$", "", text, flags=re.MULTILINE)
        # Strip markdown bold/italic markers
        text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
        # Scrub machine learning technicalities to preserve executive presentation
        text = re.sub(
            r"\bDeBERTa's pre-classification\b",
            "automated pre-screening",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bDeBERTa's classification\b",
            "pre-screening assessment",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bDeBERTa's? (preliminary|pre-classification)? (label|prediction|classification) of (\w+) is confirmed\b",
            r"The pre-screening risk rating of \3 is confirmed",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r"\bDeBERTa\b", "automated pre-screening engine", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\bFAISS index\b", "precedent database", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\bsimilarity score(s)?\b", "relevance match", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\bsimilarity threshold\b", "relevance criteria", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\bprecedent_search\b", "precedent search tool", text, flags=re.IGNORECASE
        )
        text = re.sub(
            r"\bcontract_search\b",
            "contract lookup search tool",
            text,
            flags=re.IGNORECASE,
        )
        # Collapse multiple blank lines into a single paragraph break
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # 3. Build ReportClause for HIGH and MEDIUM
    def to_report_clause(c: RiskAssessedClause) -> ReportClause:
        return ReportClause(
            clause_id=c.clause_id,
            clause_type=c.clause_type,
            risk_level=c.risk_level,
            explanation=clean_explanation(c.risk_explanation),
            recommendation=get_recommendation(c.clause_type, c.risk_level),
            similar_clauses=c.similar_clauses,
            cross_references=c.cross_references,
            page_no=c.page_no,
            agent_trace=c.agent_trace,
            risk_confidence=c.confidence / 100.0 if c.confidence > 1.0 else c.confidence,
            extraction_confidence=getattr(c, "extraction_confidence", 0.0),
            is_override=getattr(c, "is_override", False),
            agent_confidence=getattr(c, "agent_confidence", 0.0) / 100.0 if getattr(c, "agent_confidence", 0.0) > 1.0 else getattr(c, "agent_confidence", 0.0),
            extraction_confidence_logit=getattr(c, "extraction_confidence_logit", None),
            content_label=getattr(c, "content_label", None),
            clause_text=c.clause_text,
        )

    high_report = [to_report_clause(c) for c in groups["HIGH"]]
    medium_report = [to_report_clause(c) for c in groups["MEDIUM"]]
    low_report = [to_report_clause(c) for c in groups["LOW"]]

    # 3. Low-risk summary (legacy count - kept for overall stats)
    low_summary = f"{len(groups['LOW'])} clauses were assessed as low risk and are detailed below."

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

    # 5. Missing Protections Logic (CUAD Taxonomy)
    # These are the "Must-Haves" from the 41 CUAD types
    essential_cuad_types = {
        "Governing Law",
        "Cap On Liability",
        "Termination For Convenience",
        "Anti-Assignment",
        "Insurance",
        "Audit Rights",
    }
    found_types = {c.clause_type for c in clauses}
    missing = sorted(list(essential_cuad_types - found_types))

    # 6. Assemble
    return RiskReport(
        document_id=document_id,
        summary=summary,
        high_risk=high_report,
        medium_risk=medium_report,
        low_risk=low_report,
        low_risk_summary=low_summary,
        missing_protections=missing,
        overall_risk_score=score,
        total_clauses=len(clauses),
        metadata=ReportMetadata(
            generated_at=datetime.now(timezone.utc).isoformat(),
            models_used={
                "extraction": "Docling + DeBERTa-v3-base",
                "risk_classification": "Ens-F (DeBERTa-v3-base CE+CORN)",
                "explanation": cfg["agent_model"],
                "report_summary": cfg["agent_model"],
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


def _format_clause(i: int, c: ReportClause) -> list[str]:
    lines = []
    page_str = f" (Page {c.page_no})" if c.page_no else ""
    lines.append(f"### {i}. {c.clause_type}{page_str}\n")
    lines.append(f"**Explanation**: {c.explanation}\n")

    # Reasoning trace
    if c.agent_trace:
        # Handle both dict (from JSON) and object
        steps = []
        for t in c.agent_trace:
            t_tool = (
                t.get("tool", "Unknown")
                if isinstance(t, dict)
                else getattr(t, "tool", "Unknown")
            )
            t_hits = (
                t.get("result_count", 0)
                if isinstance(t, dict)
                else getattr(t, "result_count", 0)
            )
            steps.append(f"`{t_tool}` ({t_hits} hits)")

        trace_str = " → ".join(steps)
        lines.append(f"**Reasoning Trace**: {trace_str}\n")

    lines.append(f"**Recommendation**: {c.recommendation}\n")

    # Evidence / References
    if c.similar_clauses or c.cross_references:
        lines.append("#### 📚 Supporting References")
        if c.similar_clauses:
            lines.append("- **Market Precedents**:")
            for s in c.similar_clauses[:3]:  # Top 3
                # Handle both dict (from JSON) and object
                s_type = (
                    s.get("clause_type", "Unknown")
                    if isinstance(s, dict)
                    else getattr(s, "clause_type", "Unknown")
                )
                s_sim = (
                    s.get("similarity", 0.0)
                    if isinstance(s, dict)
                    else getattr(s, "similarity", 0.0)
                )
                s_text = (
                    s.get("text", "") if isinstance(s, dict) else getattr(s, "text", "")
                )
                lines.append(
                    f'  - *[{s_type}]* (Similarity: {s_sim:.2f}): "{s_text[:150]}..."'
                )
            lines.append("")  # Blank line after list

        if c.cross_references:
            lines.append("- **Internal Cross-References**:")
            for ref in c.cross_references[:5]:
                # Handle both dict (from stage 3) and string
                ref_text = (
                    ref.get("clause_type", "Clause")
                    if isinstance(ref, dict)
                    else "Related Clause"
                )
                lines.append(f"  - {ref_text}")

    lines.append("---")
    return lines


def export_markdown_report(report: RiskReport) -> str:
    """Generate a human-readable Markdown version of the RiskReport."""
    lines = []
    lines.append(f"# Legal Risk Report: {report.document_id}")
    lines.append(f"\n**Generated at**: {report.metadata.generated_at}")
    lines.append(f"\n**Overall Risk Score**: `{report.overall_risk_score:.2f} / 1.00`")

    lines.append("\n## 📝 Executive Summary")
    lines.append(report.summary)

    if report.high_risk:
        lines.append("\n## 🔴 HIGH RISK CLAUSES")
        for i, c in enumerate(report.high_risk, 1):
            lines.extend(_format_clause(i, c))

    if report.medium_risk:
        lines.append("\n## 🟡 MEDIUM RISK CLAUSES")
        for i, c in enumerate(report.medium_risk, 1):
            lines.extend(_format_clause(i, c))

    if report.missing_protections:
        lines.append("\n## ⚠️ MISSING PROTECTIONS (CUAD)")
        lines.append(
            "The following essential CUAD protections were NOT detected in this contract:"
        )
        for p in report.missing_protections:
            lines.append(f"- {p}")

    if report.low_risk:
        lines.append("\n## ✅ LOW RISK CLAUSES")
        for i, c in enumerate(report.low_risk, 1):
            lines.extend(_format_clause(i, c))

    lines.append("\n\n---\n*Report generated by AI Legal Assistant*")
    return "\n".join(lines)


def save_markdown_report(report: RiskReport, output_path: str) -> None:
    """Serialise and save a RiskReport to Markdown."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    md_content = export_markdown_report(report)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Markdown report saved to %s", output_path)
