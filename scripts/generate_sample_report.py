"""
One-off demo: generate a sample Stage 4 report (DOCX + PDF) for a real CUAD
contract using synthetic upstream data.

Why this exists:
  - The user wants to see what the new Stage 4 report looks like end-to-end.
  - Stage 1+2 (DeBERTa) and Stage 3 (Mistral) aren't wired yet — we
    synthesize plausible outputs from real CUAD clause text.
  - Stage 4 itself runs production code unchanged via `node_report_generation`.
  - LibreOffice / docx2pdf isn't installed on this host, so PDF generation
    here uses reportlab directly from the same report dict the Stage 4
    pipeline assembled (no DOCX→PDF conversion). The DOCX is the production
    deliverable; this PDF is a faithful preview using reportlab.

Output:
  - data/reports/sample_<contract_slug>.docx  (production code path)
  - data/reports/sample_<contract_slug>.pdf   (reportlab preview)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from app.schemas.domain import ExtractedClause, RiskAssessedClause, SimilarClause

from src.stage3_risk_agent.tools import _load_corpus, _load_index
from src.stage4_report_gen.nodes import node_report_generation

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    KeepTogether,
)


# ---------------------------------------------------------------------------
# 1. Pick a real CUAD contract with a diverse clause set
# ---------------------------------------------------------------------------

DOC_ID = (
    "DovaPharmaceuticalsInc_20181108_10-Q_EX-10.2_11414857_EX-10.2_"
    "Promotion Agreement"
)

# Risk levels assigned to the contract's clause types — chosen to produce a
# balanced report (HIGH / MEDIUM / LOW each populated). In production these
# come from Stage 3's DeBERTa classifier + Mistral agent.
RISK_ASSIGNMENTS: dict[str, tuple[str, str, float]] = {
    # type:                       (risk_level, risk_reason, confidence)
    "Cap On Liability":           ("LOW",    "Standard 12-month-fees cap; mutual; commercially reasonable.", 0.91),
    "Uncapped Liability":         ("HIGH",   "Uncapped exposure for IP indemnity is one-sided and creates material liability for Dova.", 0.78),
    "Liquidated Damages":         ("MEDIUM", "Termination fee triggers only on Dova's breach — asymmetric remedy.", 0.71),
    "Insurance":                  ("LOW",    "Both parties carry $5M coverage; standard for the industry.", 0.95),
    "Covenant Not To Sue":        ("MEDIUM", "Broad waiver scope; could limit Dova's IP defense rights against Valeant.", 0.66),
    "Termination For Convenience":("MEDIUM", "30-day notice is short for a multi-year promotion deal of this scale.", 0.74),
    "Non-Compete":                ("HIGH",   "Geographic scope unbounded; 5-year tail extends well past contract term.", 0.80),
    "Audit Rights":               ("LOW",    "Annual audit cadence with reasonable advance notice; standard.", 0.93),
    "Anti-Assignment":            ("MEDIUM", "No affiliate carve-out; blocks routine corporate restructuring.", 0.69),
    "Exclusivity":                ("HIGH",   "Exclusive dealing in North America with no minimum-volume off-ramps.", 0.82),
    "Renewal Term":               ("LOW",    "Auto-renewal with 60-day opt-out; reasonable.", 0.88),
    "Most Favored Nation":        ("MEDIUM", "MFN clause limits future pricing flexibility on Dova's side.", 0.70),
    "Warranty Duration":          ("MEDIUM", "24-month warranty exceeds 12-month industry norm — minor cost exposure.", 0.75),
}


# ---------------------------------------------------------------------------
# 2. Build synthetic upstream state from the real CUAD corpus
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:40].strip("_")


def build_synthetic_state() -> dict:
    """Build a LangGraph state slice as if Stage 1+2 + Stage 3 had just run."""
    # Pull the real clauses Stage 1+2 would have extracted from this contract.
    try:
        index = _load_index("data/processed/contract_clause_index.json")
        entry = index[DOC_ID]
        all_clauses = [{**c, "document_id": DOC_ID} for c in entry["clauses"]]
    except (FileNotFoundError, KeyError):
        # Fallback to the spans corpus
        all_spans = _load_corpus("data/processed/all_positive_spans.json")
        all_clauses = [c for c in all_spans if c["document_id"] == DOC_ID]

    print(f"Loaded {len(all_clauses)} clauses for: {DOC_ID[:60]}...")

    # Build extracted_clauses (Stage 1+2 output schema — pydantic).
    extracted: list[ExtractedClause] = []
    for c in all_clauses:
        extracted.append(ExtractedClause(
            clause_id=c["clause_id"],
            clause_text=c["clause_text"],
            clause_type=c["clause_type"],
            start_pos=c.get("start_pos", 0),
            end_pos=c.get("start_pos", 0) + len(c["clause_text"]),
            confidence=0.9,
            confidence_logit=2.5,
            page_no=str((c.get("start_pos", 0) // 3000) + 1),
            content_label=f"Section {c['clause_type']}",
        ))

    # Build risk_assessed_clauses (Stage 3 output) — only for non-metadata
    # types that have a risk assignment defined above.
    risk_assessed: list[RiskAssessedClause] = []
    for c in all_clauses:
        ctype = c["clause_type"]
        if ctype not in RISK_ASSIGNMENTS:
            continue
        level, reason, conf = RISK_ASSIGNMENTS[ctype]
        risk_assessed.append(RiskAssessedClause(
            clause_id=c["clause_id"],
            clause_text=c["clause_text"],
            clause_type=ctype,
            start_pos=c.get("start_pos", 0),
            end_pos=c.get("start_pos", 0) + len(c["clause_text"]),
            confidence=conf,
            confidence_logit=2.5,
            page_no=str((c.get("start_pos", 0) // 3000) + 1),
            content_label=f"Section {ctype}",
            risk_level=level,
            risk_reason=reason,
            similar_clauses=[
                SimilarClause(text="(precedent example)", risk=level, similarity=0.85),
            ],
            cross_references=["precedent_search", "contract_search"],
            overridden=False,
        ))

    # Synthetic full contract text — concatenate the clause texts as a stand-in
    # for what Stage 1's preprocess_contract would have produced from the PDF.
    contract_text = "\n\n".join(
        f"[{c['clause_type']}]\n{c['clause_text']}" for c in all_clauses
    )

    return {
        "document_id": _slug(DOC_ID),
        "contract_text": contract_text,
        "extracted_clauses": extracted,
        "risk_assessed_clauses": risk_assessed,
    }


# ---------------------------------------------------------------------------
# 3. Render PDF directly from the report dict (reportlab path)
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n].rstrip() + " ..."


def render_pdf_from_report(report: dict, output_dir: Path | str) -> Path:
    """Render the assembled report dict to PDF using reportlab.

    Mirrors the DOCX layout produced by `docx_renderer.render_docx` but uses
    reportlab so it works without LibreOffice. Used as the user-facing
    deliverable on hosts where docx2pdf can't run.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    document_id = str(report.get("document_id") or "unknown")
    out_path = out_dir / f"{document_id}.pdf"

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title=f"Risk Report — {document_id}",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleH", parent=styles["Title"], textColor=colors.HexColor("#1F3A5F"),
        spaceAfter=8, fontSize=20,
    ))
    styles.add(ParagraphStyle(
        "H1c", parent=styles["Heading1"], textColor=colors.HexColor("#1F3A5F"),
        spaceBefore=10, spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        "H2c", parent=styles["Heading2"], spaceBefore=8, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "BodySmall", parent=styles["BodyText"], fontSize=9, leading=11,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer", parent=styles["BodyText"], fontSize=9, leading=12,
        textColor=colors.HexColor("#555555"), alignment=4,  # justify
    ))
    styles.add(ParagraphStyle(
        "BulletItem", parent=styles["BodyText"],
        leftIndent=14, bulletIndent=4, spaceAfter=2,
    ))

    tier_colors = {
        "HIGH":   colors.HexColor("#C0392B"),
        "MEDIUM": colors.HexColor("#E67E22"),
        "LOW":    colors.HexColor("#27AE60"),
    }

    story = []

    # ── Title block ─────────────────────────────────────────────────────
    story.append(Paragraph("Contract Risk Analysis Report", styles["TitleH"]))

    score = report.get("overall_risk_score") or 0.0
    info = (
        f"<b>Document ID:</b> {report.get('document_id', '—')}<br/>"
        f"<b>Generated:</b> {report.get('generated_at', '—')}<br/>"
        f"<b>Overall Risk Score:</b> {score:.2f} / 10"
    )
    story.append(Paragraph(info, styles["BodyText"]))
    story.append(Spacer(1, 0.4 * cm))

    # ── Document metadata ───────────────────────────────────────────────
    story.append(Paragraph("Document Metadata", styles["H1c"]))
    metadata = report.get("metadata", {})
    meta_rows = [["Field", "Value"]]
    for field, value in metadata.items():
        meta_rows.append([
            Paragraph(f"<b>{field}</b>", styles["BodyText"]),
            Paragraph(str(value or "—"), styles["BodyText"]),
        ])
    meta_table = Table(meta_rows, colWidths=[4.5 * cm, 12.5 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3A5F")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN",      (0, 0), (-1, 0), "LEFT"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F5F7FA")),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
        ("VALIGN",     (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.4 * cm))

    # ── Contract summary ────────────────────────────────────────────────
    story.append(Paragraph("Contract Summary", styles["H1c"]))
    story.append(Paragraph(
        report.get("contract_summary") or "(no summary generated)",
        styles["BodyText"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    # ── Three risk tables ───────────────────────────────────────────────
    story.append(Paragraph("Risk-Classified Clauses", styles["H1c"]))
    risk_tables = report.get("risk_tables", {}) or {}
    for tier in ("HIGH", "MEDIUM", "LOW"):
        rows = risk_tables.get(tier, [])
        tier_color = tier_colors.get(tier, colors.black)

        heading = Paragraph(
            f'<font color="#{tier_color.hexval()[2:]}">'
            f'{tier} Risk Clauses (n={len(rows)})</font>',
            styles["H2c"],
        )

        if not rows:
            story.extend([heading, Paragraph(
                f"<i>No {tier.lower()}-risk clauses identified.</i>",
                styles["BodyText"],
            ), Spacer(1, 0.3 * cm)])
            continue

        data = [[
            Paragraph("<b>Clause Type</b>", styles["BodyText"]),
            Paragraph("<b>Clause Text</b>", styles["BodyText"]),
            Paragraph("<b>Reasoning</b>", styles["BodyText"]),
            Paragraph("<b>Conf.</b>", styles["BodyText"]),
        ]]
        for row in rows:
            data.append([
                Paragraph(row.get("clause_type", ""), styles["BodySmall"]),
                Paragraph(_truncate(row.get("clause_text", ""), 600).replace("\n", "<br/>"),
                          styles["BodySmall"]),
                Paragraph(row.get("reasoning", ""), styles["BodySmall"]),
                Paragraph(
                    f"{row['confidence']:.2f}" if row.get("confidence") is not None else "—",
                    styles["BodySmall"],
                ),
            ])

        risk_table = Table(
            data,
            colWidths=[3.5 * cm, 7.0 * cm, 5.0 * cm, 1.5 * cm],
            repeatRows=1,
        )
        risk_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), tier_color),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("ALIGN",      (-1, 1), (-1, -1), "CENTER"),
            ("VALIGN",     (0, 0), (-1, -1), "TOP"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#CCCCCC")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",   (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F8F8F8")]),
        ]))
        story.extend([heading, risk_table, Spacer(1, 0.4 * cm)])

    # ── Conclusion ──────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Conclusion & Recommendations", styles["H1c"]))
    conclusion = report.get("conclusion", {}) or {}
    story.append(Paragraph(
        conclusion.get("overall_assessment") or "(no overall assessment)",
        styles["BodyText"],
    ))

    high_actions = conclusion.get("high_priority_actions") or []
    if high_actions:
        story.append(Paragraph("High-Priority Actions", styles["H2c"]))
        for action in high_actions:
            story.append(Paragraph(f"• {action}", styles["BulletItem"]))

    medium_actions = conclusion.get("medium_priority_actions") or []
    if medium_actions:
        story.append(Paragraph("Medium-Priority Actions", styles["H2c"]))
        for action in medium_actions:
            story.append(Paragraph(f"• {action}", styles["BulletItem"]))

    # ── Disclaimer ──────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph("Disclaimer", styles["H2c"]))
    story.append(Paragraph(
        f"<i>{report.get('disclaimer', '')}</i>",
        styles["Disclaimer"],
    ))

    doc.build(story)
    return out_path


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    state = build_synthetic_state()

    print(f"\nRunning Stage 4 with synthetic upstream data...")
    out = node_report_generation(state)
    report = out["final_report"]

    docx_path = Path(report["docx_path"])
    print(f"\nDOCX written → {docx_path} ({docx_path.stat().st_size:,} bytes)")

    # Render PDF directly via reportlab (LibreOffice not available on this host).
    pdf_path = render_pdf_from_report(report, output_dir="data/reports")
    print(f"PDF  written → {pdf_path} ({pdf_path.stat().st_size:,} bytes)")

    # Also persist the raw JSON for inspection.
    json_path = Path("data/reports") / f"{report['document_id']}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON written → {json_path} ({json_path.stat().st_size:,} bytes)")

    print(f"\n--- Report summary ---")
    print(f"  Risk score:    {report['overall_risk_score']:.2f}/10")
    print(f"  Total clauses: {report['total_clauses']}")
    print(f"  HIGH:   {len(report['risk_tables']['HIGH'])} clauses")
    print(f"  MEDIUM: {len(report['risk_tables']['MEDIUM'])} clauses")
    print(f"  LOW:    {len(report['risk_tables']['LOW'])} clauses")
    print(f"  Models used: {report['models_used']}")


if __name__ == "__main__":
    main()
