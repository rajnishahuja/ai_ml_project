"""
Shared data contracts for all pipeline stages.

All stages communicate through these typed dataclasses.
This file is the single source of truth for inter-stage data formats.
See ARCHITECTURE.md for the full JSON schema documentation.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Stage 1+2 Output → Stage 3 Input
# ---------------------------------------------------------------------------

@dataclass
class ClauseObject:
    """A single extracted and classified clause from a contract."""
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Full extraction result for one contract document."""
    document_id: str
    clauses: list[ClauseObject] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "clauses": [c.to_dict() for c in self.clauses],
        }


# ---------------------------------------------------------------------------
# Stage 3: FAISS retrieval result
# ---------------------------------------------------------------------------

@dataclass
class SimilarClause:
    """A clause retrieved from FAISS as similar to the query clause."""
    text: str
    risk_level: str
    similarity: float


# ---------------------------------------------------------------------------
# Stage 3 Output → Stage 4 Input
# ---------------------------------------------------------------------------

@dataclass
class AgentTraceEntry:
    """One tool invocation in the LangGraph agent's reasoning trace."""
    tool: str
    result_count: Optional[int] = None
    related_clauses: Optional[int] = None


@dataclass
class RiskAssessedClause:
    """A clause with risk assessment from the Stage 3 agent."""
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_explanation: str
    similar_clauses: list[SimilarClause] = field(default_factory=list)
    cross_references: list[str] = field(default_factory=list)
    confidence: float = 0.0
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Synthetic risk labels (training data for Stage 3)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticRiskLabel:
    """An LLM-generated risk label for a CUAD clause."""
    clause_text: str
    clause_type: str
    risk_level: str  # "LOW", "MEDIUM", "HIGH"
    risk_reason: str
    labeled_by: str  # e.g. "qwen-32b", "gemini", "openai"


# ---------------------------------------------------------------------------
# Stage 4 Output (Final Report)
#
# Gemini-driven hierarchical design. See ARCHITECTURE.md §"Stage 4 Output
# (Final Report)" for the JSON shape these serialize to, and
# docs/STAGE4_DECISION_LOG.md for design decisions.
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """One remediation suggestion looked up by (clause_type, risk_pattern).

    Produced by `src/stage4_report_gen/recommender.py` from the curated
    `recommendations_data.yaml` lookup table. `match_level` records which
    fallback tier hit so we can track lookup-table coverage over time.
    """
    text: str
    market_standard: str = ""
    fallback_position: str = ""
    priority: str = "MEDIUM"            # HIGH / MEDIUM / LOW / negotiable
    match_level: str = "universal"      # exact / type / risk_level / universal


@dataclass
class MissingProtection:
    """One expected-clause-type checklist entry reported as missing.

    Driven by `missing_protections.yaml` (per-contract-type checklists).
    `importance` controls whether this entry contributes to the overall
    risk score's missing-protection boost — see ARCHITECTURE.md.
    """
    clause_type: str
    importance: str                      # critical / important / standard
    rationale: str


@dataclass
class ClauseReport:
    """A risk-assessed clause, normalized + enriched for the final report.

    Built by `src/stage4_report_gen/aggregator.py` (clause type
    normalization, risk pattern derivation) and `explainer.py` (LLM-driven
    polished_explanation). `clause_type_original` preserves Stage 3's
    free-form output for audit; `clause_type` is the canonical CUAD type.
    """
    clause_id: str
    document_id: str
    clause_text: str
    clause_type: str                     # canonical CUAD type
    clause_type_original: str            # what Stage 3 emitted
    risk_level: str                      # HIGH / MEDIUM / LOW
    risk_pattern: str                    # controlled-vocab pattern code
    risk_explanation: str                # raw, from Stage 3
    polished_explanation: str            # Gemini-generated user-facing text
    recommendation: Recommendation
    confidence: float
    overridden: bool = False
    similar_clauses: list[dict] = field(default_factory=list)


@dataclass
class ScoreBreakdown:
    """Reproducibility: how the overall_risk_score was computed.

    Embedded in `ContractReport` so every report can be audited end-to-end
    without re-running the aggregator.
    """
    high_count: int
    medium_count: int
    low_count: int
    base_score: float                    # (3H + 2M + L) / total × (10/3)
    missing_critical_or_important: int
    missing_boost: float                 # 0.5 × count
    final_score: float                   # min(base + boost, 10.0)
    note: str = ""                       # e.g. "no clauses → score 0"


@dataclass
class ExecutiveSummaryDigest:
    """Structured input to `explainer.generate_executive_summary()`.

    Hard rule #1 enforcement boundary: this is the ONLY thing the LLM sees
    when generating the contract-level summary. Raw contract text is never
    sent to Gemini. Excerpts in `top_high_risk` are capped at 200 chars
    (validated by `explainer.generate_executive_summary` before the call).

    The 5 fields are:
        1. metadata             — parties, contract_type, dates, title
        2. statistics           — total/HIGH/MEDIUM/LOW counts
        3. top_high_risk        — up to 5 HIGH-risk clauses with ≤200-char
                                  excerpts: {clause_type, risk_pattern, excerpt}
        4. missing_protections  — clause_type names only, not full objects
        5. metadata_provided    — True if metadata sidecar supplied; False if
                                  fields auto-defaulted
    """
    metadata: dict[str, str]
    statistics: dict[str, int]
    top_high_risk: list[dict[str, str]]
    missing_protections: list[str]
    metadata_provided: bool


@dataclass
class ContractReport:
    """The full report payload — canonical Stage 4 output.

    Produced by `src/stage4_report_gen/report_builder.generate_report`.
    Serialized to four formats by the renderers in `renderers/`.
    """
    document_id: str
    summary: str                                       # Gemini executive summary
    statistics: dict[str, int]
    high_risk: list[ClauseReport] = field(default_factory=list)
    medium_risk: list[ClauseReport] = field(default_factory=list)
    low_risk: list[ClauseReport] = field(default_factory=list)
    low_risk_summary: str = ""
    missing_protections: list[MissingProtection] = field(default_factory=list)
    overall_risk_score: float = 0.0
    score_breakdown: Optional[ScoreBreakdown] = None
    total_clauses: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""
    models_used: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReportArtifacts:
    """File paths and timing returned by `report_builder.generate_report`.

    `cache_hit_rate` is the fraction of LLM calls that were served from the
    on-disk cache during this run — useful for the live demo.
    """
    document_id: str
    report: ContractReport
    json_path: Path
    markdown_path: Path
    pdf_path: Path
    docx_path: Path
    duration_seconds: float
    cache_hit_rate: float = 0.0
