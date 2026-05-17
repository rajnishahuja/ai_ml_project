"""
Shared data contracts for all pipeline stages.

All stages communicate through these typed dataclasses.
This file is the single source of truth for inter-stage data formats.
See ARCHITECTURE.md for the full JSON schema documentation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Any


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
    extractor_confidence: float = 0.0  # Alias for extraction/QA model confidence
    confidence_logit: Optional[float] = None
    page_no: Optional[str] = None
    content_label: Optional[str] = None

    def __post_init__(self):
        if self.extractor_confidence == 0.0 and self.confidence != 0.0:
            self.extractor_confidence = self.confidence
        elif self.confidence == 0.0 and self.extractor_confidence != 0.0:
            self.confidence = self.extractor_confidence

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
    clause_type: str
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
    cross_references: list[Any] = field(default_factory=list)
    confidence: float = 0.0  # Double-sided fallback field
    risk_confidence: float = 0.0  # Stage 3 risk classifier confidence (alias)
    deberta_confidence: float = 0.0  # Stage 3 risk classifier confidence (primary)
    recommendation: str = ""  # Optional recommendation from agent or lookup table
    extraction_confidence: float = 0.0  # Stage 1 extraction confidence
    extractor_confidence: float = 0.0  # Stage 1 extraction confidence alias
    extraction_confidence_logit: Optional[float] = None  # Stage 1 logit
    content_label: Optional[str] = None  # Stage 1 content label
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)
    page_no: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Optional[dict] = None

    def __post_init__(self):
        # 1. Synchronize Stage 3 Risk Prediction confidence
        if self.deberta_confidence == 0.0 and self.risk_confidence != 0.0:
            self.deberta_confidence = self.risk_confidence
        elif self.risk_confidence == 0.0 and self.deberta_confidence != 0.0:
            self.risk_confidence = self.deberta_confidence

        # 2. Synchronize Stage 1 Extraction confidence
        if self.extractor_confidence == 0.0 and self.extraction_confidence != 0.0:
            self.extractor_confidence = self.extraction_confidence
        elif self.extraction_confidence == 0.0 and self.extractor_confidence != 0.0:
            self.extraction_confidence = self.extractor_confidence

        # 3. Synchronize 'confidence' fallback property
        if self.confidence == 0.0:
            if self.extractor_confidence != 0.0:
                self.confidence = self.extractor_confidence
            elif self.deberta_confidence != 0.0:
                self.confidence = self.deberta_confidence
        else:
            if self.extractor_confidence == 0.0:
                self.extractor_confidence = self.confidence
                self.extraction_confidence = self.confidence

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
# ---------------------------------------------------------------------------


@dataclass
class ReportClause:
    """A single clause entry in the final risk report."""

    clause_id: str
    clause_type: str
    risk_level: str
    explanation: str
    recommendation: str
    similar_clauses: list[SimilarClause] = field(default_factory=list)
    cross_references: list[Any] = field(default_factory=list)
    page_no: Optional[str] = None
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)
    risk_confidence: float = 0.0  # Stage 3 risk classifier confidence
    extraction_confidence: float = 0.0  # Stage 1 extraction confidence
    extraction_confidence_logit: Optional[float] = None  # Stage 1 logit
    content_label: Optional[str] = None  # Stage 1 content label
    clause_text: Optional[str] = None  # Original clause text


@dataclass
class ReportMetadata:
    """Metadata about the report generation run."""

    generated_at: str
    models_used: dict[str, str] = field(default_factory=dict)


@dataclass
class RiskReport:
    """The final structured risk report for a contract."""

    document_id: str
    summary: str
    high_risk: list[ReportClause] = field(default_factory=list)
    medium_risk: list[ReportClause] = field(default_factory=list)
    low_risk: list[ReportClause] = field(default_factory=list)
    low_risk_summary: str = ""
    missing_protections: list[str] = field(default_factory=list)
    overall_risk_score: float = 0.0
    total_clauses: int = 0
    metadata: Optional[ReportMetadata] = None

    def to_dict(self) -> dict:
        return asdict(self)
