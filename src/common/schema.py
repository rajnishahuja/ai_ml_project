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
    extraction_confidence: float = 0.0  # Stage 1 extraction confidence (primary)
    classifier_confidence: float = 0.0  # Stage 3 risk classifier confidence (primary)
    agent_confidence: float = 0.0       # Agent's RAG/Tool override confidence score
    is_override: bool = False           # True if the LangGraph agent overrode DeBERTa's preliminary label
    recommendation: str = ""  # Optional recommendation from agent or lookup table
    extraction_confidence_logit: Optional[float] = None  # Stage 1 logit
    content_label: Optional[str] = None  # Stage 1 content label
    agent_trace: list[AgentTraceEntry] = field(default_factory=list)
    page_no: Optional[str] = None
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    metadata: Optional[dict] = None

    @property
    def confidence(self) -> float:
        """Fallback field mapping to extraction confidence."""
        return self.extraction_confidence

    @confidence.setter
    def confidence(self, val: float) -> None:
        self.extraction_confidence = val

    @property
    def deberta_confidence(self) -> float:
        """Alias mapping to the classifier confidence."""
        return self.classifier_confidence

    @deberta_confidence.setter
    def deberta_confidence(self, val: float) -> None:
        self.classifier_confidence = val

    @property
    def risk_confidence(self) -> float:
        """
        Intelligently resolves which confidence score to display in the UI:
        - If the agent overrode DeBERTa (is_override is True), return agent's confidence.
        - Otherwise, return DeBERTa's classifier confidence.
        """
        if self.is_override and self.agent_confidence != 0.0:
            return self.agent_confidence
        return self.classifier_confidence

    @risk_confidence.setter
    def risk_confidence(self, val: float) -> None:
        self.classifier_confidence = val

    @property
    def extractor_confidence(self) -> float:
        """Alias mapping to extraction confidence."""
        return self.extraction_confidence

    @extractor_confidence.setter
    def extractor_confidence(self, val: float) -> None:
        self.extraction_confidence = val

    def __post_init__(self):
        pass

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
    is_override: bool = False
    agent_confidence: float = 0.0
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
