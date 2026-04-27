"""Tests for Stage 4: aggregator, explainer, recommender, report_builder, evaluate."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.common.schema import RiskReport
from src.stage4_report_gen.aggregator import (
    EXPECTED_PROTECTIONS,
    compute_contract_risk_score,
    find_missing_protections,
    get_top_risks,
    group_by_risk_level,
    low_risk_summary,
)
from src.stage4_report_gen.explainer import (
    build_explanation_prompt,
    generate_explanation,
)
from src.stage4_report_gen.recommender import (
    DEFAULT_RECOMMENDATIONS,
    RECOMMENDATION_TABLE,
    get_recommendation,
)
from src.stage4_report_gen.report_builder import (
    build_report,
    build_report_dict,
    save_report,
)
from src.stage4_report_gen.evaluate import (
    evaluate_explanations,
    evaluate_report_completeness,
    recommendation_coverage,
)


# Lightweight clause stand-in. Mirrors the shape of both schema variants
# enough to exercise the duck-typed helpers.
@dataclass
class ClauseStub:
    clause_id: str
    document_id: str
    clause_type: str
    clause_text: str
    risk_level: str
    risk_explanation: str = ""
    risk_reason: str = ""
    confidence: float = 0.9


@pytest.fixture
def sample_clauses():
    return [
        ClauseStub("c1", "d1", "Indemnification", "Vendor indemnifies buyer.",
                   "HIGH", risk_explanation="One-sided.", confidence=0.9),
        ClauseStub("c2", "d1", "Termination For Convenience", "30 days notice.",
                   "MEDIUM", risk_explanation="Short notice.", confidence=0.8),
        ClauseStub("c3", "d1", "Cap On Liability", "12 months fees.",
                   "LOW", risk_explanation="Reasonable.", confidence=0.95),
        ClauseStub("c4", "d1", "Irrevocable Or Perpetual License", "Perpetual license.",
                   "HIGH", risk_explanation="No revocation.", confidence=0.7),
    ]


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class TestAggregator:

    def test_group_by_risk_level(self, sample_clauses):
        groups = group_by_risk_level(sample_clauses)
        assert {len(groups[k]) for k in ("HIGH", "MEDIUM", "LOW")} == {2, 1, 1}
        assert all(c.risk_level == "HIGH" for c in groups["HIGH"])

    def test_group_normalizes_case_and_missing(self):
        clauses = [
            ClauseStub("a", "d", "X", "x", "high"),       # lowercase
            ClauseStub("b", "d", "X", "x", ""),           # empty → LOW
        ]
        groups = group_by_risk_level(clauses)
        assert len(groups["HIGH"]) == 1
        assert len(groups["LOW"]) == 1

    def test_contract_risk_score_range(self, sample_clauses):
        score = compute_contract_risk_score(sample_clauses)
        assert 0.0 <= score <= 10.0

    def test_contract_risk_score_empty(self):
        assert compute_contract_risk_score([]) == 0.0

    def test_contract_risk_score_all_high_is_max(self):
        clauses = [ClauseStub("a", "d", "X", "x", "HIGH", confidence=1.0)]
        # severity=1.0 * conf=1.0 / weight=1.0 → 1.0 → *10 = 10.0
        assert compute_contract_risk_score(clauses) == 10.0

    def test_top_risks_ordering(self, sample_clauses):
        top = get_top_risks(sample_clauses, n=3)
        assert [c.clause_id for c in top] == ["c1", "c4", "c2"]

    def test_top_risks_capped_by_n(self, sample_clauses):
        assert len(get_top_risks(sample_clauses, n=2)) == 2

    def test_find_missing_protections(self, sample_clauses):
        missing = find_missing_protections(sample_clauses)
        # Indemnification, Termination, Cap On Liability are present;
        # Governing Law, Insurance, Warranty Duration are missing.
        assert "Governing Law" in missing
        assert "Indemnification" not in missing
        assert all(p in EXPECTED_PROTECTIONS for p in missing)

    def test_low_risk_summary(self, sample_clauses):
        low = group_by_risk_level(sample_clauses)["LOW"]
        assert "1 clause" in low_risk_summary(low)
        assert low_risk_summary([]).startswith("No low-risk")


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------

class TestExplainer:

    def test_build_prompt_includes_clause_info(self, sample_clauses):
        prompt = build_explanation_prompt(sample_clauses[0])
        assert "Indemnification" in prompt
        assert "HIGH" in prompt
        assert "Vendor indemnifies buyer" in prompt

    def test_generate_explanation_no_model_uses_existing_reason(self, sample_clauses):
        # With model=None, falls back to risk_explanation / risk_reason.
        out = generate_explanation(sample_clauses[0], model=None)
        assert out == "One-sided."

    def test_generate_explanation_falls_back_when_reason_empty(self):
        clause = ClauseStub("a", "d", "Indemnification", "x", "HIGH", risk_explanation="")
        out = generate_explanation(clause, model=None)
        assert out  # non-empty
        assert "Indemnification" in out or "HIGH" in out


# ---------------------------------------------------------------------------
# Recommender
# ---------------------------------------------------------------------------

class TestRecommender:

    def test_known_clause_type_returns_curated(self):
        rec = get_recommendation("Indemnification", "HIGH")
        # Curated entry, not the default
        assert rec != DEFAULT_RECOMMENDATIONS["HIGH"]
        assert ("indemnification", "HIGH") in RECOMMENDATION_TABLE

    def test_case_insensitive_lookup(self):
        a = get_recommendation("INDEMNIFICATION", "HIGH")
        b = get_recommendation("indemnification", "high")
        c = get_recommendation("Indemnification", "HIGH")
        assert a == b == c

    def test_unknown_clause_type_falls_back_to_default(self):
        rec = get_recommendation("Made Up Type", "HIGH")
        assert rec == DEFAULT_RECOMMENDATIONS["HIGH"]

    def test_unknown_risk_level_falls_back(self):
        rec = get_recommendation("Indemnification", "WEIRD")
        assert rec == DEFAULT_RECOMMENDATIONS["MEDIUM"]

    def test_low_returns_low_default(self):
        # No specific LOW entries — should use the LOW default.
        rec = get_recommendation("Indemnification", "LOW")
        assert rec == DEFAULT_RECOMMENDATIONS["LOW"]


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

class TestReportBuilder:

    def test_build_report_dict_shape(self, sample_clauses):
        r = build_report_dict(sample_clauses, "doc_001")
        for key in (
            "document_id", "summary", "high_risk", "medium_risk",
            "low_risk_summary", "missing_protections",
            "overall_risk_score", "total_clauses", "metadata",
        ):
            assert key in r
        assert r["document_id"] == "doc_001"
        assert r["total_clauses"] == 4

    def test_build_report_returns_risk_report(self, sample_clauses):
        report = build_report(sample_clauses, "doc_001")
        assert isinstance(report, RiskReport)
        assert report.document_id == "doc_001"
        assert len(report.high_risk) == 2
        assert all(rc.recommendation for rc in report.high_risk)

    def test_high_risk_cap_top_n(self):
        many_high = [
            ClauseStub(f"c{i}", "d", "Indemnification", "x", "HIGH", confidence=0.9)
            for i in range(20)
        ]
        r = build_report_dict(many_high, "d", top_n=5)
        assert len(r["high_risk"]) == 5

    def test_save_report_creates_file(self, sample_clauses, tmp_path):
        report = build_report(sample_clauses, "doc_001")
        out = tmp_path / "r.json"
        save_report(report, str(out))
        assert out.exists()
        loaded = json.loads(out.read_text())
        assert loaded["document_id"] == "doc_001"


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:

    def test_completeness_passes_for_full_report(self, sample_clauses):
        r = build_report_dict(sample_clauses, "doc_001")
        checks = evaluate_report_completeness(r)
        assert all(checks.values()), checks

    def test_completeness_flags_missing_summary(self, sample_clauses):
        r = build_report_dict(sample_clauses, "doc_001")
        r["summary"] = ""
        checks = evaluate_report_completeness(r)
        assert checks["has_summary"] is False

    def test_completeness_flags_score_out_of_range(self, sample_clauses):
        r = build_report_dict(sample_clauses, "doc_001")
        r["overall_risk_score"] = 99.0
        assert evaluate_report_completeness(r)["score_in_range"] is False

    def test_recommendation_coverage_curated_full(self, sample_clauses):
        r = build_report_dict(sample_clauses, "doc_001")
        cov = recommendation_coverage(r)
        # Both HIGH-risk samples (Indemnification, Irrevocable Or Perpetual License)
        # are in the curated table, so coverage = 1.0
        assert cov["coverage"] == 1.0
        assert cov["total_high"] == 2

    def test_recommendation_coverage_default_only(self):
        clauses = [ClauseStub("a", "d", "Made Up Type", "x", "HIGH")]
        r = build_report_dict(clauses, "d")
        cov = recommendation_coverage(r)
        assert cov["curated"] == 0
        assert cov["default"] == 1
        assert cov["coverage"] == 0.0

    def test_evaluate_explanations_length_mismatch(self):
        with pytest.raises(ValueError):
            evaluate_explanations(["a"], ["b", "c"])

    def test_evaluate_explanations_runs(self):
        result = evaluate_explanations(
            ["the cat sat on the mat"],
            ["a cat sits on the mat"],
        )
        assert set(result.keys()) == {"rouge1", "rouge2", "rougeL"}
        for v in result.values():
            assert 0.0 <= v <= 1.0
