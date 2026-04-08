"""Tests for Stage 4: aggregation, explanation, recommendation, report."""

import pytest

from src.common.schema import RiskAssessedClause


class TestAggregator:
    """Tests for aggregator.py."""

    def test_group_by_risk_level(self):
        """group_by_risk_level() should bucket clauses correctly."""
        raise NotImplementedError

    def test_contract_risk_score_range(self):
        """compute_contract_risk_score() should return value in [0, 1]."""
        raise NotImplementedError

    def test_top_risks_ordering(self):
        """get_top_risks() should return highest-risk first."""
        raise NotImplementedError


class TestExplainer:
    """Tests for explainer.py."""

    def test_build_prompt_includes_clause_info(self):
        """Prompt should include clause text, type, and risk level."""
        raise NotImplementedError


class TestRecommender:
    """Tests for recommender.py lookup table."""

    def test_known_clause_type_returns_recommendation(self):
        """Known (clause_type, risk_level) should return specific text."""
        raise NotImplementedError

    def test_unknown_clause_type_returns_default(self):
        """Unknown mapping should return default recommendation."""
        raise NotImplementedError


class TestReportBuilder:
    """Tests for report_builder.py."""

    def test_build_report_returns_risk_report(self):
        """build_report() should return a RiskReport dataclass."""
        raise NotImplementedError

    def test_save_report_creates_file(self, tmp_path):
        """save_report() should write a valid JSON file."""
        raise NotImplementedError


class TestReportEvaluation:
    """Tests for evaluate.py."""

    def test_evaluate_explanations_returns_rouge(self):
        """Should return dict with rouge1, rouge2, rougeL keys."""
        raise NotImplementedError

    def test_report_completeness_checks(self):
        """Completeness check should flag missing fields."""
        raise NotImplementedError
