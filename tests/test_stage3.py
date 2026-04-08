"""Tests for Stage 3: risk agent, tools, classifier, embeddings."""

import pytest


class TestRiskClassifier:
    """Tests for risk_classifier.py."""

    def test_predict_returns_risk_level(self):
        """predict() should return dict with risk_level and confidence."""
        raise NotImplementedError

    def test_valid_risk_levels(self):
        """risk_level must be one of LOW, MEDIUM, HIGH."""
        raise NotImplementedError


class TestTools:
    """Tests for tools.py FAISS retrieval and contract search."""

    def test_faiss_retrieval_returns_similar_clauses(self):
        """faiss_retrieval() should return list of SimilarClause."""
        raise NotImplementedError

    def test_contract_search_same_document(self):
        """contract_search() should only return clauses from same doc."""
        raise NotImplementedError


class TestAgent:
    """Tests for the LangGraph risk agent."""

    def test_assess_clauses_returns_risk_assessed(self):
        """assess_clauses() should return list of RiskAssessedClause."""
        raise NotImplementedError

    def test_agent_max_iterations(self):
        """Agent should stop after max_iterations from config."""
        raise NotImplementedError


class TestSyntheticLabels:
    """Tests for synthetic label generation."""

    def test_build_prompt_contains_clause(self):
        """Prompt should contain the clause text."""
        raise NotImplementedError

    def test_parse_response_valid(self):
        """parse_llm_response() should handle well-formed LLM output."""
        raise NotImplementedError
