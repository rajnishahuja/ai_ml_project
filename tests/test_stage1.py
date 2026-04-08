"""Tests for Stage 1+2: clause extraction and classification."""

import pytest


class TestDeBERTaQAModel:
    """Tests for model.py DeBERTaQAModel."""

    def test_predict_returns_answer_span(self):
        """predict() should return a dict with 'answer' and 'score'."""
        raise NotImplementedError

    def test_predict_no_answer(self):
        """predict() should handle no-answer case gracefully."""
        raise NotImplementedError


class TestPredictClauses:
    """Tests for predict.py predict_clauses()."""

    def test_returns_list_of_clause_objects(self):
        """Should return list of ClauseObject dataclasses."""
        raise NotImplementedError

    def test_deduplication_removes_overlapping_spans(self):
        """deduplicate_spans() should merge overlapping clause spans."""
        raise NotImplementedError

    def test_confidence_threshold_filters_low_scores(self):
        """Clauses below threshold should be excluded."""
        raise NotImplementedError


class TestCUADClauseTypes:
    """Tests for the CUAD clause type list."""

    def test_41_clause_types(self):
        """Should have exactly 41 CUAD clause types."""
        from src.stage1_extract_classify.predict import CUAD_CLAUSE_TYPES
        assert len(CUAD_CLAUSE_TYPES) == 41
