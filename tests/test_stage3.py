"""Tests for Stage 3: tools (currently only contract_search is implemented)."""

from __future__ import annotations

import pytest

from src.stage3_risk_agent.tools import (
    METADATA_CLAUSE_TYPES,
    contract_search,
    precedent_search,
)


@pytest.fixture
def fake_clauses() -> list[dict]:
    return [
        {"clause_id": "a1", "document_id": "D1", "clause_type": "Indemnification",
         "clause_text": "Vendor indemnifies buyer."},
        {"clause_id": "a2", "document_id": "D1", "clause_type": "Parties",
         "clause_text": "Acme Inc and Buyer Co."},
        {"clause_id": "a3", "document_id": "D1", "clause_type": "Cap On Liability",
         "clause_text": "Liability capped at 12 months fees."},
        {"clause_id": "b1", "document_id": "D2", "clause_type": "Indemnification",
         "clause_text": "Mutual indemnity."},
    ]


class TestContractSearchInMemory:
    """Inference-mode contract_search — passes `all_clauses` directly."""

    def test_filters_by_document_id(self, fake_clauses):
        result = contract_search("D1", all_clauses=fake_clauses)
        assert {c["clause_id"] for c in result} == {"a1", "a3"}

    def test_excludes_metadata_by_default(self, fake_clauses):
        result = contract_search("D1", all_clauses=fake_clauses)
        types = {c["clause_type"] for c in result}
        assert types.isdisjoint(METADATA_CLAUSE_TYPES)
        assert "Parties" not in types

    def test_include_metadata_flag(self, fake_clauses):
        result = contract_search("D1", all_clauses=fake_clauses, include_metadata=True)
        types = {c["clause_type"] for c in result}
        assert "Parties" in types

    def test_unknown_doc_returns_empty(self, fake_clauses):
        assert contract_search("not_a_doc", all_clauses=fake_clauses) == []

    def test_empty_doc_id_returns_empty(self, fake_clauses):
        assert contract_search("", all_clauses=fake_clauses) == []

    def test_output_shape(self, fake_clauses):
        result = contract_search("D1", all_clauses=fake_clauses)
        assert result, "expected non-empty result"
        for c in result:
            assert set(c.keys()) >= {
                "clause_id", "document_id", "clause_type",
                "clause_text", "start_pos",
            }

    def test_accepts_dataclass_input(self, fake_clauses):
        from dataclasses import dataclass

        @dataclass
        class C:
            clause_id: str
            document_id: str
            clause_type: str
            clause_text: str

        objs = [C(**{k: v for k, v in d.items()}) for d in fake_clauses]
        result = contract_search("D1", all_clauses=objs)
        assert len(result) == 2


class TestContractSearchCorpus:
    """Disk-backed mode — reads from the pre-built index, falls back to spans."""

    def test_real_contract_returns_clauses(self):
        doc_id = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
        result = contract_search(doc_id)
        assert len(result) > 0
        assert all(c["document_id"] == doc_id for c in result)
        assert all(c["clause_type"] not in METADATA_CLAUSE_TYPES for c in result)

    def test_unknown_doc_id_returns_empty(self):
        assert contract_search("definitely_not_a_real_contract_xyz") == []

    def test_corpus_fallback_when_index_missing(self):
        """Bogus index path → falls through to the spans corpus, same result."""
        from src.stage3_risk_agent.tools import _load_index, _load_corpus
        _load_index.cache_clear()
        _load_corpus.cache_clear()
        doc_id = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
        from_index = contract_search(doc_id)
        from_corpus = contract_search(doc_id, index_path="/nonexistent/idx.json")
        assert {c["clause_id"] for c in from_index} == {c["clause_id"] for c in from_corpus}

    def test_index_mode_includes_is_metadata_flag(self):
        """Index-loaded clauses carry the `is_metadata` flag."""
        doc_id = "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"
        result = contract_search(doc_id, include_metadata=True)
        assert any(c.get("is_metadata") is True for c in result)
        assert any(c.get("is_metadata") is False for c in result)


class TestPrecedentSearch:
    def test_not_implemented_in_this_branch(self):
        with pytest.raises(NotImplementedError):
            precedent_search("some text", "some/path")
