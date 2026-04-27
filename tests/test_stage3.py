"""Tests for Stage 3: tools (currently only contract_search is implemented)."""

from __future__ import annotations

import pytest

from src.stage3_risk_agent.tools import (
    METADATA_CLAUSE_TYPES,
    _load_relations,
    contract_search,
    extract_metadata_block,
    make_contract_search,
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


class TestContractSearchWithClauseType:
    """clause_type filter pulls the related-types list from the static JSON."""

    @pytest.fixture
    def doc_id(self):
        return "LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT"

    def test_filter_returns_related_types_only(self, doc_id):
        """License Grant filter on this contract → License Grant + Exclusivity only."""
        result = contract_search(doc_id, clause_type="License Grant")
        types = {c["clause_type"] for c in result}
        assert "License Grant" in types               # self always included
        assert "Exclusivity" in types                  # in relations[License Grant]
        # Types in this contract that are NOT in License Grant's related list
        # should be filtered out.
        assert "Renewal Term" not in types
        assert "Insurance" not in types
        assert "Governing Law" not in types

    def test_filter_includes_self_type(self, doc_id):
        """The target's own type is always allowed (multi-clause same-type case)."""
        result = contract_search(doc_id, clause_type="Cap On Liability")
        # This contract has no Cap On Liability clause but the filter
        # should not crash and should return only related types if any
        # are present.
        types = {c["clause_type"] for c in result}
        relations = _load_relations("data/reference/clause_type_relations.json")
        allowed = relations["Cap On Liability"] | {"Cap On Liability"}
        assert types.issubset(allowed)

    def test_unknown_clause_type_falls_back_to_all_siblings(self, doc_id):
        """If clause_type is not in the relations file, return all non-metadata."""
        with_filter = contract_search(doc_id, clause_type="Made Up Type")
        without_filter = contract_search(doc_id)
        assert {c["clause_id"] for c in with_filter} == {c["clause_id"] for c in without_filter}

    def test_clause_type_none_matches_old_behavior(self, doc_id):
        """clause_type=None (default) → same result as before the feature."""
        a = contract_search(doc_id, clause_type=None)
        b = contract_search(doc_id)
        assert {c["clause_id"] for c in a} == {c["clause_id"] for c in b}

    def test_in_memory_mode_with_clause_type(self):
        """clause_type filter works on in-memory clauses too."""
        clauses = [
            {"clause_id": "i1", "document_id": "D", "clause_type": "Indemnification",
             "clause_text": "x"},  # Indemnification not in CUAD — no relations entry
            {"clause_id": "c1", "document_id": "D", "clause_type": "Cap On Liability",
             "clause_text": "x"},
            {"clause_id": "i2", "document_id": "D", "clause_type": "Insurance",
             "clause_text": "x"},
            {"clause_id": "g1", "document_id": "D", "clause_type": "Governing Law",
             "clause_text": "x"},
        ]
        result = contract_search("D", clause_type="Cap On Liability", all_clauses=clauses)
        types = {c["clause_type"] for c in result}
        # Cap On Liability + Insurance (related); Governing Law and Indemnification excluded
        assert "Cap On Liability" in types
        assert "Insurance" in types
        assert "Governing Law" not in types
        assert "Indemnification" not in types

    def test_relations_file_missing_disables_filter(self, doc_id, tmp_path, caplog):
        """If the relations file is missing, returns all non-metadata + warns."""
        bogus = tmp_path / "no_such_relations.json"
        result = contract_search(doc_id, clause_type="License Grant",
                                 relations_path=str(bogus))
        # Same as the no-filter case
        baseline = contract_search(doc_id)
        assert {c["clause_id"] for c in result} == {c["clause_id"] for c in baseline}


class TestRelationsFile:
    """Sanity checks on data/reference/clause_type_relations.json."""

    def test_covers_all_41_cuad_types(self):
        relations = _load_relations("data/reference/clause_type_relations.json")
        assert len(relations) == 41

    def test_no_self_references(self):
        relations = _load_relations("data/reference/clause_type_relations.json")
        for src, targets in relations.items():
            assert src not in targets, f"{src!r} is listed as related to itself"

    def test_all_referenced_types_exist(self):
        relations = _load_relations("data/reference/clause_type_relations.json")
        all_types = set(relations.keys())
        for src, targets in relations.items():
            for t in targets:
                assert t in all_types, f"{src!r} references unknown type {t!r}"


class TestMakeContractSearch:
    """Closure factory: binds extracted_clauses; Mistral passes only IDs."""

    @pytest.fixture
    def contract_clauses(self):
        return [
            {"clause_id": "d__Parties",   "document_id": "d", "clause_type": "Parties",
             "clause_text": "Acme and Buyer."},
            {"clause_id": "d__Cap",       "document_id": "d", "clause_type": "Cap On Liability",
             "clause_text": "Total liability capped at 12mo fees."},
            {"clause_id": "d__Uncapped",  "document_id": "d", "clause_type": "Uncapped Liability",
             "clause_text": "IP indemnity uncapped."},
            {"clause_id": "d__Insurance", "document_id": "d", "clause_type": "Insurance",
             "clause_text": "$5M coverage."},
            {"clause_id": "d__GL",        "document_id": "d", "clause_type": "Governing Law",
             "clause_text": "Delaware law."},
        ]

    def test_factory_returns_callable(self, contract_clauses):
        bound = make_contract_search(contract_clauses)
        assert callable(bound)

    def test_bound_tool_filters_by_clause_type(self, contract_clauses):
        bound = make_contract_search(contract_clauses)
        result = bound("d", "Cap On Liability")
        types = {c["clause_type"] for c in result}
        assert "Cap On Liability" in types
        assert "Uncapped Liability" in types
        assert "Insurance" in types
        # Governing Law is NOT in relations[Cap On Liability] → filtered out.
        assert "Governing Law" not in types

    def test_bound_tool_excludes_metadata_by_default(self, contract_clauses):
        bound = make_contract_search(contract_clauses)
        result = bound("d", "")  # no clause_type filter
        types = {c["clause_type"] for c in result}
        assert "Parties" not in types

    def test_include_metadata_flag_via_factory(self, contract_clauses):
        bound = make_contract_search(contract_clauses, include_metadata=True)
        result = bound("d", "")
        types = {c["clause_type"] for c in result}
        assert "Parties" in types

    def test_wrong_document_id_returns_empty(self, contract_clauses):
        bound = make_contract_search(contract_clauses)
        assert bound("not_d", "Cap On Liability") == []

    def test_closure_independent_of_caller_mutation(self, contract_clauses):
        """Mutating the caller's list after binding does not affect the closure."""
        bound = make_contract_search(contract_clauses)
        contract_clauses.clear()
        result = bound("d", "Cap On Liability")
        assert len(result) > 0  # closure has its own snapshot


class TestExtractMetadataBlock:
    """The 5 metadata fields → flat dict for prompt injection."""

    def test_extracts_present_fields_in_canonical_order(self):
        clauses = [
            {"clause_id": "1", "clause_type": "Cap On Liability", "clause_text": "..."},
            {"clause_id": "2", "clause_type": "Effective Date", "clause_text": "Jan 1 2024"},
            {"clause_id": "3", "clause_type": "Parties", "clause_text": "Acme and Buyer"},
        ]
        result = extract_metadata_block(clauses)
        # Canonical order: Parties, Effective Date, Expiration Date, Agreement Date, Document Name.
        assert list(result.keys()) == ["Parties", "Effective Date"]
        assert result["Parties"] == "Acme and Buyer"
        assert result["Effective Date"] == "Jan 1 2024"

    def test_first_occurrence_wins(self):
        clauses = [
            {"clause_id": "1", "clause_type": "Parties", "clause_text": "FIRST"},
            {"clause_id": "2", "clause_type": "Parties", "clause_text": "SECOND"},
        ]
        assert extract_metadata_block(clauses)["Parties"] == "FIRST"

    def test_no_metadata_returns_empty_dict(self):
        clauses = [
            {"clause_id": "1", "clause_type": "Cap On Liability", "clause_text": "..."},
        ]
        assert extract_metadata_block(clauses) == {}

    def test_empty_input(self):
        assert extract_metadata_block([]) == {}

    def test_accepts_pydantic_models(self):
        from app.schemas.domain import ExtractedClause
        clauses = [
            ExtractedClause(clause_id="1", clause_text="Acme and Buyer",
                            clause_type="Parties",
                            start_pos=0, end_pos=10, confidence=0.99,
                            confidence_logit=5.0),
        ]
        result = extract_metadata_block(clauses)
        assert result["Parties"] == "Acme and Buyer"


class TestPrecedentSearch:
    def test_not_implemented_in_this_branch(self):
        with pytest.raises(NotImplementedError):
            precedent_search("some text", "some/path")
