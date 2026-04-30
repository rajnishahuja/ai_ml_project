"""Stage 4 tests.

All Gemini SDK calls are mocked via pytest-mock. No test makes a live
network call (hard rule from STAGE4_RESUME_HANDOFF.md § 9).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.stage4_report_gen.cache import GeminiCache
from src.stage4_report_gen.rate_limiter import TokenBucketRateLimiter, backoff_delays
from src.stage4_report_gen.llm_client import (
    GeminiClient,
    SOFT_FAIL_PLACEHOLDER,
)
from src.stage4_report_gen import pattern_deriver
from src.stage4_report_gen.pattern_deriver import (
    PATTERN_CODES,
    UNKNOWN,
    derive_pattern,
    derive_tier1,
    derive_tier2,
)
from src.stage4_report_gen.aggregator import (
    METADATA_CLAUSE_TYPES,
    aggregate,
    build_executive_digest,
    compute_score_breakdown,
    find_missing_protections,
    infer_contract_type,
    normalize_clause_type,
)
from src.stage4_report_gen.recommender import (
    attach_recommendations,
    lookup,
)
from src.common.schema import ClauseReport, MissingProtection, Recommendation


# ===========================================================================
# Group 2 — cache.py
# ===========================================================================

class TestGeminiCache:
    def test_miss_then_hit(self, tmp_path: Path) -> None:
        cache = GeminiCache(tmp_path / "cache")
        assert cache.get("m", "p") is None
        assert cache.misses == 1
        cache.put("m", "p", "response-text")
        assert cache.get("m", "p") == "response-text"
        assert cache.hits == 1
        assert cache.hit_rate == pytest.approx(0.5)

    def test_key_includes_model(self, tmp_path: Path) -> None:
        cache = GeminiCache(tmp_path / "cache")
        cache.put("model-a", "shared-prompt", "A")
        cache.put("model-b", "shared-prompt", "B")
        assert cache.get("model-a", "shared-prompt") == "A"
        assert cache.get("model-b", "shared-prompt") == "B"

    def test_corrupted_entry_is_discarded(self, tmp_path: Path) -> None:
        cache = GeminiCache(tmp_path / "cache")
        path = cache._path("m", "p")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not valid json")
        assert cache.get("m", "p") is None
        assert not path.exists()

    def test_reset_stats(self, tmp_path: Path) -> None:
        cache = GeminiCache(tmp_path / "cache")
        cache.get("m", "p")
        cache.put("m", "p", "x")
        cache.get("m", "p")
        cache.reset_stats()
        assert cache.hits == 0 and cache.misses == 0


# ===========================================================================
# Group 2 — rate_limiter.py
# ===========================================================================

class TestTokenBucketRateLimiter:
    def test_initial_capacity_does_not_block(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=12)
        sleep_calls: list[float] = []
        for _ in range(12):
            limiter.acquire(sleep_fn=sleep_calls.append)
        assert sleep_calls == [], "first RPM tokens should not require sleep"

    def test_thirteenth_acquire_waits(self) -> None:
        limiter = TokenBucketRateLimiter(rate_per_minute=12)
        sleeps: list[float] = []
        for _ in range(12):
            limiter.acquire(sleep_fn=lambda s: None)
        limiter.acquire(sleep_fn=sleeps.append)
        assert sleeps and sleeps[0] > 0
        assert sleeps[0] == pytest.approx(60 / 12, rel=0.1)

    def test_invalid_rpm_raises(self) -> None:
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(rate_per_minute=0)


class TestBackoffDelays:
    def test_yields_exactly_n_with_jitter_bounds(self) -> None:
        base = [2, 4, 8]
        delays = list(backoff_delays(base, jitter=0.25))
        assert len(delays) == 3
        for got, want in zip(delays, base):
            assert want * 0.75 <= got <= want * 1.25

    def test_zero_jitter_returns_base(self) -> None:
        assert list(backoff_delays([1, 2, 3], jitter=0.0)) == [1, 2, 3]


# ===========================================================================
# Group 2 — llm_client.py (with mocked SDK)
# ===========================================================================

class _FakeSDK:
    """Minimal stand-in for google.generativeai. responses is a list whose
    items are either strings (success) or Exception instances (raise)."""

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.configure_calls: list[dict] = []
        self.GenerativeModel_calls: list[str] = []
        self.generate_calls: list[str] = []

    def configure(self, *, api_key: str) -> None:
        self.configure_calls.append({"api_key": api_key})

    def GenerativeModel(self, model_name: str):  # noqa: N802 — SDK shape
        self.GenerativeModel_calls.append(model_name)
        sdk = self

        class _Model:
            def generate_content(self, prompt: str):
                sdk.generate_calls.append(prompt)
                if not sdk.responses:
                    raise RuntimeError("no scripted response")
                item = sdk.responses.pop(0)
                if isinstance(item, Exception):
                    raise item
                return MagicMock(text=item)

        return _Model()


@pytest.fixture
def client_factory(tmp_path, monkeypatch):
    """Build a GeminiClient pointed at a temporary cache and a fake SDK."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def _make(responses: list, *, model_override: str | None = None) -> tuple[GeminiClient, _FakeSDK]:
        sdk = _FakeSDK(responses)
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        if model_override is not None:
            client.model = model_override
        return client, sdk

    return _make


class TestGeminiClient:
    def test_cache_hit_skips_sdk(self, client_factory) -> None:
        client, sdk = client_factory(responses=["first-response"])
        out1 = client.generate("hello", sleep_fn=lambda s: None)
        out2 = client.generate("hello", sleep_fn=lambda s: None)
        assert out1 == out2 == "first-response"
        assert len(sdk.generate_calls) == 1, "second call should be cache hit"
        assert client.cache.hits == 1
        assert client.cache.misses == 1

    def test_retries_then_succeeds(self, client_factory) -> None:
        client, sdk = client_factory(responses=[
            RuntimeError("429 transient"),
            "second-attempt",
        ])
        out = client.generate("p", sleep_fn=lambda s: None)
        assert out == "second-attempt"
        assert len(sdk.generate_calls) == 2

    def test_soft_fail_after_max_retries(self, client_factory) -> None:
        client, sdk = client_factory(responses=[
            RuntimeError("e1"), RuntimeError("e2"),
            RuntimeError("e3"), RuntimeError("e4"),
        ])
        out = client.generate("p", sleep_fn=lambda s: None)
        assert out == SOFT_FAIL_PLACEHOLDER
        assert len(sdk.generate_calls) == 4
        assert client.cache.get(client.model, "p") is None

    def test_env_var_overrides_model(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-override-model")
        sdk = _FakeSDK(responses=["ok"])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        client.generate("p", sleep_fn=lambda s: None)
        assert sdk.GenerativeModel_calls == ["gemini-override-model"]

    def test_api_key_never_logged(self, client_factory, caplog) -> None:
        client, _ = client_factory(responses=["ok"])
        with caplog.at_level("DEBUG"):
            client.generate("p", sleep_fn=lambda s: None)
        for rec in caplog.records:
            assert "test-key" not in rec.getMessage(), \
                "API key must never appear in logs"

    def test_missing_api_key_raises_only_when_calling_sdk(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        sdk = _FakeSDK(responses=["ok"])
        client = GeminiClient(sdk=sdk)
        client.cache = GeminiCache(tmp_path / "cache")
        client.cache.put(client.model, "cached-prompt", "cached")
        assert client.generate("cached-prompt", sleep_fn=lambda s: None) == "cached"
        out = client.generate("uncached", sleep_fn=lambda s: None)
        assert out == SOFT_FAIL_PLACEHOLDER

    def test_placeholder_api_key_treated_as_missing(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "PASTE_KEY_HERE")
        sdk = _FakeSDK(responses=["should-not-be-called"])
        client = GeminiClient(sdk=sdk)
        client.cache = GeminiCache(tmp_path / "cache")
        out = client.generate("p", sleep_fn=lambda s: None)
        assert out == SOFT_FAIL_PLACEHOLDER
        assert sdk.generate_calls == []


# ===========================================================================
# Group 4 — pattern_deriver.py
# ===========================================================================

class TestPatternDeriverTier1:
    @pytest.mark.parametrize("explanation,expected", [
        ("Clause is one-sided against the signing party.",
         "one_sided_against_signing_party"),
        ("Mutual balanced obligation between both parties.",
         "mutual_balanced"),
        ("Perpetual irrevocable license granted to counterparty.",
         "perpetual_irrevocable_grant_to_counterparty"),
        ("Uncapped liability exposure for the signing party.",
         "uncapped_financial_exposure"),
        ("Liability is capped at 12 months of fees.",
         "capped_financial_exposure"),
        ("Auto-renewal with 15 day notice — short notice burden.",
         "automatic_renewal_short_notice"),
        ("Unilateral right to renew sits with the counterparty.",
         "unilateral_renewal_right_counterparty"),
        ("MFN clause burdens the signing party.",
         "mfn_against_signing_party"),
        ("Non-compete is broad and long — 5 years across all geographies.",
         "non_compete_broad_and_long"),
        ("Audit rights are excessive and unrestricted.",
         "audit_rights_excessive_against_signing_party"),
        ("Governing law is in an unfavorable jurisdiction.",
         "governing_law_unfavorable_jurisdiction"),
        ("Force majeure is narrow and excludes pandemics.",
         "force_majeure_narrow"),
        ("Warranty is disclaimed — products provided as-is.",
         "warranty_disclaimer_broad"),
        ("Change of control triggers termination against the signing party.",
         "change_of_control_termination_against_signing_party"),
        ("Vague undefined terms create ambiguity.",
         "vague_undefined_terms"),
        ("Missing carve-outs for affiliate transfers.",
         "missing_carveouts"),
        ("Void ab initio consequence on breach.",
         "void_ab_initio_consequence"),
        ("Auto-assignment to counterparty affiliates is permitted.",
         "auto_assignment_to_counterparty_affiliates"),
        ("Insufficient text for a meaningful assessment.",
         "insufficient_text_for_assessment"),
        ("Truncated clause fragment.",
         "insufficient_text_for_assessment"),
    ])
    def test_tier1_known_patterns(self, explanation: str, expected: str) -> None:
        assert derive_tier1("Indemnification", explanation) == expected

    def test_tier1_returns_unknown_for_novel_text(self) -> None:
        assert derive_tier1(
            "Indemnification",
            "This clause references arcane patent doctrines from 1850.",
        ) == UNKNOWN

    def test_tier1_empty_returns_insufficient(self) -> None:
        assert derive_tier1("Indemnification", "") == "insufficient_text_for_assessment"
        assert derive_tier1("Indemnification", "   ") == "insufficient_text_for_assessment"


class TestPatternDeriverTier2:
    def _make_client(self, monkeypatch, tmp_path, response: str) -> GeminiClient:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        sdk = _FakeSDK(responses=[response])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        return client

    def test_tier2_returns_valid_code_when_response_is_clean(
        self, monkeypatch, tmp_path
    ) -> None:
        client = self._make_client(monkeypatch, tmp_path, "mutual_balanced")
        out = derive_tier2(
            clause_id="c1", clause_type="Indemnification",
            risk_explanation="Some explanation.",
            clause_text_excerpt="Excerpt text.",
            client=client,
        )
        assert out == "mutual_balanced"

    def test_tier2_extracts_code_from_chatty_response(
        self, monkeypatch, tmp_path
    ) -> None:
        client = self._make_client(
            monkeypatch, tmp_path,
            "Sure! The best match is `mfn_against_signing_party` because...",
        )
        out = derive_tier2(
            clause_id="c2", clause_type="Most Favored Nation",
            risk_explanation="MFN flagged.",
            clause_text_excerpt="MFN obligation",
            client=client,
        )
        assert out == "mfn_against_signing_party"

    def test_tier2_softfails_to_unknown_on_invalid_response(
        self, monkeypatch, tmp_path
    ) -> None:
        client = self._make_client(monkeypatch, tmp_path, "this is not a valid code")
        out = derive_tier2(
            clause_id="c3", clause_type="Indemnification",
            risk_explanation="x", clause_text_excerpt="y", client=client,
        )
        assert out == UNKNOWN

    def test_tier2_excerpt_truncated_to_200_chars(
        self, monkeypatch, tmp_path
    ) -> None:
        client = self._make_client(monkeypatch, tmp_path, "unknown_pattern")
        long_excerpt = "x" * 500
        derive_tier2(
            clause_id="c4", clause_type="Indemnification",
            risk_explanation="explanation",
            clause_text_excerpt=long_excerpt, client=client,
        )
        sent_prompt = client._sdk.generate_calls[0]
        assert "x" * 200 in sent_prompt
        assert "x" * 201 not in sent_prompt


class TestPatternDeriverTopLevel:
    def test_tier1_hit_short_circuits_no_llm_call(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        sdk = _FakeSDK(responses=[])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        out = derive_pattern(
            clause_id="c1", clause_type="Indemnification",
            risk_explanation="One-sided against the signing party.",
            config={"pattern_deriver": {"use_llm_tier2": True}},
            client=client,
        )
        assert out == "one_sided_against_signing_party"
        assert sdk.generate_calls == []

    def test_tier1_unknown_with_tier2_disabled_returns_unknown(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        sdk = _FakeSDK(responses=["mutual_balanced"])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        out = derive_pattern(
            clause_id="c2", clause_type="Indemnification",
            risk_explanation="Arcane novel construct.",
            config={"pattern_deriver": {"use_llm_tier2": False}},
            client=client,
        )
        assert out == UNKNOWN
        assert sdk.generate_calls == []

    def test_tier1_unknown_with_tier2_enabled_calls_llm_and_logs(
        self, monkeypatch, tmp_path, caplog
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        sdk = _FakeSDK(responses=["mutual_balanced"])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        with caplog.at_level("INFO", logger="src.stage4_report_gen.pattern_deriver"):
            out = derive_pattern(
                clause_id="cl-77", clause_type="Indemnification",
                risk_explanation="Arcane novel construct.",
                config={"pattern_deriver": {"use_llm_tier2": True}},
                client=client,
            )
        assert out == "mutual_balanced"
        assert sdk.generate_calls, "tier-2 should call the LLM"
        log_text = " ".join(rec.getMessage() for rec in caplog.records)
        assert "cl-77" in log_text
        assert "mutual_balanced" in log_text

    def test_env_var_disables_tier2_even_if_config_says_true(
        self, monkeypatch, tmp_path
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        monkeypatch.setenv("STAGE4_PATTERN_DERIVE_USE_LLM", "false")
        sdk = _FakeSDK(responses=["mutual_balanced"])
        client = GeminiClient(sdk=sdk, api_key="test-key")
        client.cache = GeminiCache(tmp_path / "cache")
        out = derive_pattern(
            clause_id="c3", clause_type="Indemnification",
            risk_explanation="Arcane novel construct.",
            config={"pattern_deriver": {"use_llm_tier2": True}},
            client=client,
        )
        assert out == UNKNOWN
        assert sdk.generate_calls == []

    def test_pattern_codes_count_meets_minimum(self) -> None:
        assert len(PATTERN_CODES) >= 38
        assert UNKNOWN in PATTERN_CODES
        assert "insufficient_text_for_assessment" in PATTERN_CODES


# ===========================================================================
# Group 5 — aggregator.py
# ===========================================================================

class TestNormalizeClauseType:
    @pytest.mark.parametrize("raw,canonical", [
        ("Indemnification", "Indemnification"),
        ("Indemnity", "Indemnification"),
        ("indemnification", "Indemnification"),
        ("  INDEMNIFICATION  ", "Indemnification"),
        ("MFN", "Most Favored Nation"),
        ("Most Favored Nation", "Most Favored Nation"),
        ("Limitation of Liability", "Cap On Liability"),
        ("Anti Assignment", "Anti-Assignment"),
        ("Auto-Renewal", "Renewal Term"),
        ("Perpetual License", "Irrevocable Or Perpetual License"),
        ("Choice of Law", "Governing Law"),
    ])
    def test_known_aliases(self, raw: str, canonical: str) -> None:
        assert normalize_clause_type(raw) == canonical

    def test_unknown_passes_through(self) -> None:
        assert normalize_clause_type("Some Novel Type") == "Some Novel Type"

    def test_empty_returns_empty(self) -> None:
        assert normalize_clause_type("") == ""


class TestInferContractType:
    def test_license_signature_wins(self) -> None:
        present = {"License Grant", "Irrevocable Or Perpetual License",
                   "Indemnification"}
        assert infer_contract_type(present) == "license"

    def test_distribution_signature_wins(self) -> None:
        present = {"Exclusivity", "Minimum Commitment", "Volume Restriction"}
        assert infer_contract_type(present) == "distribution"

    def test_no_signal_returns_universal(self) -> None:
        assert infer_contract_type({"Indemnification", "Cap On Liability"}) == "universal"


class TestFindMissingProtections:
    def test_universal_always_applied(self) -> None:
        missing = find_missing_protections(set(), "universal")
        clause_types = {m.clause_type for m in missing}
        assert "Indemnification" in clause_types
        assert "Cap On Liability" in clause_types
        assert "Governing Law" in clause_types

    def test_present_clauses_excluded(self) -> None:
        present = {"Indemnification", "Cap On Liability"}
        missing = find_missing_protections(present, "universal")
        clause_types = {m.clause_type for m in missing}
        assert "Indemnification" not in clause_types
        assert "Cap On Liability" not in clause_types

    def test_contract_type_specific_added(self) -> None:
        missing = find_missing_protections(set(), "license")
        clause_types = {m.clause_type for m in missing}
        assert "License Grant" in clause_types
        assert "IP Ownership Assignment" in clause_types

    def test_no_duplicate_when_universal_and_type_overlap(self) -> None:
        missing = find_missing_protections(set(), "vendor")
        clause_types = [m.clause_type for m in missing]
        assert len(clause_types) == len(set(clause_types))


class TestComputeScoreBreakdown:
    def test_zero_clauses_returns_note(self) -> None:
        sb = compute_score_breakdown(0, 0, 0, missing=[])
        assert sb.final_score == 0.0
        assert sb.note == "no clauses assessed"

    def test_zero_clauses_with_missing_still_zero(self) -> None:
        missing = [MissingProtection("Indemnification", "critical", "x")]
        sb = compute_score_breakdown(0, 0, 0, missing=missing)
        assert sb.final_score == 0.0
        assert sb.note == "no clauses assessed"
        assert sb.missing_critical_or_important == 1

    def test_all_high_max_score(self) -> None:
        sb = compute_score_breakdown(10, 0, 0, missing=[])
        assert sb.base_score == pytest.approx(10.0)
        assert sb.final_score == pytest.approx(10.0)
        assert sb.note == ""

    def test_all_low_minimum_meaningful_score(self) -> None:
        sb = compute_score_breakdown(0, 0, 10, missing=[])
        assert sb.base_score == pytest.approx(10.0 / 3.0, abs=0.01)

    def test_score_capped_at_10(self) -> None:
        critical_missing = [
            MissingProtection(f"x{i}", "critical", "r") for i in range(20)
        ]
        sb = compute_score_breakdown(10, 0, 0, missing=critical_missing)
        assert sb.final_score == 10.0

    def test_only_critical_or_important_boost(self) -> None:
        missing = [
            MissingProtection("a", "critical", "r"),
            MissingProtection("b", "important", "r"),
            MissingProtection("c", "standard", "r"),
        ]
        sb = compute_score_breakdown(1, 1, 1, missing=missing)
        assert sb.missing_critical_or_important == 2
        assert sb.missing_boost == 1.0

    def test_mixed_distribution(self) -> None:
        sb = compute_score_breakdown(2, 3, 5, missing=[])
        expected_base = (2 * 3 + 3 * 2 + 5 * 1) / 10.0 * (10.0 / 3.0)
        assert sb.base_score == pytest.approx(expected_base, abs=0.01)


def _stage3_clause(
    *, clause_id: str, document_id: str = "contract_test_001",
    clause_text: str, clause_type: str, risk_level: str,
    risk_explanation: str = "Some explanation.",
    confidence: float = 0.85, overridden: bool = False,
    similar_clauses: list | None = None,
) -> dict:
    return {
        "clause_id": clause_id,
        "document_id": document_id,
        "clause_text": clause_text,
        "clause_type": clause_type,
        "risk_level": risk_level,
        "risk_explanation": risk_explanation,
        "confidence": confidence,
        "overridden": overridden,
        "similar_clauses": similar_clauses or [],
        "agent_trace": [],
        "cross_references": [],
    }


class TestAggregate:
    def test_metadata_clauses_routed_to_header(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="Parties",
                clause_text="Acme Inc. and BetaCo LLC",
                risk_level="LOW",
            ),
            _stage3_clause(
                clause_id="c2", clause_type="Indemnification",
                clause_text="Indemnify against all claims.",
                risk_level="HIGH",
            ),
        ]
        result = aggregate(clauses, metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert len(result["reports"]) == 1
        assert result["reports"][0].clause_type == "Indemnification"
        assert result["metadata_block"]["Parties"] == "Acme Inc. and BetaCo LLC"

    def test_alias_normalization_in_pipeline(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="MFN",
                clause_text="Most favored nation language.",
                risk_level="MEDIUM",
            ),
        ]
        result = aggregate(clauses, metadata={"contract_type": "vendor"},
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["reports"][0].clause_type == "Most Favored Nation"
        assert result["reports"][0].clause_type_original == "MFN"

    def test_zero_clauses_yields_note(self) -> None:
        result = aggregate([], metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["score_breakdown"].note == "no clauses assessed"
        assert result["score_breakdown"].final_score == 0.0
        assert result["statistics"]["total"] == 0

    def test_metadata_provided_flag_true_when_metadata_passed(self) -> None:
        result = aggregate([], metadata={"Parties": "Acme / BetaCo"},
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["metadata_provided"] is True

    def test_metadata_provided_flag_false_when_no_metadata(self) -> None:
        result = aggregate([], metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["metadata_provided"] is False

    def test_contract_type_inferred_when_metadata_absent(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="License Grant",
                clause_text="grant", risk_level="HIGH",
            ),
            _stage3_clause(
                clause_id="c2", clause_type="Irrevocable Or Perpetual License",
                clause_text="perpetual", risk_level="HIGH",
            ),
        ]
        result = aggregate(clauses, metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["contract_type"] == "license"
        assert result["contract_type_inferred"] is True

    def test_contract_type_explicit_overrides_inference(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="License Grant",
                clause_text="grant", risk_level="HIGH",
            ),
        ]
        result = aggregate(clauses, metadata={"contract_type": "vendor"},
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["contract_type"] == "vendor"
        assert result["contract_type_inferred"] is False

    def test_missing_protections_populated(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="Indemnification",
                clause_text="Mutual indemnification.",
                risk_level="LOW",
            ),
        ]
        result = aggregate(clauses, metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        types = {m.clause_type for m in result["missing_protections"]}
        assert "Cap On Liability" in types
        assert "Indemnification" not in types

    def test_statistics_counts(self) -> None:
        clauses = [
            _stage3_clause(clause_id="h1", clause_type="Indemnification",
                          clause_text="t", risk_level="HIGH"),
            _stage3_clause(clause_id="h2", clause_type="Cap On Liability",
                          clause_text="t", risk_level="HIGH"),
            _stage3_clause(clause_id="m1", clause_type="Renewal Term",
                          clause_text="t", risk_level="MEDIUM"),
            _stage3_clause(clause_id="l1", clause_type="Governing Law",
                          clause_text="t", risk_level="LOW"),
        ]
        result = aggregate(clauses, metadata=None,
                          config={"pattern_deriver": {"use_llm_tier2": False}})
        assert result["statistics"] == {"high": 2, "medium": 1, "low": 1, "total": 4}


class TestBuildExecutiveDigest:
    def test_excerpt_capped_at_200_chars(self) -> None:
        long_text = "A" * 500
        clauses = [
            _stage3_clause(
                clause_id="c1", clause_type="Indemnification",
                clause_text=long_text, risk_level="HIGH",
            ),
        ]
        agg = aggregate(clauses, metadata=None,
                       config={"pattern_deriver": {"use_llm_tier2": False}})
        digest = build_executive_digest(agg)
        assert len(digest.top_high_risk) == 1
        assert len(digest.top_high_risk[0]["clause_text_excerpt"]) == 200

    def test_top_5_cap(self) -> None:
        clauses = [
            _stage3_clause(
                clause_id=f"h{i}", clause_type="Indemnification",
                clause_text=f"clause {i}", risk_level="HIGH",
                confidence=0.9 - i * 0.01,
            )
            for i in range(8)
        ]
        agg = aggregate(clauses, metadata=None,
                       config={"pattern_deriver": {"use_llm_tier2": False}})
        digest = build_executive_digest(agg, top_n=5)
        assert len(digest.top_high_risk) == 5

    def test_metadata_provided_flag_propagated(self) -> None:
        agg = aggregate([], metadata={"Parties": "X / Y"},
                       config={"pattern_deriver": {"use_llm_tier2": False}})
        digest = build_executive_digest(agg)
        assert digest.metadata_provided is True
        agg2 = aggregate([], metadata=None,
                        config={"pattern_deriver": {"use_llm_tier2": False}})
        digest2 = build_executive_digest(agg2)
        assert digest2.metadata_provided is False

    def test_missing_protections_are_clause_type_strings(self) -> None:
        agg = aggregate([], metadata=None,
                       config={"pattern_deriver": {"use_llm_tier2": False}})
        digest = build_executive_digest(agg)
        assert all(isinstance(m, str) for m in digest.missing_protections)
        assert "Indemnification" in digest.missing_protections


class TestMetadataConstants:
    def test_metadata_clause_types_complete(self) -> None:
        assert METADATA_CLAUSE_TYPES == frozenset({
            "Document Name", "Parties", "Agreement Date",
            "Effective Date", "Expiration Date",
        })


# ===========================================================================
# Group 6 — recommender.py
# ===========================================================================

def _bare_clause_report(clause_type: str, risk_pattern: str, risk_level: str) -> ClauseReport:
    return ClauseReport(
        clause_id="cid",
        document_id="doc",
        clause_text="text",
        clause_type=clause_type,
        clause_type_original=clause_type,
        risk_level=risk_level,
        risk_pattern=risk_pattern,
        risk_explanation="x",
        polished_explanation="",
        recommendation=Recommendation(text="", priority="LOW"),
        confidence=0.8,
    )


class TestRecommenderLookup:
    def test_exact_match_anti_assignment(self) -> None:
        rec = lookup(
            "Anti-Assignment",
            "assignment_restricted_only_for_signing_party",
            "HIGH",
        )
        assert rec.match_level == "exact"
        assert "reciproc" in rec.text.lower() or "consent" in rec.text.lower()
        assert rec.priority == "HIGH"

    def test_exact_match_change_of_control(self) -> None:
        rec = lookup(
            "Change of Control",
            "change_of_control_termination_against_signing_party",
            "HIGH",
        )
        assert rec.match_level == "exact"
        assert rec.priority == "HIGH"

    def test_exact_match_governing_law(self) -> None:
        rec = lookup(
            "Governing Law",
            "governing_law_unfavorable_jurisdiction",
            "MEDIUM",
        )
        assert rec.match_level == "exact"
        assert rec.priority == "MEDIUM"

    def test_type_fallback_when_pattern_unmatched(self) -> None:
        rec = lookup("Indemnification", "some_uncatalogued_pattern", "HIGH")
        assert rec.match_level == "type"
        assert rec.priority == "HIGH"

    def test_risk_level_generic_fallback_high(self) -> None:
        rec = lookup("Some Novel Clause Type", "unknown_pattern", "HIGH")
        assert rec.match_level == "risk_level"
        assert rec.priority == "HIGH"

    def test_risk_level_generic_fallback_medium(self) -> None:
        rec = lookup("Some Novel Clause Type", "unknown_pattern", "MEDIUM")
        assert rec.match_level == "risk_level"
        assert rec.priority == "MEDIUM"

    def test_universal_fallback_when_risk_level_unknown(self) -> None:
        rec = lookup("Some Novel Clause Type", "unknown_pattern", "WEIRD")
        assert rec.match_level == "universal"


class TestAttachRecommendations:
    def test_high_and_medium_get_filled(self) -> None:
        reports = [
            _bare_clause_report(
                "Anti-Assignment",
                "assignment_restricted_only_for_signing_party",
                "HIGH",
            ),
            _bare_clause_report("Renewal Term", "automatic_renewal_short_notice", "MEDIUM"),
        ]
        out = attach_recommendations(reports)
        assert all(r.recommendation.text for r in out)
        assert out[0].recommendation.match_level == "exact"
        assert out[1].recommendation.match_level == "exact"

    def test_low_skipped(self) -> None:
        report = _bare_clause_report("Anti-Assignment", "x", "LOW")
        original_text = report.recommendation.text
        out = attach_recommendations([report])
        assert out[0].recommendation.text == original_text  # untouched

    def test_unknown_pattern_falls_through_to_generic_high(self) -> None:
        report = _bare_clause_report("Novel Clause Type", "novel_pattern", "HIGH")
        out = attach_recommendations([report])
        assert out[0].recommendation.match_level == "risk_level"
        assert out[0].recommendation.priority == "HIGH"
