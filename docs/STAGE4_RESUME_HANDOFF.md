# Stage 4 — Session Resume Handoff

> **Purpose**: Stage 4 is being built incrementally; this file is the single source of truth for resuming after a session pause. Read end-to-end before touching code. After reading, jump to **§ Resumption Point** at the bottom.

---

## 1. Project context

**Project**: Legal Contract Risk Analyzer
**Branch**: `stage3_4_changes` (project lead has confirmed this is correct; the original task prompt mistakenly said `stage3_4`)
**Repo root**: `/home/vsharma/code/aiml`
**User email**: vishal.sharma@mavenir.com
**Today's working date**: 2026-04-30 or later

The project is a 4-stage ML pipeline (Stage 1+2 extract clauses, Stage 3 risk-classify with Mistral agent, Stage 4 generate report). We are mid-implementation of **Stage 4 (Report Generation)**. Stages 1+2 and 3 are out of scope — do not modify them.

**Authoritative reference**: `ARCHITECTURE.md` at the repo root. If this handoff and `ARCHITECTURE.md` disagree on anything Stage 4–related, stop and ask the user.

---

## 2. Stage 4 design (locked)

A hybrid Python + LLM report generator with these sub-tasks:

1. **Aggregator** (`aggregator.py`) — deterministic Python: alias-normalize clause types → derive `risk_pattern` → bucket by risk → compute `ScoreBreakdown` → run missing-protection check → infer contract_type when metadata absent. Returns `ClauseReport[]` + `MissingProtection[]` + `ScoreBreakdown`.
2. **Explainer** (`explainer.py`) — the **only** module that imports `llm_client`. `polish_clause_explanation(clause_report)` and `generate_executive_summary(digest: ExecutiveSummaryDigest)`. Both soft-fail to a placeholder string.
3. **Recommender** (`recommender.py`) — lookup against `recommendations_data.yaml` keyed by `(canonical_clause_type, risk_pattern)` with HIGH/MEDIUM generic fallbacks. No LLM.
4. **Report builder** (`report_builder.py`) — orchestrator: validate Stage 3 input → aggregator → recommender → explainer → renderers → `ReportArtifacts`.
5. **Evaluation** (`evaluate.py`) — ROUGE on summaries + structural completeness on `ContractReport`.

Output formats: JSON (canonical), Markdown, PDF (`reportlab`), DOCX (`python-docx`). All four every run.

LLM provider: **Google Gemini 2.5 Flash** (free tier). Default `gemini-2.5-flash`, overridable via `GEMINI_MODEL` env var.

### Hard rules (non-negotiable)

1. **Never** send full contract text or all clauses in a single LLM call. Inputs are clause-level structured objects only.
2. The LLM is used only by `explainer.py` and `pattern_deriver.py` tier-2 fallback. No LLM calls hidden in aggregator/recommender/report_builder.
3. All LLM calls go through `src/stage4_report_gen/llm_client.py`.
4. API key from environment only. `.env` at project root. Never hardcoded, never logged, never committed.
5. Caching mandatory. SHA-256-keyed disk cache under `.cache/gemini/`.
6. Rate-limit handling mandatory. Token bucket at 12 RPM, exponential backoff on 429, max 3 retries, soft-fail to placeholder string.
7. Failure isolation — failed LLM call must not abort the report.
8. `logging` module, not `print()`.
9. Use shared dataclasses in `src/common/schema.py`. Extend, don't redefine.

### Design decisions confirmed by the lead (durable record: `docs/STAGE4_DECISION_LOG.md`)

1. **Zero-clause case**: `final_score = 0.0` with `note = "no clauses assessed"` in `ScoreBreakdown`, surfaced in the rendered PDF/DOCX score section so a reader doesn't mistake an empty assessment for "low risk".
2. **Pattern deriver tier-2**: gated behind config flag `pattern_deriver.use_llm_tier2` (env override `STAGE4_PATTERN_DERIVE_USE_LLM`), default `true`. Fires whenever tier-1 returns `unknown_pattern`. Cached by content hash. Every tier-2 invocation logged at INFO with `clause_id` + resulting pattern code.
3. **Renderer file naming**: `output/<document_id>/<document_id>.{json,md,pdf,docx}`. Subdir per document, no timestamp suffix, overwrite on re-run.
4. **Fixture realism**: synthesized CUAD-style fixture, ≥8 clauses spanning Indemnification, Anti-Assignment, Change of Control, Liquidated Damages, Renewal Term, Cap On Liability, Governing Law, Irrevocable/Perpetual License variant. Mix of HIGH/MEDIUM/LOW. At least one `overridden=true`, at least one `confidence < 0.6`, at least one `cross_references` entry.
5. **Gemini model**: `gemini-2.5-flash` default in `configs/stage4_config.yaml`, overridable via `GEMINI_MODEL` env var.

### Risk score formula (already implemented in `aggregator.compute_score_breakdown`)

```
base_score    = (HIGH × 3 + MEDIUM × 2 + LOW × 1) / total_clauses × (10/3)
missing_boost = 0.5 × number_of_missing_critical_or_important_protections
final_score   = min(base_score + missing_boost, 10.0)
```

Edge case: `total_clauses == 0` → `final_score = 0.0` with `note = "no clauses assessed"`.

---

## 3. Target deliverable structure

```
AIML_project/
├── ARCHITECTURE.md                              # Gemini Flash, decision-log pointer ✅
├── .env                                         # gitignored, with placeholder ✅
├── .env.example                                 # ✅
├── .gitignore                                   # includes .env, .cache/, output/ ✅
├── requirements.txt                             # Gemini, reportlab, python-docx ✅
├── configs/
│   └── stage4_config.yaml                       # gemini-2.5-flash, pattern_deriver flag ✅
├── docs/
│   ├── STAGE4_DECISION_LOG.md                   # ✅
│   └── STAGE4_RESUME_HANDOFF.md                 # this file (updated each pause)
├── prompts/
│   └── stage4_resume_prompt.md                  # self-contained resume prompt ✅
├── src/
│   ├── common/
│   │   ├── schema.py                            # Stage 4 dataclasses present ✅
│   │   └── utils.py                             # config loader, logging, JSON I/O ✅
│   └── stage4_report_gen/
│       ├── __init__.py                          # empty ✅
│       ├── aggregator.py                        # ✅ Group 5
│       ├── explainer.py                         # ❌ Group 8 (TODO)
│       ├── recommender.py                       # ✅ Group 6
│       ├── report_builder.py                    # ❌ Group 9 (TODO)
│       ├── evaluate.py                          # ❌ Group 10 (TODO)
│       ├── llm_client.py                        # ✅ Group 2
│       ├── rate_limiter.py                      # ✅ Group 2
│       ├── cache.py                             # ✅ Group 2
│       ├── pattern_deriver.py                   # ✅ Group 4
│       ├── nodes.py                             # ⚠️ stub (replace in Group 11)
│       ├── recommendations_data.yaml            # ✅ Group 3 (16 specific + 3 type + 3 generic)
│       ├── missing_protections.yaml             # ✅ Group 3 (universal + 6 contract types)
│       ├── clause_type_aliases.yaml             # ✅ Group 3 (133 entries)
│       └── renderers/                           # ❌ Group 7 (TODO — full directory)
│           ├── __init__.py
│           ├── json_renderer.py
│           ├── markdown_renderer.py
│           ├── pdf_renderer.py
│           └── docx_renderer.py
├── tests/
│   └── test_stage4.py                           # ✅ 98 tests, all passing
├── data/
│   └── stage4_fixtures/                         # ❌ Group 9 (TODO)
│       ├── sample_stage3_output.json
│       └── sample_metadata.json
└── output/                                      # gitignored ✅
```

---

## 4. Out of scope — DO NOT MODIFY

- `src/stage1_extract_classify/` — Stage 1+2 code
- `src/stage3_risk_agent/` — Stage 3 code
- `src/workflow/graph.py` — LangGraph wiring (we touch only `src/stage4_report_gen/nodes.py`)
- The CUAD dataset
- The training pipeline
- `data/processed/` and `data/review/` — read-only reference

---

## 5. Completed work — Groups 1 through 6

### Group 1 — Cleanup, decision log, ARCHITECTURE.md ✅

- **Deleted** (Mistral-era + stale): `src/stage4_report/` (whole stale dir), `src/stage4_report_gen/{generator,aggregator,report_builder,evaluate}.py`, stale `__pycache__` files, `app/routers/stage4_report.py`, `app/services/stage4_report_svc.py`. The Group-5 aggregator/report_builder are NEW files at the same paths — the Mistral-era ones were deleted before re-creation.
- **Already-staged deletions** (from prior session): `src/common/llm_client.py`, `src/stage4_report_gen/{docx_renderer,pdf_converter,prompts}.py`, `tests/test_stage4.py`. Renderers will be re-created in `renderers/` subdir in Group 7. `tests/test_stage4.py` has been re-created (Group 2 onward).
- **Edited**: `src/common/schema.py` (legacy `RiskReport`/`ReportClause`/`ReportMetadata` removed; new Stage 4 dataclasses kept); `app/main.py` (dangling commented imports cleaned).
- **Created**: `docs/STAGE4_DECISION_LOG.md`, `docs/STAGE4_RESUME_HANDOFF.md` (this file), `prompts/stage4_resume_prompt.md`.
- **Config bumped**: `configs/stage4_config.yaml` — `gemini.model: gemini-2.5-flash` and new `pattern_deriver.use_llm_tier2: true` block. Both env-var overrides documented.
- **ARCHITECTURE.md updated**: 5 FLAN-T5 → Gemini Flash edits + decision-log pointer + Stage 4 directory tree expanded + Key Models table row + sample-report `models_used` + `stage4_config.yaml` example block rewritten.

### Group 2 — Foundations (cache + rate_limiter + llm_client) ✅

- **`src/stage4_report_gen/cache.py`** — `GeminiCache` class. SHA-256 over `(model, prompt)`, JSON-on-disk, hit/miss tracking, corrupted-entry recovery (delete + log warning), `hit_rate` property, `reset_stats`.
- **`src/stage4_report_gen/rate_limiter.py`** — `TokenBucketRateLimiter` (thread-safe, capacity = RPM, refill = RPM/60 per second, `acquire()` blocks until token available, returns time waited) + `backoff_delays(delays_sec, jitter=0.25)` generator.
- **`src/stage4_report_gen/llm_client.py`** — `GeminiClient` class. Loads `.env` via python-dotenv (idempotent), reads config, honors `GEMINI_MODEL` env var override, lazy-configures the SDK on first call. `generate(prompt, sleep_fn=None)`: cache check → rate-limit acquire → SDK call → on exception, exponential backoff up to `max_retries` → terminal: log warning + return `SOFT_FAIL_PLACEHOLDER`. Treats `PASTE_KEY_HERE` as missing key. Never logs the API key. Lazy singleton `get_default_client()` for production code; tests inject `sdk=` kwarg directly.
- **Tests**: 16 Stage 4 tests covering cache (4), rate-limiter (5), client (7).

### Group 3 — YAML data files ✅

- **`src/stage4_report_gen/clause_type_aliases.yaml`** — 133 entries: 41 canonical identity entries (covering all CUAD types including the 5 metadata) + ~92 free-form aliases (Indemnity → Indemnification, MFN → Most Favored Nation, Limitation of Liability → Cap On Liability, Auto-Renewal → Renewal Term, etc.). Lookups are case-insensitive (the aggregator handles that).
- **`src/stage4_report_gen/missing_protections.yaml`** — 7 contract-type checklists: `universal` (Indemnification, Cap On Liability, Governing Law, Termination For Convenience, Force Majeure) plus `vendor`, `license`, `employment`, `distribution`, `services`, `nda`. Each item carries `clause_type`, `importance` (critical / important / standard), and `rationale`.
- **`src/stage4_report_gen/recommendations_data.yaml`** — 16 specific `(clause_type, risk_pattern)` entries (Anti-Assignment×2, Change of Control, Liquidated Damages, Cap On Liability, MFN, Non-Compete, Non-Disparagement, Renewal Term×2, Irrevocable License, Audit Rights, Governing Law, Termination For Convenience, Indemnification, Volume Restriction). Plus 3 type-fallbacks (Indemnification, Cap On Liability, Non-Compete). Plus 3 generic fallbacks (HIGH, MEDIUM, UNIVERSAL). Each entry has `recommendation`, `market_standard`, `fallback_position`, `priority`.

### Group 4 — Pattern deriver ✅

- **`src/stage4_report_gen/pattern_deriver.py`** — Tier-1 = ordered list of ~50 regex rules over the controlled vocabulary (`PATTERN_CODES` tuple, 39 codes). `derive_tier1(clause_type, risk_explanation)` returns the first matching code or `unknown_pattern` (or `insufficient_text_for_assessment` for empty text). Tier-2 = `derive_tier2(...)` builds a constrained-output prompt with the codes list, sends through the cached `GeminiClient`, parses the response (first whitespace-delimited token, then a contains-check fallback). `derive_pattern(...)` is the top-level entry point; it short-circuits on tier-1 hit, then checks `STAGE4_PATTERN_DERIVE_USE_LLM` env override, then `pattern_deriver.use_llm_tier2` config flag, then invokes tier-2. Logs every tier-2 invocation at INFO with `clause_id` + resulting pattern code (per decision #2). Tier-2 prompt input is clause-level only (clause_type + risk_explanation + ≤200-char excerpt).
- **Tests**: 41 added (20 parameterized tier-1 hits, tier-1 unknown/empty, tier-2 clean/chatty/invalid/excerpt-truncation, top-level short-circuit/disabled/enabled-with-logging/env-override, code-count assertion).

### Group 5 — Aggregator ✅

- **`src/stage4_report_gen/aggregator.py`** — `aggregate(raw_clauses, metadata, *, config, client)` is the top-level entry. Pipeline: route 5 metadata clause types to header → for non-metadata, call `to_clause_report(...)` (alias-normalize via `normalize_clause_type` + derive risk_pattern via `pattern_deriver.derive_pattern` + extract `clause_text_excerpt`, `confidence`, `overridden`, capped `similar_clauses`). Then: build `metadata_block` (canonical 5-key order, "—" placeholders), determine `contract_type` (explicit-from-metadata > inferred via `infer_contract_type` over clause-type signatures > fallback "universal"), `find_missing_protections(present_canonical, contract_type)` (universal + contract-type checklists, deduped), `compute_score_breakdown(...)` (with the zero-clause edge case from decision #1), and statistics dict. Returns a single dict with all the above (plus `contract_type_inferred` and `metadata_provided` flags).
- Also `build_executive_digest(aggregation, top_n=5, excerpt_max=200)` — produces the `ExecutiveSummaryDigest` dataclass, with hard cap on excerpt length and HIGH-risk clauses sorted by confidence descending. Hard rule #1 boundary (this is what the LLM sees, not raw text).
- All YAML files cached on module import.
- **Tests**: 41 added (alias normalization parametric, contract-type inference, missing-protection logic, score breakdown — including zero-clause and cap edge cases — full `aggregate()` pipeline through metadata routing / alias / inference / explicit-override / missing population / statistics, executive digest excerpt-cap / top-N cap / metadata flag propagation / missing-protection-strings-only).

### Group 6 — Recommender ✅

- **`src/stage4_report_gen/recommender.py`** — `lookup(clause_type, risk_pattern, risk_level)` walks four tiers: exact `(clause_type, risk_pattern)` → type fallback by clause_type → generic by risk_level (HIGH/MEDIUM) → universal. Returns a `Recommendation` with `match_level` set so coverage can be tracked. `attach_recommendations(reports)` mutates HIGH/MEDIUM `ClauseReport.recommendation`, leaves LOW untouched, logs coverage counts.
- **Tests**: 10 added (exact match for 3 specific patterns, type fallback, risk-level generic for HIGH and MEDIUM, universal fallback for unknown risk level, attach_recommendations populates HIGH/MEDIUM and skips LOW, unknown-pattern falls through to generic-HIGH).

### Test suite status

```
98 Stage 4 tests, all passing.
Full suite: 131 passed, 10 failed (10 failures are pre-existing
NotImplementedError stubs in tests/test_preprocessing.py and
tests/test_stage1.py — unrelated to Stage 4, must remain unchanged).
```

---

## 6. Remaining work — Groups 7 through 12

### Group 7 — Renderers (TODO — next to do)

Create the entire `src/stage4_report_gen/renderers/` directory:

- **`renderers/__init__.py`** — re-export `render_json`, `render_markdown`, `render_pdf`, `render_docx`.
- **`renderers/json_renderer.py`** — `render_json(report: ContractReport, dest: Path) -> Path`. Use `report.to_dict()` (dataclass `asdict`), serialize with indent=2, return path.
- **`renderers/markdown_renderer.py`** — `render_markdown(report: ContractReport, dest: Path) -> Path`. Sections: title, metadata block, executive summary paragraph, score section (must surface `ScoreBreakdown.note` from decision #1 — when note is non-empty, show "Note: {note}" prominently), three risk tables (HIGH, MEDIUM, LOW) with columns `clause_type | risk_pattern | confidence | reasoning | recommendation`, missing-protections list with importance levels, statistics + score breakdown summary, fixed disclaimer.
- **`renderers/pdf_renderer.py`** — `render_pdf(report: ContractReport, dest: Path, *, config: dict) -> Path`. Use `reportlab.platypus` for paragraph + table flow. Color severities from `config.renderers.pdf.{high,medium,low}_color`. Same content as markdown, including ScoreBreakdown.note. Generate to `dest`.
- **`renderers/docx_renderer.py`** — `render_docx(report: ContractReport, dest: Path) -> Path`. Use `python-docx` heading styles, table for each risk tier. Same content + ScoreBreakdown.note.

Test bundle (~6–10 tests): each renderer produces a non-empty file at the right path, JSON parses as valid JSON matching the dataclass shape, markdown contains expected section headers, PDF/DOCX magic-bytes/file-format check (avoid full content parsing — those libs are well-tested, just sanity-verify).

### Group 8 — Explainer

- **`src/stage4_report_gen/explainer.py`** — only module that imports `llm_client`.
  - `polish_clause_explanation(clause_report: ClauseReport, *, client: GeminiClient | None = None) -> str` — one targeted Gemini call per HIGH/MEDIUM clause (LOW skipped — return `risk_explanation` verbatim). Prompt input: clause_type, risk_level, risk_pattern, risk_explanation, ≤200-char clause_text excerpt. **Never** sends full contract text. Soft-fails to placeholder.
  - `generate_executive_summary(digest: ExecutiveSummaryDigest, *, client: GeminiClient | None = None) -> str` — single call from the structured digest. **MUST** validate digest excerpt cap (≤200 chars) before sending; raise `ValueError` if any excerpt exceeds. Honor `digest.metadata_provided` — when False, prompt MUST instruct the LLM not to invent party names.
- Tests: clause polish soft-fails on Gemini failure, LOW skipped, exec summary validates excerpt cap, exec summary instructs no-party-invention when `metadata_provided=False`, all calls go through `client.generate(...)`, no full contract text in any prompt.

### Group 9 — Orchestrator + fixtures

- **`data/stage4_fixtures/sample_stage3_output.json`** — synthesized per decision #4: ≥8 clauses spanning Indemnification, Anti-Assignment, Change of Control, Liquidated Damages, Renewal Term, Cap On Liability, Governing Law, Irrevocable/Perpetual License variant; mix of HIGH/MEDIUM/LOW; at least one `overridden=true`, at least one `confidence < 0.6`, at least one `cross_references` entry pointing to another fixture clause. Schema matches ARCHITECTURE.md "Stage 3 Output → Stage 4 Input".
- **`data/stage4_fixtures/sample_metadata.json`** — Parties, contract_type, effective_date, title.
- **`src/stage4_report_gen/report_builder.py`** — `generate_report(stage3_output: list[dict], metadata: dict | None = None, *, config_path: str = "configs/stage4_config.yaml", client: GeminiClient | None = None) -> ReportArtifacts`. Pipeline: jsonschema-validate input shape → `aggregate(...)` → `attach_recommendations(...)` → `polish_clause_explanation(...)` per HIGH/MEDIUM clause (skip LOW) → `build_executive_digest(...)` → `generate_executive_summary(...)` → assemble `ContractReport` dataclass → invoke all four renderers writing to `output/<document_id>/<document_id>.{json,md,pdf,docx}` (decision #3) → return `ReportArtifacts` (paths + duration_seconds + cache_hit_rate from `client.cache.hit_rate`). Document_id taken from the first clause (all clauses share one).
- Tests: end-to-end on the fixture with mocked Gemini, all four artifacts produced, JSON loads cleanly, schema-valid `ContractReport`, second-run cache hit rate = 1.0, soft-fail does not abort (one clause's polish call returning placeholder still produces full report).

### Group 10 — Evaluate

- **`src/stage4_report_gen/evaluate.py`** — `evaluate_summaries(generated, reference) -> dict` (rouge1/rouge2/rougeL F-measures via `rouge-score`; soft-fail to zeros on missing dependency); `evaluate_report_completeness(report: ContractReport) -> dict[str, bool]` (validates the new `ContractReport` schema: document_id, summary, statistics, three risk lists exist, score_breakdown present, score in [0,10], totals match, disclaimer non-empty for the rendered version… actually disclaimer lives in the renderer not the dataclass — adjust checks to dataclass fields only).

### Group 11 — LangGraph adapter

- **`src/stage4_report_gen/nodes.py`** — replace the stub. `node_report_generation(state: RiskAnalysisState) -> dict`: pull `risk_assessed_clauses` and any `metadata_block` from state, convert to the Stage 3 → Stage 4 input shape, call `report_builder.generate_report(...)`, return `{"final_report": <ContractReport.to_dict()>, "report_artifacts": {<paths>}}`.

### Group 12 — Final smoke

- Full `pytest tests/ -q`, expect Stage 4 tests all passing and the same 10 pre-existing failures unchanged.
- Show test count delta and any timing observations.

### After Group 12 — Steps 5 and 6

- **Step 5**: ask the lead for the Gemini API key. Point at https://aistudio.google.com/app/apikey, free tier. Tell them to paste into `.env` at `GEMINI_API_KEY=`. Don't ask them to share the key. Wait for confirmation.
- **Step 6**: live demo — run end-to-end on the fixture, show four artifact paths/sizes, head/tail of Markdown, second-run cache hit ~100%, timing breakdown, no errors. Write `STAGE4_DEMO_RESULTS.md`.

---

## 7. Test status — current

```
$ pytest tests/test_stage4.py -q
98 passed in 5.35s

$ pytest tests/ -q
131 passed, 10 failed
```

The 10 failures are all `NotImplementedError` stubs in `tests/test_preprocessing.py` (5) and `tests/test_stage1.py` (5). They are pre-existing, unrelated to Stage 4, and must remain unchanged. The Stage 3 tests (33) continue to pass.

---

## 8. Stage 4 dataclasses (already in `src/common/schema.py`)

- `ClauseObject`, `ExtractionResult` — Stage 1+2
- `SimilarClause`, `AgentTraceEntry`, `RiskAssessedClause` — Stage 3 → 4
- `SyntheticRiskLabel` — Stage 3 training data
- `Recommendation` — `text, market_standard, fallback_position, priority, match_level`
- `MissingProtection` — `clause_type, importance, rationale`
- `ClauseReport` — `clause_id, document_id, clause_text, clause_type, clause_type_original, risk_level, risk_pattern, risk_explanation, polished_explanation, recommendation, confidence, overridden, similar_clauses`
- `ScoreBreakdown` — `high_count, medium_count, low_count, base_score, missing_critical_or_important, missing_boost, final_score, note`
- `ExecutiveSummaryDigest` — `metadata, statistics, top_high_risk, missing_protections, metadata_provided`
- `ContractReport` — `document_id, summary, statistics, high_risk, medium_risk, low_risk, low_risk_summary, missing_protections, overall_risk_score, score_breakdown, total_clauses, metadata, generated_at, models_used`
- `ReportArtifacts` — `document_id, report, json_path, markdown_path, pdf_path, docx_path, duration_seconds, cache_hit_rate`

---

## 9. Behavioral guardrails for the resuming session

- Show full file contents — don't say "rest unchanged".
- Use markdown link syntax `[file.py:42](src/file.py#L42)` for code references in user-facing text.
- After each logical group: run relevant tests, show output, do not advance until tests pass.
- If a hidden Mistral reference, schema contradiction, or unexpected file shows up — stop and tell the user.
- Don't invent files outside the deliverable structure — ask first if you genuinely need one.
- Never log the API key. Never commit `.env`.
- Default to no comments in code; only when WHY is non-obvious.
- `logging` not `print()`.
- Type hints on all signatures.
- Python 3.10+ idioms.
- Only `explainer.py` and `pattern_deriver.py` may import `llm_client`.

---

## 10. Red flags — STOP and ask the lead

- `ARCHITECTURE.md` content disagrees with this handoff (other than the FLAN-T5 → Gemini change, which is done).
- Mistral-related code surfaced anywhere in Stage 4 paths.
- LLM call appearing outside `llm_client.py`, `explainer.py`, or `pattern_deriver.py` tier-2.
- A test that requires a live Gemini call.
- Stage 3 output not matching the schema in `ARCHITECTURE.md` "Stage 3 Output → Stage 4 Input".
- Hardcoded API key anywhere.
- Free-tier rate limit being exceeded during normal testing.

---

## 11. Resumption point — START HERE NEXT SESSION

When the new session starts:

1. Read this file end-to-end and `docs/STAGE4_DECISION_LOG.md`.
2. Run `git status` to confirm working tree state matches § 5.
3. Run `pytest tests/test_stage4.py -q` — expect `98 passed`.
4. Run `pytest tests/ -q` — expect `131 passed, 10 failed` (10 are the pre-existing unrelated stubs).
5. Confirm to the user: *"Resumed from `docs/STAGE4_RESUME_HANDOFF.md`. Verified Groups 1–6 done, 98 Stage 4 tests passing. Starting Group 7 (renderers)."*
6. **Resume at Group 7 — Renderers**:
   - Create `src/stage4_report_gen/renderers/` directory.
   - Build the four renderer modules in this order: `json_renderer.py` (simplest), `markdown_renderer.py` (becomes the reference content layout), `pdf_renderer.py`, `docx_renderer.py`. All driven by a `ContractReport` dataclass. The score section in markdown / PDF / DOCX MUST surface `ScoreBreakdown.note` (zero-clause case from decision #1).
   - Write 6–10 renderer tests in `tests/test_stage4.py`: each renderer writes a non-empty file at the right path, JSON parses cleanly and matches dataclass shape, markdown contains expected section headers and the ScoreBreakdown.note when non-empty, PDF/DOCX file-format sanity check.
   - Run tests, show output, confirm Group 7 done before moving to Group 8.
7. After Group 7 → Group 8 (explainer) → 9 (orchestrator + fixtures) → 10 (evaluate) → 11 (nodes.py adapter) → 12 (final smoke). Then Step 5 (API key request), Step 6 (live demo).

---

## 12. Original task spec (verbatim from project lead)

The original task prompt is preserved in `prompts/stage4_resume_prompt.md` (see also Section 12 of the previous handoff version, which embedded it in full). If anything in this handoff disagrees with the original prompt, the original prompt wins — except where a confirmed decision in `docs/STAGE4_DECISION_LOG.md` has explicitly overridden it (e.g., model bumped from `gemini-1.5-flash` to `gemini-2.5-flash`).

End of handoff.
