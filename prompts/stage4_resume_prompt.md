# Stage 4 Resume Prompt

> Paste this entire file as the first message of a fresh Claude Code session to resume the Stage 4 build at the point where the previous session paused. Self-contained — every piece of context the new session needs is in this file. For deeper inventory of completed work, additionally read `docs/STAGE4_RESUME_HANDOFF.md` and `docs/STAGE4_DECISION_LOG.md`.

---

## ROLE

You are a senior Python engineer continuing work on the **Legal Contract Risk Analyzer** project. The project lead is mid-implementation of **Stage 4 (Report Generation)** on branch `stage3_4_changes`. Six build groups (1–6) are already complete with 98 passing Stage 4 tests; you are picking up at **Group 7 (Renderers)**.

Repo root: `/home/vsharma/code/aiml`. Today's date: 2026-04-30 or later. Python 3.10+.

## FIRST ACTIONS (do these before writing any code)

1. Read `docs/STAGE4_RESUME_HANDOFF.md` end to end. It is the comprehensive snapshot of where the prior session left off. Sections § 5 (completed work) and § 6 (remaining work) are most important.
2. Read `docs/STAGE4_DECISION_LOG.md`. Append-only durable record of locked design decisions (5 entries from 2026-04-30).
3. Run `git status`. Confirm branch is `stage3_4_changes` and that the documented working-tree state in the handoff (§ 5) still matches.
4. Run `pytest tests/test_stage4.py -q` — expect **98 passed**.
5. Run `pytest tests/ -q` — expect **131 passed, 10 failed**. The 10 failures are pre-existing `NotImplementedError` stubs in `tests/test_preprocessing.py` (5) and `tests/test_stage1.py` (5). Unrelated to Stage 4. Must remain unchanged.

If anything in steps 1–5 doesn't match what the handoff describes, **stop and tell the user** before resuming work.

Then say: *"Resumed from `prompts/stage4_resume_prompt.md`. Verified Groups 1–6 done, 98 Stage 4 tests passing. Starting Group 7 (renderers)."*

## WHAT'S ALREADY BUILT (Groups 1–6 ✅)

**Group 1 — Cleanup, decision log, ARCHITECTURE.md**: deleted Mistral-era code, removed legacy `RiskReport`/`ReportClause`/`ReportMetadata` from `src/common/schema.py`, cleaned `app/main.py`. Created `docs/STAGE4_DECISION_LOG.md` and `docs/STAGE4_RESUME_HANDOFF.md`. Bumped `configs/stage4_config.yaml` to `gemini-2.5-flash` and added `pattern_deriver.use_llm_tier2: true` flag. Updated `ARCHITECTURE.md` with all FLAN-T5 → Gemini Flash edits, decision-log pointer, expanded directory tree, Key Models table row, sample-report `models_used`, and rewritten config example.

**Group 2 — Foundations**: `src/stage4_report_gen/cache.py` (SHA-256 disk cache, hit/miss tracking, corrupted-entry recovery), `src/stage4_report_gen/rate_limiter.py` (thread-safe token bucket at 12 RPM, exponential-backoff iterator with jitter), `src/stage4_report_gen/llm_client.py` (the only Gemini wrapper — loads `.env`, integrates cache + rate-limiter, retries on transient errors, soft-fails to `SOFT_FAIL_PLACEHOLDER` after `max_retries`, never logs API key, treats `PASTE_KEY_HERE` as unset, lazy singleton via `get_default_client()`).

**Group 3 — YAML data files**:
- `src/stage4_report_gen/clause_type_aliases.yaml` (133 entries)
- `src/stage4_report_gen/missing_protections.yaml` (universal + 6 contract-type checklists)
- `src/stage4_report_gen/recommendations_data.yaml` (16 specific entries + 3 type-fallbacks + 3 generic fallbacks)

**Group 4 — Pattern deriver**: `src/stage4_report_gen/pattern_deriver.py` with `PATTERN_CODES` (39 controlled-vocab codes), `derive_tier1` (regex/keyword), `derive_tier2` (LLM via cached `GeminiClient` with constrained-output prompt), `derive_pattern` (top-level entry: tier-1 → check `STAGE4_PATTERN_DERIVE_USE_LLM` env override → check `pattern_deriver.use_llm_tier2` config flag → tier-2; logs every tier-2 invocation at INFO with `clause_id` + result).

**Group 5 — Aggregator**: `src/stage4_report_gen/aggregator.py` with `aggregate(...)` (top-level pipeline: route metadata clauses to header → alias-normalize + derive_pattern per non-metadata clause → infer contract_type → find missing_protections → compute_score_breakdown with zero-clause edge case → return dict). Plus `build_executive_digest(...)` (Hard rule #1 boundary — only thing the LLM sees for the contract-level summary).

**Group 6 — Recommender**: `src/stage4_report_gen/recommender.py` with `lookup(clause_type, risk_pattern, risk_level)` doing four-tier match (exact → type → risk-level generic → universal) and `attach_recommendations(reports)` mutating HIGH/MEDIUM `ClauseReport.recommendation` (LOW skipped) and logging coverage counts.

**Tests**: 98 Stage 4 tests in `tests/test_stage4.py`, all passing.

## REMAINING WORK — Groups 7–12

### Group 7 — Renderers (DO THIS FIRST)

Create the entire `src/stage4_report_gen/renderers/` directory:

- `renderers/__init__.py` — re-export `render_json`, `render_markdown`, `render_pdf`, `render_docx`.
- `renderers/json_renderer.py` — `render_json(report: ContractReport, dest: Path) -> Path`. Use `report.to_dict()`, indent=2, return path.
- `renderers/markdown_renderer.py` — `render_markdown(report: ContractReport, dest: Path) -> Path`. Sections: title, metadata block, executive summary paragraph, score section (must surface `ScoreBreakdown.note` from decision #1 — when note is non-empty, show "Note: {note}" prominently), three risk tables (HIGH, MEDIUM, LOW) with columns `clause_type | risk_pattern | confidence | reasoning | recommendation`, missing-protections list with importance, statistics + score breakdown, fixed disclaimer.
- `renderers/pdf_renderer.py` — `render_pdf(report: ContractReport, dest: Path, *, config: dict) -> Path`. Use `reportlab.platypus` for paragraph + table flow. Color severities from `config.renderers.pdf.{high,medium,low}_color`. Same content as markdown, including ScoreBreakdown.note.
- `renderers/docx_renderer.py` — `render_docx(report: ContractReport, dest: Path) -> Path`. Use `python-docx` heading styles, table for each risk tier. Same content + ScoreBreakdown.note.

Add 6–10 renderer tests in `tests/test_stage4.py`:
- Each renderer writes a non-empty file at the right path
- JSON parses cleanly and matches dataclass shape
- Markdown contains expected section headers and the ScoreBreakdown.note when non-empty (build a `ContractReport` with empty clauses to trigger the note)
- PDF/DOCX file-format sanity check (PDF starts with `%PDF`, DOCX is a zip archive starting with `PK`)

Run `pytest tests/test_stage4.py -q`, show output, confirm Group 7 done before moving on.

### Group 8 — Explainer

`src/stage4_report_gen/explainer.py` — only module that imports `llm_client` (besides `pattern_deriver`).
- `polish_clause_explanation(clause_report: ClauseReport, *, client: GeminiClient | None = None) -> str` — one targeted Gemini call per HIGH/MEDIUM clause (LOW skipped — return `risk_explanation` verbatim). Prompt input: clause_type, risk_level, risk_pattern, risk_explanation, ≤200-char clause_text excerpt. **Never** sends full contract text. Soft-fails to placeholder.
- `generate_executive_summary(digest: ExecutiveSummaryDigest, *, client: GeminiClient | None = None) -> str` — single call from the structured digest. **MUST** validate digest excerpt cap (≤200 chars) before sending; raise `ValueError` if any excerpt exceeds. Honor `digest.metadata_provided` — when `False`, the prompt MUST instruct the LLM not to invent party names.

Tests: clause polish soft-fails on Gemini failure; LOW skipped; exec summary validates excerpt cap; exec summary instructs no-party-invention when `metadata_provided=False`; all calls go through `client.generate(...)`; no full contract text in any prompt.

### Group 9 — Orchestrator + fixtures

- `data/stage4_fixtures/sample_stage3_output.json` — synthesized per decision #4: ≥8 clauses spanning Indemnification, Anti-Assignment, Change of Control, Liquidated Damages, Renewal Term, Cap On Liability, Governing Law, Irrevocable/Perpetual License variant; mix HIGH/MEDIUM/LOW; ≥1 `overridden=true`, ≥1 `confidence < 0.6`, ≥1 `cross_references` entry pointing to another fixture clause.
- `data/stage4_fixtures/sample_metadata.json` — Parties, contract_type, effective_date, title.
- `src/stage4_report_gen/report_builder.py` — `generate_report(stage3_output, metadata=None, *, config_path="configs/stage4_config.yaml", client=None) -> ReportArtifacts`. Pipeline: jsonschema-validate input shape → `aggregate(...)` → `attach_recommendations(...)` → polish HIGH/MEDIUM clauses (skip LOW) → `build_executive_digest(...)` → `generate_executive_summary(...)` → assemble `ContractReport` → invoke all four renderers writing to `output/<document_id>/<document_id>.{json,md,pdf,docx}` (decision #3) → return `ReportArtifacts` with paths + duration + cache_hit_rate.

Tests: end-to-end on the fixture with mocked Gemini, all four artifacts produced, JSON loads cleanly, schema-valid `ContractReport`, second-run cache hit rate = 1.0, soft-fail does not abort (one clause's polish returning placeholder still produces full report).

### Group 10 — Evaluate

`src/stage4_report_gen/evaluate.py`: `evaluate_summaries(generated, reference) -> dict` (rouge1/rouge2/rougeL F-measures via `rouge-score`, soft-fail to zeros if missing); `evaluate_report_completeness(report: ContractReport) -> dict[str, bool]` (validates the new `ContractReport` schema fields).

### Group 11 — LangGraph adapter

Replace stub `src/stage4_report_gen/nodes.py` with:
```
def node_report_generation(state: RiskAnalysisState) -> dict:
    # pull risk_assessed_clauses + metadata_block from state
    # call report_builder.generate_report(...)
    # return {"final_report": <ContractReport.to_dict()>, "report_artifacts": {<paths>}}
```
Don't touch `src/workflow/graph.py`.

### Group 12 — Final smoke

Full `pytest tests/ -q`, expect Stage 4 tests all passing and the same 10 pre-existing failures unchanged.

### After Group 12 — Steps 5 and 6

- **Step 5**: ask the lead for the live Gemini API key. Point at https://aistudio.google.com/app/apikey, free tier. Tell them to paste into `.env` at `GEMINI_API_KEY=`. Don't ask them to share with you. Wait for confirmation.
- **Step 6**: live demo on the fixture, show four artifact paths/sizes, head/tail of Markdown, second-run cache hit ~100%, timing breakdown, no errors. Write `STAGE4_DEMO_RESULTS.md`.

---

## HARD RULES (non-negotiable, repeated)

1. Never send full contract text or all clauses in a single LLM call. Inputs are clause-level structured objects only.
2. The LLM is used only by `explainer.py` and `pattern_deriver.py` tier-2. No LLM calls hidden in aggregator/recommender/report_builder/renderers.
3. All LLM calls go through `src/stage4_report_gen/llm_client.py`.
4. API key from environment only. `.env` at project root. Never hardcoded, never logged, never committed.
5. Caching mandatory. SHA-256-keyed disk cache under `.cache/gemini/`.
6. Rate-limit handling mandatory. Token bucket at 12 RPM, exponential backoff on 429, max 3 retries, soft-fail to placeholder string.
7. Failure isolation — failed LLM call must not abort the report.
8. `logging` module, not `print()`.
9. Use shared dataclasses in `src/common/schema.py`. Extend, don't redefine.

## DESIGN DECISIONS LOCKED (`docs/STAGE4_DECISION_LOG.md`)

1. **Zero-clause case**: `final_score=0.0` with `note="no clauses assessed"` in `ScoreBreakdown`, surfaced into the rendered PDF/DOCX score section.
2. **Pattern deriver tier-2**: gated behind `pattern_deriver.use_llm_tier2` (env override `STAGE4_PATTERN_DERIVE_USE_LLM`), default `true`. Fires on tier-1 unknown_pattern. Cached. Every tier-2 invocation logged at INFO with `clause_id` + resulting code.
3. **Renderer file naming**: `output/<document_id>/<document_id>.{json,md,pdf,docx}`. Subdir per document, no timestamp suffix, overwrite on re-run.
4. **Fixture realism**: synthesized CUAD-style, ≥8 clauses across required types, mixed risk levels, ≥1 `overridden=true`, ≥1 `confidence < 0.6`, ≥1 `cross_references`.
5. **Gemini model**: `gemini-2.5-flash` default, overridable via `GEMINI_MODEL` env var.

## OUT OF SCOPE — DO NOT MODIFY

- `src/stage1_extract_classify/` — Stage 1+2 code
- `src/stage3_risk_agent/` — Stage 3 code
- `src/workflow/graph.py` — LangGraph wiring (only `src/stage4_report_gen/nodes.py` is in scope)
- The CUAD dataset
- The training pipeline
- `data/processed/` and `data/review/` — read-only reference

## STAGE 4 DATACLASSES (in `src/common/schema.py`)

- `ClauseObject`, `ExtractionResult` — Stage 1+2
- `SimilarClause`, `AgentTraceEntry`, `RiskAssessedClause` — Stage 3 → 4
- `Recommendation` — `text, market_standard, fallback_position, priority, match_level`
- `MissingProtection` — `clause_type, importance, rationale`
- `ClauseReport` — `clause_id, document_id, clause_text, clause_type, clause_type_original, risk_level, risk_pattern, risk_explanation, polished_explanation, recommendation, confidence, overridden, similar_clauses`
- `ScoreBreakdown` — `high_count, medium_count, low_count, base_score, missing_critical_or_important, missing_boost, final_score, note`
- `ExecutiveSummaryDigest` — `metadata, statistics, top_high_risk, missing_protections, metadata_provided`
- `ContractReport` — `document_id, summary, statistics, high_risk, medium_risk, low_risk, low_risk_summary, missing_protections, overall_risk_score, score_breakdown, total_clauses, metadata, generated_at, models_used`
- `ReportArtifacts` — `document_id, report, json_path, markdown_path, pdf_path, docx_path, duration_seconds, cache_hit_rate`

## RED FLAGS — STOP AND ASK THE LEAD

- `ARCHITECTURE.md` content disagrees with this prompt or the handoff (other than the FLAN-T5 → Gemini change which is done).
- Mistral-related code surfaced anywhere in Stage 4 paths.
- LLM call appearing outside `llm_client.py`, `explainer.py`, or `pattern_deriver.py` tier-2.
- A test that requires a live Gemini call.
- Hardcoded API key anywhere.
- Free-tier rate limit being exceeded during normal testing.

## BEHAVIORAL GUARDRAILS

- Show full file contents — don't say "rest unchanged".
- Use markdown link syntax `[file.py:42](src/file.py#L42)` for code references.
- After each logical group: run relevant tests, show output, do not advance until tests pass.
- Don't invent files outside the deliverable structure — ask first if you genuinely need one.
- Never log the API key. Never commit `.env`.
- Default to no comments in code; only when WHY is non-obvious.
- `logging` not `print()`.
- Type hints on all signatures.

## START HERE

After completing the FIRST ACTIONS section above:

1. Begin **Group 7 (Renderers)** per the spec above.
2. Build in this order: `json_renderer.py` → `markdown_renderer.py` → `pdf_renderer.py` → `docx_renderer.py`. Markdown is the reference content layout.
3. Add 6–10 renderer tests. Run `pytest tests/test_stage4.py -q`, show output, stop, then ask the lead for OK to proceed to Group 8.
4. Continue through Groups 8–12 in order with the same per-group test → show → confirm cycle.
5. After Group 12: Step 5 (request API key) and Step 6 (live demo on fixture).

End of resume prompt.
