# Stage 3 Synthetic Label Generation — Design Discussion

> Shareable discussion notes on how we should generate synthetic risk labels for the CUAD clauses that feed Stage 3 (risk classifier training + RAG corpus). Intended for teammates who are generating or auditing labels.

## What are we labeling and why?

- **Input**: ~6,702 positive clause spans from CUAD (510 contracts × 41 clause types; see `docs/all_positive_spans.json`).
- **Output**: for each clause, a `risk_level` (`LOW | MEDIUM | HIGH`) plus a human-readable reason.
- **Two downstream consumers**:
  1. **Training data** for the DeBERTa risk classifier at inference time — classifier sees one clause in isolation and predicts the label.
  2. **RAG corpus** — each labeled clause gets embedded into FAISS so the Stage 3 agent can retrieve precedents (similar past clauses with known risk).

Because both consumers operate on one clause at a time, label consistency per clause text matters more than inter-clause nuance. This shapes several decisions below.

## Key design questions we discussed

### 1. Should each clause be labeled in isolation, or with its sibling clauses from the same contract?

**Decision: Isolation-primary.** Each clause is sent to the LLM on its own.

**Why:**
- The classifier sees one clause at a time at inference, so training labels should come from the same distribution.
- The same clause text appearing in two different contracts must get the same label — otherwise training supervision is inconsistent.
- Document-level prompts introduce position bias (order of clauses in the prompt shifts scores) and a "risk-spreading" tendency where the LLM balances labels across siblings.
- RAG retrieval is clause-to-clause; no document context at retrieval time either.

**Trade-off acknowledged:** pure isolation misses real cross-clause interactions (a broad Indemnification is less scary next to a tight Cap On Liability). Plan: if Pass 1 validation reveals interaction as a real gap, add an **optional Pass 2** over a narrow set of interaction-sensitive pairs (Indemnification↔Cap On Liability, Exclusivity↔Termination For Convenience, etc.). Pass 2 output is stored in a separate field (`risk_level_contextual`) — RAG enrichment only, **not** mixed into classifier training labels.

### 2. What should the "reason" field look like?

**Decision: Split the reason into two fields.**

- `risk_driver` — the specific phrase or feature that drives the score. Examples: `"unlimited indemnification"`, `"no geographic scope limit"`, `"auto-renewal with no opt-out"`.
- `risk_reason` — one sentence explaining why that driver produces the chosen level.

**Why:** manual auditors can scan `risk_driver` in ~2 seconds per clause vs. ~20 seconds for a full sentence. Systematic errors (LLM always keying on irrelevant features, or just restating the clause type) surface immediately.

### 3. How do we validate that the labels are mostly correct?

**Decision: Stratified audit, not random sampling.**

- Sample ~10 clauses per `(clause_type × risk_level)` cell (~400 clauses total for a 41-type schema).
- Target **≥ 80% agreement** between human reviewer and LLM label. Below that → revise prompt, re-run affected slices.
- Save disagreements to a separate file — they become a hard-eval set and signal for prompt tuning.
- **Confidence-weighted spot check** in addition: inspect the lowest-confidence 5% of outputs regardless of label (errors cluster there).

## Other decisions worth calling out

### Metadata clauses route away from risk classifier entirely (Option B)

**Decision (2026-04-17):** the 5 metadata CUAD types — `Document Name`, `Parties`, `Agreement Date`, `Effective Date`, `Expiration Date` (~2,292 of 6,702 positive spans, ≈34%) — are **not** risk-labeled at all. They do not pass through the synthetic labeling pipeline and they do not enter the Stage 3 risk classifier's training data.

**Why not just auto-label them LOW?**

- Training noise — the classifier would spend ~30% of its training data learning "metadata → LOW" instead of learning actual risk reasoning.
- Report noise — the final risk report would list ~5 "LOW" items per contract that are just the document title, parties, and dates. Useless for a user reviewing risk.
- These clauses ARE useful — for the report's "Contract Metadata" header section. Routing them there keeps them available without polluting the risk analysis.

**Implementation:**

- `src/common/schema.py` adds a `clause_kind: Literal["metadata", "risk_bearing"]` field on `ClauseObject`. This is a routing distinction, NOT a fourth risk label — the risk schema stays LOW/MEDIUM/HIGH.
- Stage 1/2 extraction still extracts all 41 CUAD types (DeBERTa training unchanged).
- Post-extraction, routing splits clauses into the two buckets.
- Stage 3 risk pipeline only processes `risk_bearing` clauses.
- Stage 4 report surfaces metadata in a "Contract Metadata" section, risk in a "Risk Assessment" section.

**Borderline cases kept as `risk_bearing` for now:** `Renewal Term` (auto-renewal has real risk implications), `Notice Period To Terminate Renewal` (procedural; revisit if prompt iteration shows it's always trivially LOW).

### Deduplication before API calls

**Decision (2026-04-17):** deduplicate identical clause texts before the labeling loop. Call the LLM once per unique (whitespace-normalized) clause_text; fan the returned label back out to all rows sharing that text.

**Why:** many clauses share boilerplate language verbatim across contracts; labeling them independently wastes API calls and can produce inconsistent labels on identical text.

**Measured savings on the 6,702 positive spans:**

| Strategy | Unique rows | API calls saved |
|---|---|---|
| No dedup | 6,702 | 0 |
| Exact text dedup | 5,655 | 1,047 (15.6%) |
| Whitespace-normalized text dedup | 5,603 | 1,099 (16.4%) |

Combined with Option B metadata routing, expected full-run API count drops from **6,702 → ~4,100** (≈38% reduction).

**Note on identical text labeled under multiple CUAD categories** (e.g., Uncapped Liability + Cap On Liability annotating the same region): treat as one unique text → one label → fan to all category rows. If prompt iteration reveals category-framing actually shifts labels on identical text, revisit.

### Perspective anchor in the prompt

The prompt must state whose risk we're assessing. We've agreed on: **"from the perspective of the party signing the contract (counterparty to the drafter)"**. Without this, the LLM silently switches between buyer and seller views across clauses and labels become inconsistent.

### Clause-type context in the prompt

Inject a one-line definition of what the clause type governs (e.g., `"Indemnification = which party bears legal costs for third-party claims"`). Prevents the LLM from having to infer semantics from the label string alone — some CUAD types are ambiguous (`Rofr/Rofo/Rofn`, `Non-Transferable License`).

### Calibration anchors in the prompt

Include one inline example each for LOW / MEDIUM / HIGH so the model has concrete boundaries. Prevents drift across API calls.

### Backend for the full run (still open)

Options considered:
- **Gemini free tier** (Google AI Studio) — free, rate-limited; overnight run feasible. Easiest.
- **Self-hosted Qwen-32B** on GPU server — matches the `ARCHITECTURE.md` plan, zero per-token cost, fully reproducible.
- **Claude Code sessions (Team plan quota)** — viable for small runs; awkward for the full 6,702 due to per-turn overhead and rolling quota limits.

**Decision deferred** until prompt is stable and we can test it across candidates.

## Auditing existing labels — checklist

If teammates have already generated labels (e.g., via Copilot), use this checklist to check for the issues we discussed:

- [ ] **Perspective**: does the labeling prompt specify whose risk? If not, spot-check a sample for inconsistency (e.g., same clause type labeled both HIGH and LOW across different contracts because perspective flipped).
- [ ] **Reasoning quality**: are the reasons specific (naming a clause feature) or generic (just restating the clause type)? Example of generic: "This is an indemnification clause so it's HIGH risk" — tells you nothing.
- [ ] **Metadata noise**: are `Document Name` / `Parties` / `Agreement Date` labeled anything other than LOW? If yes, there's noise the classifier will learn from.
- [ ] **Clause-type skew**: count label distribution per clause type. Red flags:
  - Every `Governing Law` labeled the same level (should depend on jurisdiction).
  - Every `Uncapped Liability` labeled LOW (definitionally wrong).
  - Every informational type showing MEDIUM (metadata noise).
- [ ] **Calibration drift**: pick 10 clauses of the same clause type with similar text. Do they get consistent labels? If labels vary arbitrarily, calibration is off.
- [ ] **Auditability**: is the reasoning in a structured form (driver + sentence) or one blob? If blob-only, manual auditing at scale is ~10× slower.
- [ ] **Confidence scores**: are they reported? If not, there's no way to prioritize which labels to spot-check.

## Three-phase rollout plan

1. **Prompt iteration** — build a ~30-clause gold set (stratified across clause types + edge cases), hand-label as ground truth, iterate LLM prompt until ≥ 80% agreement.
2. **Pilot run** — freeze prompt, label ~500 clauses via chosen backend, audit ~100 stratified. Validates that the prompt holds up at scale.
3. **Full run** — label the remaining ~4,800 non-informational clauses. Checkpointing + stratified post-audit.

Pilot and full run are planned separately once the prompt is frozen.

## References

- `scripts/generate_synthetic_labels.py` — existing labeling script (Anthropic API, isolated labeling). Prompt will be updated during Phase 1.
- `scripts/build_gold_set.py` — deterministic 25-clause gold set builder (stratified: 8 Group A + 10 Group B + 4 edge + 3 random).
- `data/processed/all_positive_spans.json` — the 6,702 positive clauses (Stage 1/2 output). Metadata types excluded from labeling.
- `data/raw/master_clauses.csv` — CUAD source data.
- `data/reference/cuad_category_descriptions.csv` — Atticus's official one-line descriptions of the 41 CUAD categories. Authoritative for prompt context injection.
- `data/synthetic/gold_set.json` — output of the gold-set builder (gitignored).
- `ARCHITECTURE.md` — overall pipeline; specifically the Stage 3 model table and "Synthetic Risk Labels" data contract. (Known to be outdated on a few illustrative examples — will be updated after Stage 3 design is confirmed.)
- `configs/stage3_config.yaml` — Stage 3 hyperparameters including synthetic label config.
