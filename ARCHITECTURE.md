# Legal Contract Risk Analyzer — Architecture Reference

> **Purpose**: This file is the single source of truth for AI assistants (Copilot, Claude, GPT) working on this codebase. Open this file or reference it (`#file:ARCHITECTURE.md`) when starting any coding session.

## Project Summary

A modular ML pipeline that analyzes legal contracts and flags risky clauses. Takes a contract PDF/text as input, extracts and classifies clauses, assesses risk using an agent with RAG, and generates a structured risk report.

## Architecture (3 Stages)

```
Contract PDF/Text
       │
       ▼
┌─────────────────────────────┐
│  Stage 1+2: Extract &       │  Single DeBERTa-base model
│  Classify (combined)        │  CUAD dataset, QA format
│  Output: clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 3: Risk Detection    │  DeBERTa → LangGraph ReAct Agent
│  DeBERTa default signal     │  DeBERTa-v3-base (risk classifier)
│  Agent verifies with tools  │  Qwen3-30B Q4_K_XL (agent + explanation)
│  Tools: precedent_search,   │  all-MiniLM-L6-v2 (embeddings)
│         contract_search     │  FAISS vector store
│  Output: risk-assessed      │
│          clause objects     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  Stage 4: Report Generation │  Python aggregation code
│  Hybrid: code + LLM        │  Qwen3-30B (executive summary)
│  Output: structured risk    │  Lookup table (recommendations)
│          report             │
└─────────────────────────────┘
```

## Stage 3 Architecture — DeBERTa + LangGraph ReAct Agent (Updated 2026-05-04)

> **Decision**: Every clause runs through the LangGraph ReAct agent. DeBERTa classifies
> first and its label is injected into the agent's system prompt as the default signal.
> The agent uses `precedent_search` and `contract_search` to gather evidence, then
> produces a final label and explanation. It is instructed to confirm DeBERTa unless
> tool evidence provides clear, convergent consensus for a different label.

### Why This Design

- **DeBERTa as prior** — 3,400 labeled CUAD clauses are baked into the classifier. The agent
  inherits this signal for free rather than reasoning from scratch.
- **Agent verifies, not re-classifies** — LLM alone (without retrieval) was 47–58% accurate
  on overrides vs DeBERTa's 59.3% baseline. Grounding overrides in tool consensus raises the
  bar and reduces arbitrary LLM drift.
- **Full agent on every clause** — avoids a confidence threshold that was fragile to calibrate
  and caused the easy cases (high-conf) to never benefit from retrieval context.
- **Signing-party ambiguity resolved at inference** — `contract_search` gives the agent the
  Parties clause and sibling clauses to determine who bears the risk, which is the #1 driver
  of label ambiguity (81% of human-review flips).

### High-Level Flow

```
Clause  (from Stage 1+2)
     │
     ▼
┌──────────────────────────────┐
│  DeBERTa Risk Classifier     │
│  input: clause_type + text   │
│         + signing_party      │
│  output: (label, confidence) │
└────────────┬─────────────────┘
             │  label + conf passed into system prompt
             ▼
┌──────────────────────────────┐
│  LangGraph ReAct Agent       │  Qwen3-30B, max_iterations=2
│  system: "DeBERTa says X.    │
│   Confirm unless tools show  │
│   clear consensus otherwise" │
│                              │
│  Tool loop:                  │
│    precedent_search (always) │  FAISS top-5, min_sim=0.75
│    contract_search (if       │  sibling clauses, same contract
│      party/context matters)  │
│                              │
│  Synthesis call:             │  structured output (function_calling)
│    RiskAssessment {          │
│      final_label,            │
│      explanation,            │
│      override_reason }       │
└────────────┬─────────────────┘
             ▼
   RiskAssessedClause → Stage 4
```

### Agent Decision Rule (system prompt)

```
- Tool evidence agrees with DeBERTa, or is weak/split → confirm DeBERTa's label.
- Strong consensus from BOTH tools contradicts DeBERTa → may override, but must
  explain exactly what evidence outweighs the DeBERTa signal.
- Do NOT override based on solo legal reasoning — only on convergent tool evidence.
```

### Tool Definitions

- **`precedent_search`** (vector RAG) — FAISS similarity lookup over the labeled training
  corpus (4,276 vectors, train split only — test clauses excluded to prevent leakage).
  Embeddings: `all-MiniLM-L6-v2`. Only returns clauses with similarity ≥ 0.75 — every
  result is a strong semantic match. Returns `{clause_text, clause_type, risk_level,
  similarity}`. Fast (IndexFlatIP, exact cosine, microseconds per query).

- **`contract_search`** (structured lookup) — given `current_clause_id`, returns all other
  clauses already extracted from the same contract by Stage 1+2. No embeddings, no LLM calls
  inside the tool. Contracts average ~9 non-metadata clauses so all fit in context. Purpose:
  resolve party-role ambiguity and cross-clause interactions (e.g., "IP already assigned
  elsewhere — this clause is administrative paperwork").

### Worked Example

```
clause_type: "Ip Ownership Assignment"
clause_text: "Consultant hereby assigns to Company all right, title,
              and interest in any deliverables..."
signing_party: "Jane Smith Consulting"
DeBERTa:      HIGH (0.45 confidence) ← passed to agent as default
```

Agent loop:

```
Turn 1  → calls precedent_search("Consultant hereby assigns...", k=5)
          Returns:
            N1: LOW  — "signing party is the recipient of rights"
            N2: HIGH — "one-party-committed IP assignment to a vendor"
            N3: LOW  — "standard admin paperwork, IP already owned upstream"
            N4: LOW  — "mutual carve-out"
            N5: MEDIUM — "conditional on acquisition"

Turn 2  → 4 of 5 precedents are LOW; key question is party direction.
          calls contract_search(current_clause_id="...")
          Returns: Compensation clause shows AT&T pays Consultant $50K.
                   Services clause shows Consultant provides services to AT&T.

Turn 3  → Confirmed: AT&T is the client; Consultant transfers IP as standard
          work-for-hire. Signing party (Consulting) is at HIGH risk per DeBERTa,
          but tool consensus (4 LOW precedents + contract confirms work-for-hire)
          overrides.
          → final_label: LOW, override_reason: "4/5 precedents LOW + contract
            confirms Consultant is the services provider assigning IP to client."
```

### Implementation Notes

**Structured output workaround** — `langchain-openai ≥ 1.2.0` defaults
`with_structured_output()` to `method="json_schema"`, which llama.cpp rejects with
HTTP 400. Fix: always pass `method="function_calling"` explicitly. Applied to the
synthesis call in `_agent_path`.

**Agent synthesis call** — `create_react_agent` is called without `response_format`
(LangGraph's internal structured-response node uses the broken default method). Instead,
after the ReAct loop, a single explicit `with_structured_output(RiskAssessment,
method="function_calling")` call is made with a clean single-turn prompt containing the
agent's final reasoning text. Passing the full ReAct history (which contains existing
`tool_call` messages) to structured output causes the model to return plain text instead
of filling the schema.

**FAISS index** — `data/faiss_index/clauses.index` (4,276 vectors, IndexFlatIP) +
`data/faiss_index/clauses.json` (parallel metadata). Built from `training_dataset.json`
train split only (`splits_path` filter in `build_index()`). Entry point:
`scripts/build_faiss_index.py`.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| All clauses → agent | No confidence gating | Threshold was fragile; easy cases benefit from retrieval context too |
| DeBERTa as default signal | Label + conf in system prompt | 3,400 CUAD examples baked in; agent inherits vs. reasoning from scratch |
| Override policy | Tools must show clear consensus | LLM-alone overrides were 47–58% accurate vs DeBERTa's 59.3% |
| No `deberta_classify` tool | DeBERTa already ran | Redundant to expose it again; result already in system prompt |
| Precedent retrieval | FAISS (all-MiniLM-L6-v2), min_sim=0.75 | Only strong matches influence the agent |
| Contract context | `contract_search` structured lookup | Data already typed post-Stage 1+2; resolves party-role ambiguity |
| Signing party | Passed to DeBERTa + agent | #1 driver of label ambiguity (81% of human-review flips) |

### Alternative Architectures Tried

- **Hybrid Confidence-Gated** (v1, 2026-04-23) — high-conf clauses took a fast path (1 LLM
  call, no tools); low-conf escalated to the ReAct agent. *Removed*: threshold was fragile;
  high-conf clauses never got retrieval context; and eval showed LLM overrides on the
  low-conf path were 47–58% accurate vs DeBERTa's 59.3% baseline.
- **DeBERTa+FAISS Ensemble, no LLM for labels** (2026-05-04) — deterministic ensemble,
  LLM for explanation only. *Reverted*: removed the agent loop entirely, which is a core
  project learning goal; also FAISS coverage at 0.75 is only 42% so the ensemble rarely fired.
- **Static LangGraph** — fixed tool sequence, DeBERTa label always final. *Superseded*:
  went to full ReAct instead.
- **RAC (Retrieval-Augmented Classification)** — retrieved neighbors as DeBERTa training
  input. *Deferred*: risk of label leakage; harder training pipeline.

## Open Design Question — METADATA in DeBERTa Training (v1: Option 1 chosen)

> **Context**: During manual label review, we found that many HIGH↔LOW flips between Qwen and Gemini
> are caused entirely by not knowing who the signing party is — both models reason correctly about the
> clause text but assume different parties. Without METADATA, DeBERTa faces the same ambiguity.

### The problem
The same clause text can be LOW or HIGH depending on who signed:
- "Vendor grants AT&T perpetual irrevocable license" → HIGH if Vendor signs, LOW if AT&T signs
- No amount of fine-tuning on clause text alone resolves this

### Three options to evaluate after DeBERTa baseline:

**Option 1 — clause_type only (v1 CHOSEN, baseline)** ✅
`[CLS] clause_type [SEP] clause_text [SEP]`
- Partial signal — clause type hints at risk direction but doesn't resolve signing party
- Establishes baseline accuracy ceiling
- **Chosen for v1 because**: we need to measure how far text + clause_type alone can go before adding complexity. The Hybrid architecture absorbs the residual ambiguity through the low-confidence escalation path (Option 3 below, handled at inference by the reasoning agent).

<!-- Updated 2026-05-13: Option 2 effectively in place — superseded "Deferred" status -->
**Option 2 — Add party-text injection at training + inference time** ✅
`[CLS] signing_party_text [SEP] clause_type [SEP] clause_text [SEP]`
- Stage 1+2 extracts the "Parties" clause; its text is passed to DeBERTa as the first segment at both train and inference time. See `src/stage3_risk_agent/risk_classifier.py::extract_signing_party()` and Ens-F training in `src/stage3_risk_agent/train.py`.
- The classifier learns party direction directly rather than relying on the agent loop to resolve it. The agent still uses `contract_search` to disambiguate when DeBERTa is low-confidence.
- **Locked 2026-04-30** (commit `66d5bbc`): "signing-party metadata + Ens-F (macro_f1=0.607)".

**Option 3 — Reasoning model resolves METADATA ambiguity (built into Hybrid architecture)** ✅
- DeBERTa classifies on text alone, will be uncertain on party-dependent cases
- Low-confidence predictions escalate to the reasoning agent
- Agent uses METADATA + `contract_search` + `precedent_search` to confirm or override DeBERTa's label
- No retraining needed — handles ambiguity at inference time
- **Already part of v1** through the Hybrid architecture above.

**v1 Plan**: Option 1 for DeBERTa + Option 3 for inference-time ambiguity resolution (via the Hybrid architecture). Option 2 remains available as a v2 upgrade if the baseline is insufficient.

## Labeling Review Learnings (Discuss After DeBERTa Baseline)

These patterns emerged from manual review of 239 HIGH↔LOW flip cases and should inform
model design, RAG strategy, and agent reasoning.

### 1. Signing Party Ambiguity is the #1 Label Driver
81% of Rajnish's 60 rows were LOW — not because the clauses were genuinely low risk,
but because the signing party turned out to be the *recipient* of rights, not the grantor.
Both Qwen and Gemini reasoned correctly about clause text but assumed different parties.
**Implication**: A model trained on clause text alone will have a hard accuracy ceiling on
IP Ownership, License Grant, and Affiliate License clause types specifically.

### 2. Cross-Contract Clause Interaction Changes Labels
Several clauses were correctly assessed only after reading *other clauses* in the same contract:
- MR-095 (IP assistance clause): looked HIGH alone, but other clauses showed IP already
  assigned elsewhere — this was just administrative paperwork → LOW
- MR-234 (1-copy software restriction): looked restrictive alone, but full production license
  existed in another clause — test/backup restriction was supplemental → LOW
**Implication**: RAG retrieval should include *same-contract* clause context, not just
similar clauses from other contracts. The contract search tool in Stage 3 is critical.

### 3. Clause Type Labels Are Noisy at the Boundary
- Volume Restriction includes supply guarantees, image size limits, content duration caps —
  not just purchase minimums. Same clause type, very different risk patterns.
- Non-Transferable License ranges from LOW (standard distribution) to MEDIUM (M&A constraint)
  depending on deal context (affiliate carve-outs, term length, corporate structure).
**Implication**: clause_type alone is a weak feature. The RAG examples need to be diverse
within each type — DeBERTa must learn intra-type variation, not just type-level patterns.

### 4. Mutual vs One-Sided is a Strong Signal
For Uncapped Liability: mutual consequential damage exclusions are almost always LOW.
One-sided caps (only one party's liability limited) trend MEDIUM-HIGH.
**Implication**: "mutual" / "either party" / "both parties" keywords are strong LOW signals
for liability clause types. Worth checking if DeBERTa learns this or needs explicit feature.

### 5. Context-Dependent Clauses Need Confidence Flagging
Some clauses were genuinely ambiguous even with METADATA (MR-094 truncated GSK clause,
MR-093 spin-off restructuring). These warrant MEDIUM and low DeBERTa confidence scores.
**Implication**: confidence threshold + human review escalation path is not optional —
it's needed for a reliable production system.

## Stage 3 Training Data Pipeline

> **Decided 2026-04-18, merged 2026-04-23.** Canonical source: `data/review/master_label_review.csv`.
> Full labeling history + disagreement analysis: `docs/STAGE3_LABELING.md`.

<!-- Updated 2026-05-13: dataset re-consolidated via Sonnet relabeling pass; current headline supersedes the v1 row-category breakdown below -->
> **Current locked state (per `docs/STAGE3_LABELING.md`):** **4,375 rows = 4,276 hard + 99 soft**, after a Claude Sonnet relabeling pass replaced 322 inconsistently-labeled human/Gemini-Pro rows (`SONNET_REVIEW`) and consolidated 1,228 three-way-agreed rows (`SOFT_LABEL_V2_AGREED`). Train split: 3,398 rows; test split: 452 rows. The category breakdown immediately below is the **v1 pipeline (preserved for historical context)**; for the current row-category counts and label provenance, see `docs/STAGE3_LABELING.md`.

Risk labels were produced through a multi-labeler + human-review pipeline on the 6,702 positive
spans output by Stage 1+2.

1. **Metadata routing.** 2,292 spans belonging to the 5 metadata clause types (Document Name,
   Parties, Agreement Date, Effective Date, Expiration Date) were auto-assigned
   `final_label="METADATA"` and never sent to labelers — they route to the Stage 4 report
   header, not the risk classifier.
2. **Primary labeling pass (4,410 risk-relevant spans).** Qwen-30B (non-reasoning, local
   llama-server, temp=0) and Gemini 2.5 Flash (Google API, JSON mode, temp=0) each labeled
   every risk-relevant span independently.
3. **Reconciliation by disagreement type.** Where the two labelers agreed, the label was taken
   as-is (AGREED, 2,735). Disagreements were routed by severity:
   - **Extreme HIGH↔LOW flips (239 rows)** → 4 human reviewers (~60 rows each by whole clause
     type) filled `final_label` manually. 4 rows turned out to be truncated fragments and
     were re-flagged as ERROR (final count: 235 labeled MANUAL_REVIEW).
   - **Boundary disagreements on focus types (87 rows)** — MEDIUM↔HIGH or LOW↔MEDIUM on
     Uncapped Liability, Liquidated Damages, and Irrevocable/Perpetual License → Gemini 2.5 Pro
     tiebreaker (GEMINI_PRO_REVIEW).
   - **Remaining adjacent disagreements (1,327 rows)** → kept as soft-label probability
     vectors instead of collapsing to a single hard label (SOFT_LABEL).

The `category` column records which path each row took.

### Row Categories

| Category | Rows | `final_label` source | Training role |
|---|---|---|---|
| METADATA | 2,292 | Pre-filled `"METADATA"` | **Excluded** — routes to Stage 4 report header |
| AGREED | 2,735 | Qwen == Gemini → pre-filled | **Hard label** (CrossEntropyLoss) |
| MANUAL_REVIEW | 235 | 4 human reviewers (~60 rows each) filled extreme HIGH↔LOW flips | **Hard label** (CrossEntropyLoss) |
| GEMINI_PRO_REVIEW | 87 | Gemini 2.5 Pro tiebreaker on focus-type non-flip disagreements | **Hard label** (CrossEntropyLoss) |
| SOFT_LABEL | 1,327 | Probability vector computed from Qwen + Gemini at train time | **Soft label** (KLDivLoss) |
| ERROR | 26 | Labeling errors (22) + manual-review fragments (4: MR-015, MR-170, MR-173, MR-175) | **Dropped** |

**Effective training set**: **4,384 rows** = 3,057 hard + 1,327 soft.
Hard-label class mix: LOW 44.4% (1,358), MEDIUM 32.3% (987), HIGH 23.3% (712).
Effective mix (hard + soft mass, 4,384): LOW 40.9%, MEDIUM 37.7%, HIGH 21.5%.
Mild imbalance (~2:1 majority:minority) — handle with class weights in CE, not resampling.

### SOFT_LABEL Construction

Adjacent disagreements (LOW↔MEDIUM or MEDIUM↔HIGH — never extreme HIGH↔LOW flips, which went
to human review instead) are encoded as probability distributions rather than collapsed to a
single hard label. This preserves labeler uncertainty on genuinely borderline cases.

```
Qwen=MEDIUM, Gemini=LOW   → [LOW=0.5, MEDIUM=0.5, HIGH=0.0]
Qwen=LOW,    Gemini=MEDIUM → [LOW=0.5, MEDIUM=0.5, HIGH=0.0]
Qwen=MEDIUM, Gemini=HIGH  → [LOW=0.0, MEDIUM=0.5, HIGH=0.5]
Qwen=HIGH,   Gemini=MEDIUM → [LOW=0.0, MEDIUM=0.5, HIGH=0.5]
```

**Confidence weighting (default).** Each labeler's vote is scaled by its self-reported
confidence rather than given a flat 0.5:

```
w_q = qwen_conf / (qwen_conf + gemini_conf)
w_g = gemini_conf / (qwen_conf + gemini_conf)
vec[qwen_label]   += w_q
vec[gemini_label] += w_g
```

Workaround for Qwen's `conf=0.0` artifact (11.7% of its rows — labels are valid, score is
a known model-output quirk, not genuine uncertainty): floor `qwen_conf` to `0.5` before
weighting, so the label isn't silently dropped. See `scripts/build_training_dataset.py`
for the implementation. A `--no_conf_weight` flag falls back to uniform 0.5/0.5 if we
ever want to ablate.

### Known Issue — Qwen `conf=0.0` Artifact (investigated 2026-04-23)

515 of Qwen's 4,410 output rows (11.7%) carry `confidence=0.0` despite containing
coherent, well-reasoned `risk_level` and `risk_reason` fields. Root-cause analysis:

- **Bimodal distribution**: values cluster at `0.0`, `0.6–0.9`, and `0.9–1.0` — there are
  **zero** rows in the `0.01–0.30` range. 0.0 is a discrete failure mode, not the bottom
  of a continuous scale.
- **Clause-type concentration**: heavily weighted toward verbose clause types where
  reasoning is long — Non-Transferable License (42%), Affiliate License-Licensee (36%),
  Irrevocable/Perpetual License (27%), License Grant (27%), ROFR (19%). Short / simple
  clause types are barely affected.

**Likely cause**: JSON output truncation or field-recovery fallback in the llama-server
pipeline. Qwen emits `risk_reason` before `confidence`; when reasoning is long the
response is cut off at `max_tokens` with `confidence` missing, and the post-processor
defaults the absent field to 0.0.

**Mitigations for any future labeling run**:
1. Increase `max_tokens` generously (reasoning is the longest field).
2. Reorder the JSON schema so `confidence` emits *before* `risk_reason`.
3. Use a strict JSON grammar file with llama-server (grammar-constrained decoding)
   so truncation produces a parse error rather than a silently-defaulted field.

**Current mitigation**: the floor-to-0.5 workaround above. Labels are preserved; only
the vote-weight signal is neutralized on affected rows. Not regenerating — the
reconciliation pipeline (AGREED / SOFT_LABEL / MANUAL_REVIEW assignments + already-done
human reviews) would be invalidated for marginal benefit on ~5% of the training signal.

### Loss Function — Unified Soft-Target Cross-Entropy

DeBERTa trains on mixed batches where each row carries a 3-way probability vector in its
`soft_label` field (built by `scripts/build_training_dataset.py`):

- Hard rows store one-hot vectors: `[1,0,0]` / `[0,1,0]` / `[0,0,1]`
- Soft rows store probability distributions: e.g., `[0.4, 0.6, 0.0]`

Because both row types share a uniform target format, a single loss covers both:

```python
loss_fn = nn.CrossEntropyLoss(weight=class_weights)   # 3-element class weights
loss    = loss_fn(logits, soft_targets)               # logits [B,3], soft_targets [B,3]
```

`nn.CrossEntropyLoss` (PyTorch ≥ 1.10) accepts a probability distribution as target. For a
one-hot target this reduces to standard CE on the true class; for a soft target it becomes
`-Σ_i target_i · log_softmax(logits)_i` — the natural generalization. This is mathematically
equivalent to "CE for hard rows, KLDiv for soft rows" up to an additive constant (target
entropy) that does not affect gradients, but collapses to a single code path with no
per-sample dispatch.

Sharp boundaries are learned where labelers agreed (one-hot targets → standard CE).
Calibration is preserved on borderline LOW/MEDIUM and MEDIUM/HIGH cases (soft targets →
model is trained to output matching uncertainty).

<!-- Updated 2026-05-13: Option B (effective_counts) is the locked recipe; Option A caused MEDIUM collapse -->
**Class weights** (`weight` tensor above) compensate for mild imbalance. The locked recipe is
**Option B — `effective_counts`**: the per-class denominator includes soft-label probability
mass alongside hard-row counts.

```
weight_c = N_eff / (K · effective_count_c),  K = 3 classes
```

**Why Option B (locked).** The initial recipe (Option A — `hard_counts`) produced a MEDIUM
collapse: Run 1, macro_f1=0.21, MEDIUM recall ≈ 0 because soft-row probability mass on the
MEDIUM class wasn't reflected in the weights. Switching to `effective_counts` restored MEDIUM
learning and remains the production setting in `configs/stage3_config.yaml`. See `docs/STAGE3_EXPERIMENTS.md` for the run log.

**Option A — `hard_counts`** (LOW 0.749, MEDIUM 1.030, HIGH 1.442, N = 3,049 hard rows) is
preserved as an ablation flag in code but should not be used by default.

### Row Provenance (columns in `master_label_review.csv`)

| Column | Purpose |
|---|---|
| `row_num`, `review_id` | Stable identifiers. Original assignment ranges: METADATA 1–2292, AGREED 2293–5027, MANUAL_REVIEW 5028–5266, GEMINI_PRO_REVIEW 5267–5353, SOFT_LABEL 5354–6680, ERROR 6681–6702. Note: 4 MANUAL_REVIEW rows (MR-015, MR-170, MR-173, MR-175) were re-flagged to ERROR post-review and keep their original row_nums — always filter by `category`, not row_num range |
| `category` | One of 6 above — drives training-time routing (hard / soft / exclude) |
| `final_label` | Hard label, or `"METADATA"` / `"ERROR"` |
| `reviewer` | Human name, `"Gemini-2.5-Pro"`, or empty for auto-agreed rows |
| `qwen_label`, `gemini_label`, `copilot_label` | Original labeler outputs, preserved for audit |
| `qwen_confidence`, `gemini_confidence` | Used by the confidence-weighted soft-label variant |
| `qwen_reason`, `gemini_reason` | Per-labeler rationale (human-readable, for review UI + debugging) |
| `notes` | Reviewer-added comments, fragment flags |

## Directory Structure

<!-- Updated 2026-05-13: synced with actual filesystem. Added app/ (FastAPI), src/workflow/ (experimental DAG), expanded scripts/ and docs/, removed predict.py/data_loader.py and notebook stubs that were never built. Empty/dead-code files omitted — see Known Issues #6/#7. -->

```
AIML_project/
├── ARCHITECTURE.md          ← THIS FILE (AI context reference)
├── CLAUDE.md                ← Cross-session pointer file; downstream of ARCHITECTURE.md
├── README.md
├── CODE_README.md           ← Engineering-side readme (alt entry point)
├── CODE_ARCHITECTURE.md     ← Engineering-side architecture sketch
├── requirements.txt         ← Python dependencies (see Known Issues #4)
├── pyproject.toml
├── uv.lock
├── .env.example             ← Env template (OLLAMA_* keys are legacy)
├── .gitignore               ← Ignores .github/, model weights, data/raw|processed|synthetic
├── .github/                 ← Gitignored — local-only Copilot instructions
├── configs/
│   ├── stage1_config.yaml   ← Stage 1+2 hyperparams
│   ├── stage3_config.yaml   ← Stage 3 training recipe + agent/RAG inference settings
│   └── stage4_config.yaml   ← Stage 4 summary LLM settings (Qwen)
├── docs/
│   ├── LegalAgents.docx                                       ← Original proposal
│   ├── Legal_Contract_Risk_Analyzer_Research_Analysis.docx    ← Research analysis v1
│   ├── Legal_Contract_Risk_Analyzer_Project_Plan_v2.docx      ← Finalized plan v2
│   ├── STAGE3_LABELING.md           ← Canonical label-pipeline reference (4,375 rows)
│   ├── STAGE3_EXPERIMENTS.md        ← Every training run, ablation, closed hypothesis
│   ├── STAGE3_TRAINING_NOTES.md     ← Locked Section A–G training decisions
│   ├── STAGE1_REVIEW_NOTES.md       ← Stage 1+2 alignment notes
│   ├── OPTIONAL_ENHANCEMENTS.md     ← OE-1…OE-N deferred work
│   ├── TASK_LIST.md                 ← Master task tracker
│   ├── HANDOFF.md                   ← Cross-teammate handoff state
│   ├── REVIEW_REQUEST.md            ← External-review framing doc
│   └── dataset_insights.md          ← CUAD per-clause-type breakdown
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── schema.py             ← Shared dataclasses (ClauseObject, RiskAssessedClause, RiskReport, …)
│   │   ├── constants.py          ← CUAD clause types + question templates
│   │   ├── preprocessing.py      ← PDF/DOCX/TXT text extraction
│   │   └── utils.py              ← load_config, make_llm, save_json, metric helpers
│   ├── stage1_extract_classify/
│   │   ├── __init__.py
│   │   ├── model.py              ← ClauseExtractorClassifier (DeBERTa QA, bf16 inference)
│   │   ├── pipeline.py           ← End-to-end Stage 1+2 flow
│   │   ├── train.py              ← Fine-tuning entry
│   │   ├── preprocess_cuad.py    ← CUAD → QA-format conversion
│   │   ├── preprocessing.py      ← Contract-text preprocessing
│   │   ├── constants.py          ← Stage-1-local constants
│   │   ├── baseline.py           ← spaCy + regex baseline
│   │   └── evaluate.py           ← Extraction + classification metrics
│   ├── stage3_risk_agent/
│   │   ├── __init__.py
│   │   ├── agent.py              ← assess_clauses() entry; LangGraph ReAct loop; parallel workers
│   │   ├── tools.py              ← precedent_search (FAISS) + contract_search (siblings)
│   │   ├── risk_classifier.py    ← Wraps scripts/infer.py RiskClassifier; extract_signing_party()
│   │   ├── embeddings.py         ← FAISS index builder + cached query
│   │   ├── train.py              ← DeBERTa-v3 trainer (Ens-F CE + CORN)
│   │   ├── synthetic_labels.py   ← LLM-based label generation (historical)
│   │   └── evaluate.py           ← Risk detection metrics + ablation
│   ├── stage4_report_gen/
│   │   ├── __init__.py
│   │   ├── aggregator.py         ← Deterministic grouping + risk score
│   │   ├── recommender.py        ← Lookup table (clause_type, risk_level) → recommendation
│   │   ├── report_builder.py     ← Assembles RiskReport + Qwen executive summary
│   │   └── evaluate.py           ← ROUGE + optional human eval
│   └── workflow/                  ← ⚠ EXPERIMENTAL / not in active pipeline (see "End-to-End Workflow" §)
│       ├── __init__.py
│       ├── state.py              ← RiskAnalysisState (LangGraph TypedDict)
│       └── graph.py              ← Stage 1→3→4 DAG with Map-Reduce fan-out
├── app/                           ← FastAPI service layer (see "API Layer" §)
│   ├── main.py                   ← FastAPI app + CORS + router wiring
│   ├── routers/
│   │   ├── documents.py          ← /api/v1/documents — FAISS index metadata
│   │   ├── stage1_extract.py     ← /api/v1/stage1/analyze — full pipeline endpoint
│   │   ├── stage3_agent.py       ← /api/v1/stage3/assess — Stage 3 only
│   │   └── stage4_report.py      ← /api/v1/stage4/report — Stage 4 only
│   ├── services/
│   │   ├── stage1_extract_svc.py ← Synchronous Stage 1→3→4 runner (executor-friendly)
│   │   ├── stage3_agent_svc.py   ← Stage 3 wrapper around assess_clauses()
│   │   └── stage4_report_svc.py  ← Stage 4 wrapper around build_report()
│   └── schemas/
│       ├── domain.py             ← Pydantic FinalRiskReport, RiskAssessedClause (API-facing)
│       └── requests.py           ← Pydantic Stage3Request, Stage4Request, ClauseInput
├── data/
│   ├── raw/                      ← Contract PDFs/DOCX (gitignored)
│   ├── processed/                ← all_positive_spans.json, training_dataset.json, splits.json
│   ├── synthetic/                ← Raw LLM labeler outputs (gitignored except .gitkeep)
│   ├── review/                   ← master_label_review.csv + per-reviewer CSVs
│   ├── reference/                ← cuad_category_descriptions.csv
│   ├── output/                   ← Sample API response JSON
│   └── faiss_index/              ← Generated artifact (gitignored); built by scripts/build_faiss_index.py
├── notebooks/
│   ├── cuad_explore.ipynb        ← Dataset exploration (only live notebook)
│   ├── EXPERIMENT_NOTES.md       ← Free-form run notes
│   ├── analyze_relabel_v2.py     ← Ad-hoc relabel-set analysis
│   ├── plot_label_consistency.py
│   ├── tokenisation.py
│   ├── test_clause.py
│   ├── learn_sliding_window.py
│   ├── run22_errors.json         ← Error-analysis dump
│   └── label_consistency.png
├── tests/                         ← Scaffolding only — every method raises NotImplementedError (Known Issues #6)
│   ├── test_preprocessing.py
│   ├── test_stage1.py
│   ├── test_stage3.py
│   └── test_stage4.py
└── scripts/
    ├── download_cuad.py
    ├── build_faiss_index.py            ← Builds data/faiss_index/clauses.index from train split
    ├── build_splits.py                 ← Stratified train/val/test splits
    ├── build_training_dataset.py       ← Merges master_label_review.csv → training_dataset.json
    ├── build_gold_set.py
    ├── generate_synthetic.py
    ├── generate_synthetic_labels.py
    ├── infer.py                        ← Stage 3 Ens-F inference API (CE + CORN ensemble)
    ├── eval_stage3.py                  ← Stage 3 evaluation with ablation flags + resume
    ├── quick_ablation.py
    ├── run_ensemble.py                 ← Probability-averaging across multiple runs
    ├── run_gemini_pro_review.py        ← Gemini-Pro tiebreaker pass
    ├── prepare_relabel_batch.py        ← Sonnet relabel batch preparation
    ├── relabel_soft_labels_v2.py
    ├── save_relabel_results.py
    ├── show_run.py                     ← Inspect a single training run's metrics
    ├── smoke_test_stage3.py            ← Trainer numerical-stability smoke test
    ├── smoke_test_stage3_agent.py      ← End-to-end agent smoke test
    └── run_pipeline.py                 ← CLI: contract → report
```

## Data Contracts (Shared Dataclasses)

All stages communicate through typed Python dataclasses defined in `src/common/schema.py`. These are the single source of truth for inter-stage data formats.

## Data Flow (Detailed)

### Stage 1+2 Input
```json
{
  "document_id": "contract_001",
  "contract_text": "Full contract text extracted from PDF...",
  "file_path": "data/raw/affiliate_agreement.pdf",
  "queries": [
    {
      "clause_type": "Indemnification",
      "question": "Highlight the parts (if any) of this contract related to \"Indemnification\" that should be reviewed by a lawyer. Details: Liability of one party to indemnify the other party (or parties)."
    },
    {
      "clause_type": "Termination For Convenience",
      "question": "Highlight the parts (if any) of this contract related to \"Termination For Convenience\" that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause ('at will')?"
    }
  ]
}
```

> There are 41 queries total, one per CUAD clause type. The question templates come directly from the CUAD dataset. Each question asks the model to locate a specific clause type. An empty answer means that clause type is absent from this contract.

### Stage 1+2 Output → Stage 3 Input
```json
[
  {
    "clause_id": "contract_001_Indemnification_0030",
    "document_id": "contract_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "start_pos": 4521,
    "end_pos": 4687,
    "confidence": 0.94
  }
]
```

> One object per detected clause. Clauses with confidence below threshold are excluded. Clause types with empty model answers (clause absent) are omitted.

### Stage 3 Output → Stage 4 Input
<!-- Updated 2026-05-13: removed `overridden` field (not in schema), aligned to current `RiskAssessedClause` shape -->
```json
[
  {
    "clause_id": "contract_001_Indemnification_0030",
    "document_id": "contract_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "risk_level": "HIGH",
    "risk_explanation": "One-sided indemnification covering counterparty negligence",
    "similar_clauses": [],
    "cross_references": [],
    "confidence": 0.88,
    "agent_trace": [
      {"tool": "precedent_search", "result_count": 5},
      {"tool": "contract_search", "result_count": 12}
    ]
  }
]
```

> `risk_explanation` is generated by the Qwen3-30B agent (see Stage 3 architecture section above). `agent_trace` is populated for every clause — every clause now routes through the LangGraph ReAct agent, so there is no longer a gated fast-path with empty traces. `similar_clauses` and `cross_references` are reserved fields in the schema (currently emitted as empty lists by `_agent_path`); retrieval results live inside `agent_trace` instead. `confidence` is DeBERTa's confidence in its pre-classification label. See the **Stage 3 Training Data Pipeline** section below for how training labels are produced and the **Stage 3 Architecture — DeBERTa + LangGraph ReAct Agent** section above for inference flow.

### Stage 4 Output (Final Report)
```json
{
  "document_id": "contract_001",
  "summary": "This vendor agreement contains 23 clauses across 14 categories...",
  "high_risk": [
    {
      "clause_id": "contract_001_Indemnification_0030",
      "clause_type": "Indemnification",
      "risk_level": "HIGH",
      "explanation": "The indemnification provision requires...",
      "recommendation": "Renegotiate to mutual indemnification..."
    }
  ],
  "medium_risk": [
    {
      "clause_id": "contract_001_Non_Compete_0009",
      "clause_type": "Non-Compete",
      "risk_level": "MEDIUM",
      "explanation": "Geographic scope is broad but time-limited...",
      "recommendation": "Consider narrowing geographic restriction..."
    }
  ],
  "low_risk_summary": "15 clauses were assessed as standard/low risk...",
  "missing_protections": ["Data Protection", "Force Majeure"],
  "overall_risk_score": 6.8,
  "total_clauses": 23,
  "metadata": {
    "generated_at": "2026-04-08T12:00:00Z",
    "models_used": {
      "extraction": "microsoft/deberta-base",
      "risk_classification": "models/stage3_risk_deberta",
      "explanation": "Qwen3-30B-Q4_K_XL (local llama-server)",
      "report_summary": "Qwen3-30B-Q4_K_XL (local llama-server)"
    }
  }
}
```

<!-- Updated 2026-05-13: new sections — API Layer + End-to-End Workflow -->

## API Layer (FastAPI)

A FastAPI service in `app/` exposes the pipeline over HTTP. Long-running stages run in
`asyncio.run_in_executor()` threads so they never block the event loop.

### Endpoints

| Method | Path | Body | Effect |
|---|---|---|---|
| GET  | `/health` | — | Liveness probe |
| GET  | `/api/v1/documents/` | — | Returns FAISS index metadata (vectors, dim, model name) |
| POST | `/api/v1/stage1/analyze` | `multipart/form-data` file | Full pipeline: Stage 1 → 3 → 4. Returns a `FinalRiskReport`. Typical 5–15 min per contract. |
| POST | `/api/v1/stage3/assess` | `Stage3Request` (list of `ClauseInput`) | Stage 3 only: takes pre-extracted clauses, returns list of `RiskAssessedClause` dicts. |
| POST | `/api/v1/stage4/report`  | `Stage4Request` (list of `AssessedClauseInput`) | Stage 4 only: takes assessed clauses, returns a `FinalRiskReport`. |

### Service-layer flow

```
app/routers/stage1_extract.py::analyze_contract
    └─ asyncio.run_in_executor →
        app/services/stage1_extract_svc.py::run_full_pipeline
            ├─ src.stage1_extract_classify.preprocessing.preprocess_contract
            ├─ src.stage1_extract_classify.model.ClauseExtractorClassifier.extract
            ├─ src.stage3_risk_agent.agent.assess_clauses    ← all clauses, agent path
            └─ src.stage4_report_gen.report_builder.build_report
```

The Stage-3 and Stage-4 endpoints call `assess_clauses()` and `build_report()` directly without going through `src/workflow/`. The DAG in `src/workflow/` is **not** on the request path (see next section).

### Pydantic vs. dataclass schemas

`app/schemas/domain.py` and `app/schemas/requests.py` define **Pydantic** request/response models for HTTP serialisation. `src/common/schema.py` defines **dataclasses** for in-process pipeline contracts. The two layers parallel each other; the service layer converts at the boundary (`_to_schema_clause` / `_to_internal`). Keep both in sync when adding fields.

### CORS

`app/main.py` sets `allow_origins=["*"]` for development. Tighten before any non-local deployment.

## End-to-End Workflow (`src/workflow/`) — experimental, not on request path

> **Status (2026-05-13): not wired in.** The FastAPI services and `scripts/run_pipeline.py` both call `assess_clauses()` and `build_report()` directly. The LangGraph DAG below was scaffolded in commit `7a88981` as an alternative orchestration but its Stage-3 node module (`src/stage3_risk_agent/nodes.py`) is absent on `main`, so the graph cannot be instantiated as-is. Treat as deferred / experimental.

### Intended shape

```
START
  └─ Node_A_Stage1_Extract       (src/stage1_extract_classify/nodes.py — also missing)
       └─ Node_C_Stage3_RiskClassify  (dispatcher, fan-out per clause)
            ├─ Node_D_Mistral_Router   (parallel workers, one per clause)  ← named "Mistral" historically; actually Qwen now
            └─ … (Map-Reduce fan-in)
                 └─ Node_E_Stage4_ReportGen  (src/stage4_report_gen/nodes.py — exists, uses print())
                      └─ END
```

State carrier is `src/workflow/state.py::RiskAnalysisState` (TypedDict). Note the import from `app.schemas.domain` — this couples the workflow package to the API layer, which is a smell to revisit before reviving the DAG.

### Recommendation
Either (a) restore the missing `nodes.py` files and add a smoke test, or (b) retire `src/workflow/` along with `src/stage4_report_gen/nodes.py` and the empty `src/stage4_report/` package. The current request path does not need this DAG.

## Key Models

| Model | Stage | HuggingFace ID | Purpose | VRAM |
|-------|-------|---------------|--------|------|
<!-- Updated 2026-05-13: Stage 1 default model now HF Hub; dropped stale Mistral references; reflect locked all-clause agent routing -->
| DeBERTa-base | 1+2 | HF Hub default: `rajnishahuja/cuad-stage1-deberta` (fine-tuned from `microsoft/deberta-base`) | QA extraction + classification | ~2 GB (train ~8 GB) |
| DeBERTa-v3-base | 3 | `microsoft/deberta-v3-base` fine-tuned. **Ens-F** = two-model ensemble: `rajnishahuja/cuad-risk-deberta-ce-parties` (CE loss, seed 42) + `rajnishahuja/cuad-risk-deberta-corn-parties` (CORN loss, seed 7), both with signing-party text in segment A. Inference API: `scripts/infer.py::RiskClassifier`. Chose v3 over base: ELECTRA-style pretraining → stronger downstream performance, same VRAM; SentencePiece 128k vocab is more efficient on legal text (p99 292 tokens vs 358 with base). | ~2 GB (train ~8 GB) |
| Qwen3-30B (Q4_K_XL) | 3 | Local llama-server (OpenAI-compatible, `http://localhost:10006/v1`). Swap `agent_base_url` + `agent_model` in `stage3_config.yaml` for any OpenAI-compatible endpoint (vLLM, Ollama, Azure-hosted, OpenAI, etc.) when deploying outside this server. Provider can also be switched via `llm_provider: gemini \| anthropic` (`src/common/utils.py::make_llm`). | LangGraph ReAct agent — runs on every clause; produces final label + explanation. | ~20 GB (4-bit, GPU) |
| all-MiniLM-L6-v2 | 3 | `sentence-transformers/all-MiniLM-L6-v2` | Clause embeddings for FAISS | ~0.5 GB |
| Qwen3-30B (Q4_K_XL) | 4 | Local llama-server port 10006 (shared with Stage 3) | Executive summary generation | ~20 GB (shared, no extra VRAM) |
| Qwen-30B (non-reasoning) | 3 (data prep, done) | `mavenir-generic1-30b-q4_k_xl` (local llama-server on A100, temp=0) | Primary labeler — 4,410 risk-relevant spans | ~20 GB (4-bit) |
| Gemini 2.5 Flash | 3 (data prep, done) | `gemini-2.5-flash` (Google API, JSON mode, temp=0) | Primary labeler — 4,410 risk-relevant spans (independent of Qwen) | API |
| Gemini 2.5 Pro | 3 (data prep, done) | `gemini-2.5-pro` (Google API) | Boundary-disagreement tiebreaker — 87 focus-type MEDIUM↔HIGH / LOW↔MEDIUM cases | API |
| spaCy en_core_web_sm | 1+2 | N/A (pip) | Baseline comparison | negligible |

## Key Datasets

| Dataset | Source | Format | Usage |
|---------|--------|--------|-------|
| CUAD QA | `theatticusproject/cuad-qa` (HuggingFace) | Pre-flattened QA rows; train: 22,450 / test: 4,182 | Stage 1+2 extraction + classification training |
| Stage 1+2 output (positive spans) | `data/processed/all_positive_spans.json` | JSON list, 6,702 spans across 510 contracts × 41 types | Source pool for Stage 3 risk labeling |
| Merged risk labels | `data/review/master_label_review.csv` | 6,702 rows, 6 categories — see **Stage 3 Training Data Pipeline** | Stage 3 risk classifier training labels |

**CUAD details:** 510 legal contracts, 41 clause types per contract, ~20,910 QA pairs total. Each QA pair has a question ("Highlight the parts related to X..."), the full contract text as context, and an answer span (or empty if clause absent). ~67.9% of QA pairs have empty answers. Of the 32.1% positive spans (6,702), 5 metadata types (2,292 spans) route to the Stage 4 report header — leaving **4,410 risk-relevant spans** for the classifier. License: CC BY 4.0. See `docs/dataset_insights.md` for the per-type breakdown.

**Note:** `theatticusproject/cuad-qa` is pre-flattened and pre-split — no flatten/split step needed. Do **not** use the raw `theatticusproject/cuad` (PDFs) or `kenlevine/CUAD` (nested SQuAD JSON) variants. See `src/common/data_loader.py`.

## Config File Schemas

### `configs/stage1_config.yaml`
<!-- Updated 2026-05-13: synced with current YAML. NOTE: live config has `dataset: kenlevine/CUAD` which is a regression — see Known Issues #5. -->
```yaml
# Stage 1+2: Clause Extraction & Classification
model_name: microsoft/deberta-base
max_seq_length: 512
doc_stride: 128
batch_size: 8
learning_rate: 2.0e-5
epochs: 3
output_dir: models/stage1_2_deberta
confidence_threshold: 0.01
dataset: theatticusproject/cuad-qa          # CANONICAL — see Known Issues #5 (live YAML has a regression)
fp16: true
```

### `configs/stage3_config.yaml`
<!-- Updated 2026-05-13: synced with locked training recipe; removed confidence_threshold / similarity_top_k_high_conf (all clauses route through agent); dropped Mistral; added llm_provider + agent_num_workers -->
```yaml
risk_classifier:
  # Model
  model_name: microsoft/deberta-v3-base
  output_dir: models/stage3_risk_deberta_v3       # versioned per experiment

  # Data
  training_data_path: data/processed/training_dataset.json
  splits_path:        data/processed/splits.json
  max_length: 512

  # Loss & signal (unified soft-target CE)
  class_weights_method: effective_counts          # Option B (locked) — Option A produced MEDIUM collapse
  soft_label_weighting: confidence_weighted

  # Fine-tuning strategy
  fine_tuning: full                                # all 86M params trainable
  llrd: true                                       # layer-wise LR decay (locked winning recipe)
  llrd_decay: 0.95                                 # embeddings ≈ 0.54× base_lr

  # Optimizer + schedule
  batch_size: 16
  learning_rate: 5.0e-5
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  epochs: 20
  weight_decay: 0.05

  # Early stopping
  early_stopping_patience: 5
  metric_for_best_model: val_macro_f1
  greater_is_better: true

  # Precision (bf16; DO NOT use fp16 — DeBERTa attention NaN)
  precision: bf16
  allow_fp32_fallback: true

  # Reproducibility
  seed: 42                                         # independent from splits seed=100
  strict_determinism: false
  dropout: 0.1                                     # DeBERTa default

# Stage 3 inference — agent + tools
embedding_model:  sentence-transformers/all-MiniLM-L6-v2
faiss_index_path: data/faiss_index/clauses.index

# Agent LLM — Qwen3-30B Q4_K_XL via local llama-server (OpenAI-compatible).
llm_provider:     openai_compatible              # options: openai_compatible | gemini | anthropic
agent_model:      Qwen3-30B-Q4_K_XL
agent_base_url:   http://localhost:10006/v1
agent_api_key:    none                           # llama-server needs a non-empty string

agent_max_tokens:        512                     # per LLM call
agent_max_iterations:    2                       # ablation showed >2 unused
agent_num_workers:       1                       # set to llama.cpp -np slot count for parallelism
similarity_top_k_low_conf: 10                    # k=10 gave +10% MEDIUM vote accuracy vs k=5

# Raw label passes — preserved for audit, not used at train time
raw_label_passes:
  qwen:                data/synthetic/synthetic_risk_labels_qwen.json
  gemini_flash:        data/synthetic/synthetic_risk_labels_gemini.json
  gemini_pro_focus_87: data/synthetic/synthetic_risk_labels_gemini_pro.json
```

### `configs/stage4_config.yaml`
<!-- Updated 2026-05-13: replaced FLAN-T5 schema with current Qwen-via-llama-server schema. FLAN-T5 was never wired up; explainer.py is dead scaffolding (see Known Issues #7). -->
```yaml
# Stage 4: Report Generation
# Executive summary LLM (reuses the same llama-server as Stage 3)
llm_provider:     openai_compatible              # options: openai_compatible | gemini | anthropic
agent_model:      qwen3-30b-q4_k_xl
agent_base_url:   http://localhost:10006/v1
agent_api_key:    none
agent_max_tokens: 256                            # executive summary is short

output_format: json
```

## Known Issues & Limitations

<!-- Updated 2026-05-13: added items 4–8 from architecture-sync audit -->

1. **Classification evaluation is misleading in existing code.** The current `evaluate.py` infers `pred_type = true_type if pred_text else "NO_CLAUSE"`. This means any non-empty answer is counted as correctly classifying the clause type (since the type is derived from the question, not the model). This inflates classification accuracy. The proper evaluation should measure whether the model correctly identifies clause presence vs absence per type.

2. **CUAD class imbalance.** ~67.9% of QA pairs have empty answers (clause absent). Models may learn to predict empty spans by default. Training should handle this imbalance (e.g., balanced sampling or adjusted loss). See `docs/dataset_insights.md` for per-clause-type positive rates.

3. **Long contracts exceed 512 tokens.** CUAD contracts average 10K-50K characters. The sliding window approach (`doc_stride=128`) handles this but means one clause may span multiple windows. Post-processing must deduplicate overlapping predictions.

4. **`requirements.txt` is incomplete.** Several runtime imports are not declared: `langchain-openai`, `langchain-google-genai`, `langchain-anthropic` (used by `src/common/utils.py::make_llm`), `fastapi`, `uvicorn`, `pydantic` (used by `app/`). Conversely, `bitsandbytes` is listed but unused — Mistral 4-bit quantization is no longer part of the architecture (Qwen3-30B runs in a separate llama-server process). A consolidated dependency sweep is pending.

5. **`configs/stage1_config.yaml` has a stale dataset key.** Live file contains `dataset: kenlevine/CUAD`, but the canonical dataset is `theatticusproject/cuad-qa` (used by `src/common/` loaders and asserted in CLAUDE.md). The `kenlevine/CUAD` variant is nested SQuAD JSON and is explicitly warned against. Fix the YAML; no code currently reads this key but stale config misleads readers.

6. **Tests are scaffolding only.** Every method under `tests/` raises `NotImplementedError`. The directory exists to satisfy the conventions block, not to provide coverage. Treat as TODO.

7. **`src/stage4_report_gen/explainer.py` is dead code.** All functions `raise NotImplementedError` and the file references FLAN-T5, which was never wired in. Stage 4's executive summary is produced by Qwen via `report_builder.py::_generate_summary`. Per-clause explanations come from the Stage 3 agent and pass through unchanged. The file should be removed in a future cleanup commit.

8. **`src/workflow/` is experimental and not on the live request path.** The FastAPI services and `scripts/run_pipeline.py` both call `assess_clauses()` and `build_report()` directly. The LangGraph DAG in `src/workflow/graph.py` imports `src/stage3_risk_agent/nodes.py`, which does not exist on `main` — so the DAG cannot be instantiated as-is. Either restore the missing node module or retire `src/workflow/`. See "End-to-End Workflow" section below.

## Conventions

- Python 3.10+
- Type hints on all function signatures
- Configs in YAML, loaded via `configs/` directory
- All model paths configurable (local or HuggingFace hub)
- Use `logging` module, not print statements
- Shared data contracts in `src/common/schema.py` — all stages import from here
- One consolidated `requirements.txt` (no per-stage requirement files)
- Docstrings on all public functions
- Tests mirror src structure in `tests/`
