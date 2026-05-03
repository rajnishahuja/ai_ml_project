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
│  Stage 3: Risk Detection    │  Hybrid Confidence-Gated
│  DeBERTa → Agent (low-conf) │  DeBERTa-v3-base (risk classifier)
│  Tools: precedent_search,   │  Qwen3-30B Q4_K_XL (agent + explanations)
│         contract_search     │  all-MiniLM-L6-v2 (embeddings)
│  Output: risk-assessed      │  FAISS vector store
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

## Stage 3 Architecture — Hybrid Confidence-Gated (Decided 2026-04-23)

> **Decision**: Hybrid Confidence-Gated architecture. DeBERTa classifies every clause; high-confidence predictions ship directly through a single explanation LLM call; low-confidence predictions escalate to a reasoning agent with tool access that can override DeBERTa's label.
>
> **Project goal satisfied**: exercises agentic RAG on the escalated path (~30–40% of clauses expected), while keeping the easy cases cheap and deterministic.

### Why This Choice

- **Practical efficiency** — the majority of clauses are expected to be high-confidence; no need to pay agent latency/tokens for easy cases
- **Exercises agentic RAG** — low-confidence path is a full tool-calling loop (precedent_search + contract_search), satisfies the learning goal
- **Addresses known label ambiguity** — signing-party direction was the #1 driver in manual review (81% of flips). The low-confidence escalation is exactly where METADATA + tools resolve this
- **Clean academic framing** — "specialist when confident, generalist when uncertain"
- **Cost-aware** — agent loop runs only on a minority of clauses

### High-Level Flow

```
Clause + METADATA  (from Stage 1+2)
     │
     ▼
┌─────────────────────────────┐
│  DeBERTa Risk Classifier    │
│  input: clause_type + text  │
│  output: (label, conf)      │
└──────────┬──────────────────┘
           │
           ▼
      confidence ≥ 0.6 ?
     ┌──────┴──────┐
   YES             NO
     │             │
     ▼             ▼
 High-Conf     Low-Conf
  Path          Path
(1 Qwen call, (1 Qwen call +
 no tools)     ReAct tool loop —
               agent may override)
     │             │
     └──────┬──────┘
            ▼
     {label, explanation,
      similar_clauses,
      agent_trace}  → Stage 4
```

### Low-Level Design

One Qwen3-30B instance serves both paths — it's the same model, operated in two modes depending on DeBERTa's confidence. Accessed via local llama-server OpenAI-compatible API (`agent_base_url` in `stage3_config.yaml`); swap to any compatible endpoint for deployment.

#### High-Confidence Path (conf ≥ 0.6)

DeBERTa's label is final. Mistral runs once with no tools, just to generate the explanation.

```python
def high_confidence_path(clause, metadata, deberta_label, deberta_confidence):
    # Optional: retrieve a couple of precedents to enrich explanation quality
    neighbors = precedent_search(clause.text, k=3)

    prompt = f"""
    Clause: {clause.text}
    Type: {clause.type}
    METADATA: {metadata}
    Risk Level: {deberta_label}  (classifier confidence: {deberta_confidence:.2f})
    Similar labeled precedents: {format_neighbors(neighbors)}

    Write a one-sentence risk explanation grounded in the clause and precedents.
    """
    explanation = llm(prompt)   # single forward pass, no tool loop

    return {
        "label": deberta_label,
        "confidence": deberta_confidence,
        "explanation": explanation,
        "similar_clauses": neighbors,
        "agent_trace": [],        # no tool calls made
        "overridden": False,
    }
```

Properties: deterministic, ~1 LLM call, no agent overhead.

#### Low-Confidence Path (conf < 0.6)

Qwen3-30B is invoked as a LangGraph ReAct agent with tool access. It reasons, fetches evidence, and produces a final label (possibly overriding DeBERTa) + explanation in a single structured output.

```python
def low_confidence_path(clause, metadata, deberta_label, deberta_confidence):
    system_prompt = """
    You are a legal risk assessor. DeBERTa produced an uncertain preliminary label.
    Use the available tools to gather evidence (similar labeled clauses, other
    clauses from the same contract), then output JSON:
      {
        "final_label": "LOW" | "MEDIUM" | "HIGH",
        "explanation": "...",
        "override_reason": "..."    # if disagreeing with DeBERTa
      }

    Tools available:
      - precedent_search(clause_text, k=5)
          → top-K similar labeled clauses across the corpus (vector RAG)
      - contract_search(document_id)
          → all typed clauses from the same contract (structured lookup)
    """

    context = {
        "clause": clause.text,
        "clause_type": clause.type,
        "metadata": metadata,
        "deberta_preliminary": {
            "label": deberta_label,
            "confidence": deberta_confidence,
        },
    }

    # LangGraph runs the tool-calling loop. Qwen3-30B does all reasoning.
    result = agent.invoke(
        system=system_prompt,
        context=context,
        tools=[precedent_search, contract_search],
        max_iterations=5,
    )

    return {
        "label": result.final_label,
        "confidence": None,                          # agent-derived, no calibrated score
        "explanation": result.explanation,
        "similar_clauses": result.retrieved_precedents,
        "agent_trace": result.tool_calls,
        "overridden": result.final_label != deberta_label,
    }
```

Properties: non-deterministic, multi-turn, typically 2–5 LLM calls, able to override DeBERTa.

### Tool Definitions

The two retrieval tools solve different problems:

- **`precedent_search`** (vector RAG) — FAISS similarity lookup over the labeled corpus (~4,276 non-metadata clauses, skipping 99 None-label rows). Embeddings: `all-MiniLM-L6-v2`. Returns top-K clauses with `{clause_text, clause_type, risk_level, similarity}`. Fast (exact cosine, microseconds).

- **`contract_search`** (structured lookup, **not** RAG) — given a `document_id`, returns all typed clauses already extracted by Stage 1+2 for that contract. No embeddings, no navigation, no LLM calls inside the tool. A contract averages ~9 non-metadata clauses (max 27), so all fit easily in the agent's context. Purpose: resolve same-contract cross-references (e.g., "IP was already assigned in another clause").

### Worked Example — Low-Confidence Path

Clause arrives:
```
clause_type: "IP Ownership Assignment"
clause_text: "Consultant hereby assigns to Company all right, title,
              and interest in any deliverables..."
metadata:    {Parties: ["AT&T Inc.", "Jane Smith Consulting"]}
DeBERTa:     (HIGH, 0.45)   ← below threshold, escalate
```

Agent loop trace:

```
Turn 1  Qwen reads clause + DeBERTa signal
        → "I need to see how similar clauses were labeled."
        → calls precedent_search("Consultant hereby assigns...", k=5)

        Tool returns:
          N1: LOW  — "signing party is the recipient of rights"
          N2: HIGH — "one-party-committed IP assignment to a vendor"
          N3: LOW  — "standard admin paperwork, IP already owned upstream"
          N4: LOW  — "mutual carve-out"
          N5: MEDIUM — "conditional on acquisition"

Turn 2  Qwen reads neighbors
        → "4 of 5 are LOW. Key driver is who signed.
           METADATA shows AT&T and Jane Smith Consulting.
           If AT&T is the buyer/client, this is standard admin paperwork.
           Let me verify by inspecting the full contract."
        → calls contract_search(document_id="contract_042")

        Tool returns 38 sibling clauses:
          - Compensation: "AT&T shall pay Consultant $50K..."
          - Services: "Consultant shall provide advisory services to AT&T..."
          - [...36 more...]

Turn 3  Qwen reads contract
        → "Confirmed. AT&T is the client paying Consultant for services.
           Consultant transferring IP to AT&T is standard work-for-hire.
           AT&T (signing party of interest) RECEIVES rights — not at risk.
           Overriding DeBERTa HIGH → LOW."
        → returns JSON:
          {
            "final_label": "LOW",
            "explanation": "Standard work-for-hire IP assignment from
                            Consultant to AT&T. AT&T receives rights;
                            no exposure to signing party.",
            "override_reason": "DeBERTa could not resolve signing-party
                                direction from clause text alone;
                                precedents + contract context confirm LOW."
          }

Stop.
```

Final output merges into the Stage 3 → Stage 4 schema (`risk_level`, `risk_explanation`, `similar_clauses`, `agent_trace`, `overridden`).

### Implementation Notes (added 2026-05-01)

**Structured output workaround** — `langchain-openai ≥ 1.2.0` defaults `with_structured_output()` to `method="json_schema"`, which calls OpenAI's Structured Outputs endpoint (`.parse()`). llama.cpp rejects this with HTTP 400 ("Failed to initialize samplers"). Fix: always pass `method="function_calling"` explicitly. Both `_fast_path` and `_agent_path` in `agent.py` do this.

**Agent synthesis call** — `create_react_agent` is called without `response_format` (which would trigger LangGraph's internal `generate_structured_response` node using the broken default method). Instead, after the ReAct loop completes, a single explicit `with_structured_output(RiskAssessment, method="function_calling")` call is made with a clean synthesis prompt built from the agent's final text message. This avoids the model ignoring the tool when the full ReAct history (with existing `tool_call` messages) is passed.

**FAISS index** — `data/faiss_index/clauses.index` (4,276 vectors, IndexFlatIP) + `data/faiss_index/clauses.json` (parallel metadata). Built from `training_dataset.json`, skipping 99 None-label rows. Entry point: `scripts/build_faiss_index.py`.

### Key Design Decisions (v1)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | Hybrid Confidence-Gated | practical + exercises agentic RAG |
| Confidence threshold | 0.6 | tunable post-training on validation set |
| DeBERTa input | `clause_type + clause_text` only | baseline — measure signing-party ceiling first |
| Party role tagging in DeBERTa | **Not in v1** | deferred; revisit if baseline is ceilinged |
| RAG consumer | LLM (agent + explainer), **not** DeBERTa | keeps DeBERTa training simple; RAG still load-bearing at escalation |
| Explanation generator | Qwen3-30B Q4_K_XL (same instance as agent) | one LLM, two modes based on confidence |
| Same-contract retrieval | `contract_search` structured lookup | data is already typed post-Stage 1+2 — no RAG needed |
| Precedent retrieval | FAISS (vector RAG) over 4,410 clauses | standard for flat-corpus similarity |
| Low-conf label source | Agent (may override DeBERTa) | needed to resolve signing-party ambiguity |

### Alternative Architectures Considered (Rejected)

- **Static LangGraph Pipeline** (former Option A) — DeBERTa's label always final, fixed tool sequence. No dynamic reasoning; doesn't satisfy agentic-RAG learning goal. A full implementation of this design existed in `src/workflow/` (committed by a colleague) using a separate state schema (`app.schemas.domain`) and service layer (`app.services`). It was deleted because it conflicted with the chosen Hybrid Confidence-Gated architecture and introduced a parallel, incompatible schema track. The canonical pipeline entry point is `scripts/run_pipeline.py`; the canonical Stage 3 entry point is `src/stage3_risk_agent/agent.py`.
- **Full Agent on Every Clause** (former Option B) — agent loop runs on all ~4,410 risk-relevant clauses. Over-engineered; majority of clauses don't need the capacity and agent latency dominates cost.
- **Multi-Signal Verifier** — DeBERTa + Mistral label independently, reconcile on disagreement. Similar capability to Hybrid but runs expensive path even on easy cases.
- **RAC (Retrieval-Augmented Classification)** — retrieved neighbors become part of DeBERTa's training input. Deferred to v2; brittle to retrieval quality, risk of label leakage, harder training pipeline.
- **Pure Reasoning Model (no DeBERTa)** — abandons ML learning goal; synthetic labels are already LLM-distilled signal that a fine-tuned DeBERTa absorbs more cheaply at inference.

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

**Option 2 — Add party role tag at training + inference time**
`[CLS] signing_party_role=licensor [SEP] clause_type [SEP] clause_text [SEP]`
- Stage 1+2 extracts METADATA (Parties field) and infers role (licensor/licensee, vendor/customer)
- DeBERTa trained with role tag → directly resolves signing party ambiguity
- Requires role inference logic from contract METADATA
- **Deferred**: revisit only if v1 accuracy is clearly ceilinged by signing-party ambiguity and the escalation path proves insufficient.

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
> Full labeling history + disagreement analysis: `docs/STAGE3_LABEL_COMPARISON.md`.

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

**Class weights** (`weight` tensor above) compensate for mild imbalance in the hard-label
distribution (LOW 44.5% / MEDIUM 32.3% / HIGH 23.1%):

```
weight_c = N / (K · count_c),  N = 3,049 hard rows, K = 3 classes
→ LOW 0.749, MEDIUM 1.030, HIGH 1.442
```

Computed from hard-row counts only — soft rows contribute uncertainty directly through the
loss and don't add bias that needs compensating. (Option B — include soft-label probability
mass in the counts — available as a one-line ablation if HIGH recall is weak in v1; see
`docs/STAGE3_TRAINING_NOTES.md` §7.)

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

```
AIML_project/
├── ARCHITECTURE.md          ← THIS FILE (AI context reference)
├── README.md                ← Project readme (create when ready)
├── requirements.txt         ← Python dependencies
├── .github/
│   └── copilot-instructions.md  ← Auto-loaded by GitHub Copilot
├── configs/
│   ├── stage1_config.yaml   ← Hyperparams for extraction/classification
│   ├── stage3_config.yaml   ← Agent config, RAG params, model paths
│   └── stage4_config.yaml   ← Report template config
├── docs/
│   ├── LegalAgents.docx                              ← Original project proposal
│   ├── Legal_Contract_Risk_Analyzer_Research_Analysis.docx  ← Research analysis v1
│   └── Legal_Contract_Risk_Analyzer_Project_Plan_v2.docx    ← Finalized plan v2
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   ├── schema.py             ← Shared dataclasses (ClauseObject, RiskAssessedClause, RiskReport)
│   │   ├── preprocessing.py     ← PDF/DOCX text extraction, cleaning
│   │   ├── data_loader.py       ← CUAD dataset loading and formatting
│   │   └── utils.py             ← Shared utilities (config loader, metrics, logging)
│   ├── stage1_extract_classify/
│   │   ├── __init__.py
│   │   ├── model.py             ← DeBERTa QA model wrapper
│   │   ├── train.py             ← Fine-tuning script
│   │   ├── predict.py           ← Inference: contract → clause objects
│   │   ├── baseline.py          ← spaCy + regex baseline
│   │   └── evaluate.py          ← Extraction + classification metrics
│   ├── stage3_risk_agent/
│   │   ├── __init__.py
│   │   ├── agent.py             ← LangGraph agent definition
│   │   ├── tools.py             ← RAG retrieval tool + contract search tool
│   │   ├── risk_classifier.py   ← DeBERTa risk classifier
│   │   ├── embeddings.py        ← FAISS index builder + query
│   │   ├── synthetic_labels.py  ← LLM-based risk label generation
│   │   └── evaluate.py          ← Risk detection metrics + ablation
│   └── stage4_report_gen/
│       ├── __init__.py
│       ├── aggregator.py        ← Deterministic grouping + risk score (✅ done)
│       ├── recommender.py       ← Lookup table: (clause_type, risk_level) → recommendation (✅ done)
│       ├── report_builder.py    ← Assembles RiskReport + Qwen executive summary (✅ done)
│       └── evaluate.py          ← ROUGE + optional human eval
├── data/
│   ├── raw/                     ← Downloaded CUAD dataset files
│   ├── processed/               ← Preprocessed data; `all_positive_spans.json` (6,702 spans)
│   ├── synthetic/               ← Raw LLM labeler outputs (Qwen, Gemini Flash, Gemini Pro)
│   ├── review/                  ← `master_label_review.csv` (canonical merged labels, 6,702 rows)
│   └── faiss_index/             ← Built FAISS vector index
├── notebooks/
│   ├── 01_cuad_exploration.ipynb      ← Dataset exploration and stats
│   ├── 02_synthetic_labeling.ipynb    ← Risk label generation + validation
│   ├── 03_stage1_training.ipynb       ← Extraction/classification training
│   ├── 04_stage3_agent_dev.ipynb      ← Agent development and testing
│   └── 05_evaluation.ipynb            ← End-to-end eval and ablation
├── tests/
│   ├── test_preprocessing.py
│   ├── test_stage1.py
│   ├── test_stage3.py
│   └── test_stage4.py
└── scripts/
    ├── download_cuad.py         ← Download CUAD from HuggingFace
    ├── generate_synthetic.py    ← Batch synthetic label generation
    ├── build_faiss_index.py     ← Build FAISS index from labeled clauses
    └── run_pipeline.py          ← End-to-end pipeline runner
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
```json
[
  {
    "clause_id": "contract_001_Indemnification_0030",
    "document_id": "contract_001",
    "clause_text": "Contractor shall indemnify Company against all claims...",
    "clause_type": "Indemnification",
    "risk_level": "HIGH",
    "risk_explanation": "One-sided indemnification covering counterparty negligence",
    "similar_clauses": [
      {"text": "...", "risk_level": "HIGH", "similarity": 0.92},
      {"text": "...", "risk_level": "LOW", "similarity": 0.87}
    ],
    "cross_references": ["contract_001_Cap_On_Liability_0034"],
    "confidence": 0.88,
    "overridden": true,
    "agent_trace": [
      {"tool": "precedent_search", "result_count": 5},
      {"tool": "contract_search", "related_clauses": 2}
    ]
  }
]
```

> `risk_explanation` is generated by Mistral-7B-Instruct. `agent_trace` records which tools the LangGraph agent invoked for this clause (populated only on the low-confidence path; high-confidence path emits `[]`). `overridden` is `true` when the agent's `final_label` differs from DeBERTa's preliminary label. See the **Stage 3 Training Data Pipeline** section below for how these labels are produced and the **Stage 3 Architecture — Hybrid Confidence-Gated** section above for inference flow.

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

## Key Models

| Model | Stage | HuggingFace ID | Purpose | VRAM |
|-------|-------|---------------|--------|------|
| DeBERTa-base | 1+2 | `microsoft/deberta-base` | QA extraction + classification | ~2 GB (train ~8 GB) |
| DeBERTa-v3-base | 3 | `microsoft/deberta-v3-base` | Risk classification (fine-tuned on merged labels from `master_label_review.csv` — 3,048 hard + 1,327 soft). Chose v3 over base: ELECTRA-style pretraining → stronger downstream performance, same VRAM; SentencePiece 128k vocab is more efficient on legal text (p99 292 tokens vs 358 with base). | ~2 GB (train ~8 GB) |
| Qwen3-30B (Q4_K_XL) | 3 | Local llama-server (OpenAI-compatible, `http://localhost:10006/v1`). Swap `agent_base_url` + `agent_model` in `stage3_config.yaml` for any OpenAI-compatible endpoint (Mistral-7B-Instruct, Azure-hosted model, etc.) when deploying outside this server. | Risk explanation (high-conf path) + ReAct tool-calling agent (low-conf path) | ~20 GB (4-bit, GPU) |
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
```yaml
model_name: microsoft/deberta-base
max_seq_length: 512
doc_stride: 128
batch_size: 8
learning_rate: 2.0e-5
epochs: 3
output_dir: models/stage1_2_deberta
confidence_threshold: 0.01
dataset: theatticusproject/cuad-qa
fp16: true
```

### `configs/stage3_config.yaml`
```yaml
risk_classifier:
  # Model
  model_name: microsoft/deberta-v3-base
  output_dir: models/stage3_risk_deberta_v3

  # Data
  training_data_path: data/processed/training_dataset.json
  splits_path: data/processed/splits.json
  max_length: 512

  # Loss & signal (unified soft-target CE)
  class_weights_method: hard_counts          # LOW 0.749 / MED 1.030 / HIGH 1.442
  soft_label_weighting: confidence_weighted

  # Fine-tuning strategy
  fine_tuning: full                           # all 86M params trainable
  llrd: false                                 # layer-wise LR decay — deferred to v1.1

  # Optimizer + schedule
  batch_size: 16
  learning_rate: 2.0e-5
  warmup_ratio: 0.1                           # linear warmup
  lr_scheduler_type: linear                   # linear decay after warmup
  epochs: 5
  weight_decay: 0.01

  # Early stopping
  early_stopping_patience: 2
  metric_for_best_model: val_macro_f1

  # Precision
  # Target bf16; fall back to fp32 if smoke test NaNs.
  # DO NOT use fp16 — DeBERTa attention + torch.finfo(fp16).min → NaN softmax.
  precision: bf16
  allow_fp32_fallback: true

  # Reproducibility
  seed: 42                                    # independent from splits seed=100
  strict_determinism: false

# Stage 3 inference — agent + tools
embedding_model: sentence-transformers/all-MiniLM-L6-v2
faiss_index_path: data/faiss_index/clauses.index
agent_model: mistralai/Mistral-7B-Instruct-v0.3
quantization: 4bit
confidence_threshold: 0.6                     # DeBERTa confidence gate
agent_max_iterations: 5                       # low-conf path tool-calling loop cap
similarity_top_k_high_conf: 3
similarity_top_k_low_conf: 5

# Raw label passes — preserved for audit, not used at train time
raw_label_passes:
  qwen:              data/synthetic/synthetic_risk_labels_qwen.json
  gemini_flash:      data/synthetic/synthetic_risk_labels_gemini.json
  gemini_pro_focus_87: data/synthetic/synthetic_risk_labels_gemini_pro.json
```

### `configs/stage4_config.yaml`
```yaml
explanation_model: google/flan-t5-base
risk_thresholds:
  high: 0.75
  medium: 0.40
output_format: json
max_explanation_length: 200
```

## Known Issues & Limitations

1. **Classification evaluation is misleading in existing code.** The current `evaluate.py` infers `pred_type = true_type if pred_text else "NO_CLAUSE"`. This means any non-empty answer is counted as correctly classifying the clause type (since the type is derived from the question, not the model). This inflates classification accuracy. The proper evaluation should measure whether the model correctly identifies clause presence vs absence per type.

2. **CUAD class imbalance.** ~67.9% of QA pairs have empty answers (clause absent). Models may learn to predict empty spans by default. Training should handle this imbalance (e.g., balanced sampling or adjusted loss). See `docs/dataset_insights.md` for per-clause-type positive rates.

3. **Long contracts exceed 512 tokens.** CUAD contracts average 10K-50K characters. The sliding window approach (`doc_stride=128`) handles this but means one clause may span multiple windows. Post-processing must deduplicate overlapping predictions.

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
