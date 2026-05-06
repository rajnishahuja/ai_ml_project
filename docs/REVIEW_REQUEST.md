# Review Request — Legal Contract Risk Analyzer (Stage 3)

Please review the design, methodology, and experimental results for the Stage 3 risk
classification pipeline of a legal contract analysis system. The goal is to identify
alternative approaches, potential oversights, and improvements we may have missed after
working closely on this for several weeks.

---

## Project Context (2 min read)

**What it does:** Analyzes legal contracts (CUAD dataset — 510 contracts, 41 clause types)
and classifies each extracted clause as LOW / MEDIUM / HIGH risk from the perspective of
the signing party.

**4-stage pipeline:**
1. Extract clause spans (DeBERTa-v3 QA model on CUAD)
2. Classify clause types
3. **[This review] Assess risk level** — fine-tuned DeBERTa-v3-base + LangGraph ReAct agent with FAISS RAG
4. Generate risk report (Qwen 30B + lookup table)

**Current Stage 3 performance:** macro F1 = 0.607 (Ens-F: CE + CORN ensemble) on 452
hard-labeled test rows. This has been stable across 24 training runs.

---

## Key Files to Read

Read these in order — each builds on the previous:

### 1. Architecture (start here)
**`ARCHITECTURE.md`** (root) — end-to-end pipeline, Stage 3 agent design, data flow,
tool definitions, implementation notes. The Stage 3 section is the most relevant.

### 2. Labeling methodology
**`docs/STAGE3_LABELING.md`** — how the 4,375-row training dataset was built:
multi-LLM consensus (Qwen 30B + Gemini Flash + Claude Sonnet), party-metadata injection,
soft-label handling, and a recent label swap for human-reviewed rows.

### 3. Training decisions & findings
**`docs/STAGE3_TRAINING_NOTES.md`** — dataset shape, model choice rationale, split
strategy, loss function decision, class weights, hyperparameter choices, all with evidence.

### 4. Full experimental record
**`docs/STAGE3_EXPERIMENTS.md`** — every training run (Runs 1–24 + 14 ensembles),
what was tried, what failed, closed hypotheses, and the error analysis. The most
information-dense document.

### 5. Considered enhancements (not yet implemented)
**`docs/OPTIONAL_ENHANCEMENTS.md`** — OE-8 through OE-11, with rationale for deferring.

### 6. Core agent code (if you want implementation depth)
- **`src/stage3_risk_agent/agent.py`** — LangGraph ReAct loop, DeBERTa-as-default
  system prompt, synthesis call, checkpoint/resume logic
- **`src/stage3_risk_agent/tools.py`** — `precedent_search` (FAISS, min_sim=0.75)
  and `contract_search` (sibling clauses from same contract)
- **`configs/stage3_config.yaml`** — all hyperparameters in one place

---

## Current State Summary

**Training data:** 4,375 rows (4,276 hard labels, 99 soft probability vectors)

| Source | Rows | How labeled |
|---|---|---|
| AGREED | 2,726 | Qwen 30B + Gemini Flash both agreed (first pass) |
| SOFT_LABEL_V2_AGREED | 1,228 | 3-way vote (Qwen + Gemini + Sonnet) with party metadata |
| SONNET_REVIEW | 322 | Sonnet labels replacing inconsistent human/Gemini Pro labels |
| SOFT_LABEL | 99 | 3-way disagreement, kept as soft probability vectors |

**Classifier:** DeBERTa-v3-base, fine-tuned, 4,276 hard-label rows (train split: 3,398).
Two-model ensemble: CE loss (seed=42) + CORN loss (seed=7), both with signing-party
metadata in segment A.

**Agent:** LangGraph ReAct. DeBERTa runs first; its label + confidence injected into the
system prompt as the default. Agent verifies with tools and may override only when both
tools provide convergent consensus. Tools: FAISS precedent search (k=5, min_sim=0.75)
and contract_search (sibling clauses).

---

## Pain Points and Open Questions

### 1. F1 ceiling at 0.607 — cause uncertain

24 training runs (loss functions: CE, CORN, hybrid CE+EMD, SORD, EMD; regularization:
WD, dropout, LLRD; label modes: soft, hard_only, argmax_soft) all converge to 0.583–0.627.
The current hypothesis is a **data ceiling**: 8–10 of the 36 clause types have label
poverty (<5 examples of at least one class in the test split). But we haven't ruled out
that the ceiling is from correlated labeling bias — three LLMs trained on similar data
may agree on the same systematic errors.

**Question:** Given the experimental record, does this look like a data ceiling, a
labeling ceiling, or something architectural we've missed?

### 2. MEDIUM F1 is consistently the weakest class

Every run: MEDIUM is the hardest. Error analysis on Run 22: **81% of remaining errors
are MEDIUM boundary** (LOW→MEDIUM: 40, MEDIUM→LOW: 63, HIGH→MEDIUM: 26, MEDIUM→HIGH: 16).
We've traced this to label noise (inconsistent human reviewers, soft-label ambiguity).
We're mid-fix: Sonnet labels replaced 322 inconsistently-labeled rows; training is
running now.

**Question:** Is there a structural reason MEDIUM is hard beyond label noise? The CUAD
risk scale has no objective anchor for MEDIUM — it's defined by exclusion ("not clearly
HIGH, not clearly LOW"). Would a different label schema help?

### 3. FAISS retrieval is weak for HIGH clauses

At similarity ≥ 0.75 threshold:
- LOW: 90.1% precision, ~42% coverage
- MEDIUM: 67.8% precision
- HIGH: **25.8% precision**, only **27% coverage**

HIGH-risk clauses use adversarial/negotiated phrasing that doesn't cluster by semantic
similarity. The current embedding model (all-MiniLM-L6-v2) is general-purpose, not
legal-domain. OE-9 (BM25+FAISS RRF hybrid) and OE-10 (Jina Embeddings v3) are deferred.

**Question:** For HIGH-risk legal clause retrieval specifically, is semantic similarity
even the right retrieval signal? What would you recommend?

### 4. Agent value-add not empirically measured

The LangGraph ReAct agent runs on every clause. DeBERTa's label is the default; the
agent may override only when both `precedent_search` and `contract_search` provide
convergent evidence. When the LLM freely overrode DeBERTa in earlier ablations, accuracy
dropped to 47–58% vs DeBERTa's 59.3% baseline — so the consensus constraint was added.

We've never measured: **does the constrained agent actually improve on DeBERTa alone at
the clause level?** The old eval (0.810 F1) was on a different architecture. Current
architecture eval hasn't been run.

**Question:** Is this agent architecture well-designed for a 3-class ordinal problem, or
are we adding latency and complexity without measurable benefit? What's a better design?

### 5. Multi-LLM labeling methodology — potential correlated bias

The entire training dataset rests on consensus between Qwen 30B, Gemini Flash, and Claude
Sonnet. All three models were trained on similar web data, may share similar legal
reasoning patterns, and may agree on the same wrong answer for certain clause types.

We have no independent ground truth to validate against (CUAD doesn't provide risk labels;
those are synthetic). The only validation is that DeBERTa trained on these labels achieves
0.607 macro F1 on the held-out test set — but the test labels were generated the same way.

**Question:** Is circular validation (train labels → DeBERTa → evaluate on same-source
test labels) a fundamental problem here? How would you recommend validating label quality?

### 6. 99 unresolved soft-label rows

Three-way (Qwen + Gemini + Sonnet) complete disagreement on 99 rows. All are adjacent
gaps (LOW↔MEDIUM or MEDIUM↔HIGH, label_gap=1.0). Dominant pattern: 65/99 are Q=HIGH /
G=LOW / C=MEDIUM — a three-way ladder split.

Current treatment: kept as soft probability vectors [0.0, 0.5, 0.5] or [0.5, 0.5, 0.0]
in training. Most common clause types: Cap On Liability (12), Exclusivity (8), License
Grant (8).

**Question:** Is training on soft vectors from complete model disagreement helpful or
harmful? Would dropping these 99 rows entirely be better?

---

## What We're NOT Looking For

- Suggestions to use GPT-4 / larger models for classification — the constraint is a
  locally-hosted DeBERTa-v3-base for inference latency reasons.
- Suggestions to re-label from scratch — the labeling pipeline took several weeks.
- Stage 1/2 (clause extraction) improvements — out of scope for this review.

---

## Quick Reference — Key Numbers

| Metric | Value | Context |
|---|---|---|
| Best ensemble macro F1 | 0.607 | Ens-F (CE+CORN, parties metadata) on 452 hard test rows |
| MEDIUM F1 (best) | 0.544 | Run 22 single model; Ens-F ensemble = 0.512 |
| HIGH F1 (best) | 0.681 | Run 17 CORN, but MEDIUM collapsed to 0.074 |
| FAISS HIGH coverage | 27% | At min_similarity=0.75 |
| FAISS HIGH precision | 25.8% | At min_similarity=0.75 |
| LLM override accuracy | 47–58% | When LLM freely overrides DeBERTa (ablation) |
| DeBERTa standalone | 59.3% | Accuracy on same ablation set |
| Training rows | 4,375 | 4,276 hard + 99 soft |
| Clause types | 36 | 8–10 have label poverty in test split |
