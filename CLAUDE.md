# Legal Contract Risk Analyzer — Claude Context

## Project Overview
ML pipeline that analyzes legal contracts and flags risky clauses using the CUAD dataset.
4-stage pipeline: Extract clauses (DeBERTa) → Assess risk (Agent + RAG) → Generate report (FLAN-T5).

## Current State (as of 2026-04-17)

### Branch: `main`

### What's Done
- **Phase 0 foundation**: configs (T0.1-T0.3), schema.py (T0.9), utils.py (T0.6) — all complete
- **data_loader.py (T0.5)**: Implemented with `theatticusproject/cuad-qa` dataset (pre-flattened, pre-split)
  - `load_cuad_dataset()` — loads train (22,450) and test (4,182) from HuggingFace
  - `preprocess_for_qa()` — sliding window tokenization for DeBERTa (max_length=512, stride=128)
- **Stage 1/2 extraction**: `data/processed/all_positive_spans.json` — 6,702 positive clause spans across 510 contracts and 41 clause types (Stage 1/2 output, committed)
- **Stage 3 synthetic label pipeline** (prompt iteration phase complete):
  - `scripts/generate_synthetic_labels.py` — v1 prompt with perspective anchor, clause-type description injection, `risk_driver` + `risk_reason` schema, metadata filtering (Option B), and dedup (~40% API call reduction). Ready to run.
  - `scripts/build_gold_set.py` — deterministic 25-clause stratified gold set builder
  - `data/synthetic/gold_set.json` — 25-clause gold set (8 high-risk + 10 mixed + 4 edge + 3 random)
  - `data/reference/cuad_category_descriptions.csv` — Atticus official one-line descriptions for all 41 CUAD types (used by labeling prompt)

### Immediate Next Step (do this on GPU server)
**Stage 3 pilot run — synthetic label generation via Qwen**

The labeling script is ready. Run a test batch first, then the full pilot:

```bash
# 1. Verify the pipeline works (25 clauses, ~2 min)
python scripts/generate_synthetic_labels.py --n_samples 25

# 2. Inspect output
cat data/synthetic/synthetic_risk_labels.json | python3 -m json.tool | head -80

# 3. Full pilot run (~500 clauses, stratified)
python scripts/generate_synthetic_labels.py --n_samples 500
```

The script currently calls `claude-sonnet-4-20250514` (Anthropic API). To use Qwen instead,
update the `label_clause()` function in `scripts/generate_synthetic_labels.py` — specifically
the `client.messages.create(model=...)` call — to use your local Qwen endpoint or LiteLLM wrapper.

After the pilot run, audit ~100 clauses stratified by `(clause_type × risk_level)`.
See `docs/STAGE3_SYNTHETIC_LABELS_DISCUSSION.md` for the full audit checklist and three-phase rollout plan.

### What's Done (Stage 3 labeling — 2026-04-17/18)
- **Qwen/30B labels**: `data/synthetic/synthetic_risk_labels_qwen.json` — 4,410 rows, temp=0, GPU
- **Gemini 2.5 Flash labels**: `data/synthetic/synthetic_risk_labels_gemini.json` — 4,410 rows, temp=0, JSON mode
- **Copilot labels cleaned**: `data/cuad_risk_labels_copilot.csv` — 6,702 rows (removed 100 bad rows)
- **Master review file**: `data/review/master_label_review.csv` — all 6,702 spans with categories,
  disagreement analysis, reviewer assignments, row_num 1–6702
- **Analysis docs**: `docs/STAGE3_LABEL_ANALYSIS.md`, `docs/STAGE3_LABEL_COMPARISON.md`

### Immediate Next Steps
1. **Colleagues review 239 MANUAL_REVIEW rows** — filter `master_label_review.csv` by `reviewer` column,
   fill `final_label` (rows 5028–5266)
2. **Run Gemini 2.5 Pro on 87 GEMINI_PRO_REVIEW rows** — DONE (`scripts/run_gemini_pro_review.py`)
3. **Merge final labels** — join reviewed `final_label` back on `row_num`
4. **Build training dataset** — hard labels for AGREED/reviewed rows, soft labels for SOFT_LABEL rows
5. **Train DeBERTa risk classifier** — fine-tune on merged labels (GPU needed)
6. **Build FAISS index** — embed labeled clauses for Stage 3 RAG retrieval

### Stage 1 — still pending (GPU needed)
1. **Tokenize full training set** — run `preprocess_for_qa()` on all 22,450 examples
2. **Implement train.py (T1.2)** — extract training logic from pipeline.py
3. **Train DeBERTa** — fine-tune on CUAD QA task
4. **Run baseline evaluation** — benchmark spaCy/regex baseline on CUAD test set

### Key Decisions Made
- **Dataset**: Using `theatticusproject/cuad-qa` (HuggingFace). Pre-flattened QA rows, no flatten/split needed.
- **Metadata routing (Option B)**: 5 metadata CUAD types (Document Name, Parties, Agreement Date, Effective Date, Expiration Date) are NOT risk-labeled — they route to the Stage 4 report header instead. See `docs/STAGE3_SYNTHETIC_LABELS_DISCUSSION.md`.
- **Dedup**: Label once per unique (whitespace-normalized) clause text, fan label to duplicate rows. Saves ~40% API calls combined with metadata routing (6,702 → 3,974).
- **Labeling perspective**: Always from the signing party (counterparty to the drafter).
- **Anushka's code**: Her Stage 1+2 work is in `src/stage1_extract_classify/` using `CUAD_v1.json` locally. Our `data_loader.py` uses HuggingFace `cuad-qa` independently. Review suggestions in `docs/STAGE1_REVIEW_NOTES.md`.

## Working Style
- **Interactive and incremental**: Small pieces of code, explain each step, user runs and verifies before moving on.
- **Learning-focused**: The goal is to understand and implement, not just finish. Explain concepts before code.
- **Two machines**: Local laptop (no GPU) for development, separate server with GPU for training. Keep code portable.

## Architecture Reference
- `ARCHITECTURE.md` — full data flow, directory structure, model table
- `docs/TASK_LIST.md` — all tasks with status and dependencies
- `docs/STAGE1_REVIEW_NOTES.md` — alignment suggestions for Anushka's code
- `configs/stage1_config.yaml` — DeBERTa training hyperparameters
