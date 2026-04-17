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

### What's Next (after pilot)
1. **Audit pilot labels** — stratified sample ~100 clauses, check `risk_driver` specificity and label correctness
2. **Full labeling run** — ~3,974 API calls (after metadata filter + dedup)
3. **Compare with Copilot labels** — `data/CUAD_clause_risk_dataset_copilot.csv` has colleague's labels; join key is `(Document Name - .pdf extension, Clause Type)` → matches our `(contract, clause_type)`
4. **Train DeBERTa (Stage 1)** — still needs GPU; see Stage 1 notes below
5. **Build FAISS index** — embed labeled clauses for Stage 3 RAG retrieval

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
