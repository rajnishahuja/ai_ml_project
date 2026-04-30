# Experiment Notes — Rajnish

## Goal
Train DeBERTa on CUAD with improvements over the baseline Stage 1 code,
without modifying `src/stage1_extract_classify/`.

---

## Baseline (Stage 1 original — pipeline.py)
- **Model**: `microsoft/deberta-base` (184M params)
- **Tokenizer**: deberta-base (BPE, ~50k vocab)
- **Max length**: 384 tokens, stride 128
- **Split**: Random QA-level split (has data leakage — same contract in train/val/test)
- **Precision**: FP32 (`fp16=None` in pipeline.py)
- **Status**: Training confirmed working (loss 5.56 → 0.94 in first 800 steps, 1 epoch partial)

## Planned improvements (wrapper script)

1. **Contract-level split** (GroupShuffleSplit) — prevent data leakage where same contract appears in train/val/test. Already implemented in preprocess.
2. **512 token window** — DeBERTa supports it, captures more context per window than 384. Already implemented as a parameter.
3. **DeBERTa v3-base** — disentangled attention v2, better pretrained. Was the plan, but hit NaN issues.
4. **BF16 training** — faster, less VRAM than FP32. Needs the overflow patch to work with any DeBERTa variant.
5. **Dynamic warmup** — 10% of total steps instead of hardcoded 2300. Already implemented.
6. **GPU check** — refuse to train without CUDA. Already implemented.
7. **Early stopping** — patience=3 vs baseline patience=2. Already implemented.

---

## Known issues

### DeBERTa FP16/BF16 overflow (ALL variants)
In transformers 4.57.0, `modeling_deberta.py` and `modeling_deberta_v2.py` use
`torch.finfo(dtype).min` for attention masking. This value overflows in FP16 and BF16,
producing NaN loss. Affects deberta-base, deberta-v3-base, and all DeBERTa variants.

- Baseline avoids this by accident (`fp16=None` → FP32)
- Monkey-patch approach (replace finfo.min with -10000) was implemented but not verified
- **Workaround**: Train in FP32. Costs ~2x VRAM, ~30% slower.

### NaN loss with deberta-v3-base — not fully diagnosed
Attempted training with deberta-v3-base + tokenized_cuad_512 (correct v3 tokenizer).
Got NaN loss in all configurations tested:
- bf16=True + monkey-patch → NaN
- bf16=False (FP32) → NaN
- Manual forward pass outside Trainer → loss=6.95, grads fine (no NaN)

Root cause unclear. Possibly related to `processing_class=tokenizer` in Trainer
interacting differently with v3 tokenizer, or the monkey-patch itself. Changed to
`DefaultDataCollator()` but never tested before switching to baseline verification.

**Decision**: Park v3 experiments. Focus on getting v1 (deberta-base) working with
our improvements (contract-level split, 512 window) first.

---

## Experiment results

### Run 1: train_wrapper.py — deberta-base + 512 + contract-level split + FP32
- **Date**: 2026-04-14
- **Script**: `notebooks/train_wrapper.py`
- **Config**: deberta-base, 512 tokens, contract-level split, FP32, batch=8, grad_accum=2, lr=2e-5
- **Data**: `tokenized_cuad_512/` (train: 32,949, val: 4,380, test: 64,977)
- **Improvements over baseline**: #1 (contract split), #2 (512 window), #5 (dynamic warmup=205), #6 (GPU check), #7 (patience=3)
- **Results (1 epoch)**:

| Epoch | Eval Loss |
|-------|-----------|
| 0.10  | 1.361     |
| 0.87  | 0.483     |
| 0.97  | 0.478     |
| 1.00  | 0.476     |

- **Train loss (avg)**: 0.111
- **Status**: Completed. Eval loss plateauing at ~0.476. Model saved to `models/rajnish_deberta/`.
- **Note**: SSH session died mid-training; resumed from checkpoint-1600 using `--resume auto`.

---

## Datasets on disk

| Path | Model | Window | Split | Tokenizer |
|------|-------|--------|-------|-----------|
| `tokenized_cuad_384/` | deberta-base | 384 | Random QA-level | deberta-base (BPE) |
| `tokenized_cuad_512/` | deberta-base | 512 | Contract-level | deberta-base (BPE) |

## Scripts

| Script | Purpose |
|--------|---------|
| `notebooks/train_wrapper.py` | New wrapper with safe improvements (#1,#2,#5,#6,#7) |
| `notebooks/run_experiment.py` | Retired — had v3/BF16 experiments that hit NaN issues |

## Next steps
1. Test BF16 training with deberta-base (single-variable change from working Run 1)
2. If BF16 works, test deberta-v3-base in FP32 (isolate v3 as a variable)
3. Run full 3-epoch training with best config
4. Build evaluation script (EM/F1 on test set)

---

*Updated 2026-04-15*
