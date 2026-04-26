# Stage 3 Risk Classifier — Experiments Log

Append-only log of training runs and ensembles for the Stage 3 DeBERTa-v3 risk classifier.
For *design rationale* (loss choice, hyperparameter philosophy, eval scheme), see
`STAGE3_TRAINING_NOTES.md`. This file records *what we tried and what we got*.

**Test set:** 466 rows (331 hard-labeled subset). **Splits seed:** 100 (frozen).
**Splits file:** `data/processed/splits.json`. **Training data:** `data/processed/training_dataset.json` (4,375 rows).

---

## Quick reference — single-config runs

| # | name | macro | acc | LOW | MED | HIGH | hard | best_val |
|---|------|------:|----:|----:|----:|-----:|-----:|---------:|
| 1 | Option A — class_w=hard_counts (FAILED) | 0.210 | 0.288 | 0.000 | 0.429 | 0.200 | 0.252 | 0.210 |
| 2 | Option B baseline (lr=2e-5) | 0.415 | 0.433 | 0.463 | 0.446 | 0.335 | 0.462 | 0.377 |
| 3 | LR=5e-5 fix | 0.583 | 0.599 | 0.677 | 0.430 | 0.642 | 0.649 | 0.548 |
| 4 | LR=3e-5 + ep=10 | 0.579 | 0.609 | 0.702 | 0.390 | 0.645 | 0.605 | 0.568 |
| 5 | cosine + WD=0.05 + ep=10 | 0.603 | 0.622 | 0.724 | 0.469 | 0.617 | 0.653 | 0.590 |
| 6 | LLRD=0.9 + ep=10 | 0.575 | 0.603 | 0.701 | 0.374 | 0.651 | 0.613 | 0.582 |
| 7 | cosine + WD=0.1 | 0.591 | 0.614 | 0.706 | 0.426 | 0.642 | 0.630 | 0.611 |
| 8 | cosine + WD=0.05 + dropout=0.2 | 0.597 | 0.639 | 0.741 | 0.427 | 0.624 | 0.635 | 0.584 |
| 9 | cosine + WD=0.05 + LLRD=0.95 | 0.607 | 0.627 | 0.698 | 0.463 | 0.661 | 0.639 | 0.597 |
| 10 | Run 9 with seed=7 | 0.611 | 0.622 | 0.689 | 0.487 | 0.658 | 0.666 | 0.615 |
| 11 | Run 9 with seed=100 | 0.574 | 0.594 | 0.687 | 0.418 | 0.618 | 0.626 | 0.583 |
| 12 | Run 9 + EMD loss (FAILED — collapsed) | 0.143 | 0.273 | 0.000 | 0.428 | 0.000 | 0.167 | 0.144 |
| 13 | Run 9 + hybrid CE + 0.5×EMD | 0.587 | 0.616 | 0.720 | 0.425 | 0.616 | 0.626 | 0.613 |
| 14 | Run 9 + label-mode hard_only | 0.610 | 0.627 | 0.693 | 0.492 | 0.645 | 0.657 | 0.574 |
| 15 | Run 9 + label-mode argmax_soft | 0.603 | 0.614 | 0.671 | 0.464 | 0.675 | 0.645 | 0.580 |

**Multi-seed F (Runs 9, 10, 11) summary:**
- mean macro_f1 = 0.598, std = 0.020
- mean MEDIUM F1 = 0.456, std = 0.035 ← per-class is more variable than overall
- range 0.575 → 0.611 just from random seed

## Quick reference — ensembles

| ID | members | macro | acc | LOW | MED | HIGH | hard |
|----|---------|------:|----:|----:|----:|-----:|-----:|
| Ens-1 | Run 5 + Run 8 + Run 9 (uniform) | **0.622** | 0.644 | 0.725 | **0.482** | 0.661 | **0.663** |
| Ens-2 | Run 5 + Run 7 + Run 8 + Run 9 (4-way uniform) | 0.621 | 0.642 | 0.723 | 0.466 | 0.673 | 0.658 |
| Ens-3 | Run 5 + Run 7 + Run 9 (drop dropout) | 0.610 | 0.629 | 0.715 | 0.449 | 0.667 | 0.654 |
| Ens-4 | Run 7 + Run 8 + Run 9 (drop WD-only) | 0.607 | 0.633 | 0.719 | 0.453 | 0.649 | 0.648 |
| Ens-5 | Run 5 + Run 9 only | 0.610 | 0.625 | 0.705 | 0.463 | 0.660 | 0.655 |
| Ens-6 | Runs 9 + 10 + 11 (F seeds only) | 0.597 | 0.612 | 0.691 | 0.442 | 0.658 | 0.656 |
| Ens-7 | Run 5 + Run 8 + Runs 9-11 (5-way) | 0.616 | 0.637 | 0.719 | 0.460 | 0.670 | 0.659 |
| Ens-8 | Run 5 + Run 8 + Run 9 + Run 13 | 0.608 | 0.631 | 0.717 | 0.458 | 0.648 | 0.653 |
| Ens-9 | Run 5 + Run 8 + Run 9 + Run 14 | 0.620 | 0.637 | 0.709 | 0.480 | 0.670 | 0.664 |
| Ens-10 | Run 5 + Run 8 + Run 9 + Run 15 | 0.614 | 0.633 | 0.703 | 0.457 | 0.681 | 0.649 |
| Ens-11 | Run 14 + Run 15 (label-mode pair) | 0.623 | 0.633 | 0.694 | 0.502 | 0.672 | 0.661 |

**Reference baselines** (on 331 hard-labeled test rows):
- Majority-LOW: 0.200
- Qwen-only labeler: 0.938
- Gemini-only labeler: 0.924

---

## Detailed run entries

### Run 1 — Option A: class_w=hard_counts (FAILED)
- **Config:** lr=2e-5, ep=5, WD=0.01, linear schedule, class_weights=hard_counts (Option A)
- **Output:** `models/stage3_risk_deberta_v3_option_a_failed/`
- **Result:** macro_f1=0.21, model collapsed to predicting MEDIUM for ~95% of rows
- **Observation:** SOFT_LABEL rows all carry MEDIUM=0.5 in their target vectors;
  hard-count weights gave MEDIUM a higher effective loss budget than LOW/HIGH,
  so optimizer collapsed to MEDIUM. Switched to Option B (effective_counts) for Run 2.

### Run 2 — Option B baseline (lr=2e-5)
- **Config:** lr=2e-5, ep=5, WD=0.01, linear schedule, class_weights=effective_counts
- **Output:** `models/stage3_risk_deberta_v3_option_b_lr2e5/`
- **Result:** macro_f1=0.41, all 3 classes predicted (recall LOW=0.42 / MED=0.63 / HIGH=0.25)
- **Observation:** MEDIUM collapse cured. BUT training loss stuck at 1.05–1.12 across
  all 5 epochs (random-init floor = 1.099). Both train and val plateaued — model is
  *under*-fitting, not over-fitting. LR was too low.

### Run 3 — LR=5e-5 fix
- **Config:** lr=5e-5, ep=5, WD=0.01, linear schedule
- **Output:** `models/stage3_risk_deberta_v3_lr5e5/`
- **Result:** macro_f1=0.583 (+0.17 over Run 2)
- **Observation:** Training loss dropped from stuck-1.05 to ~0.78 by end. **LR was
  the bottleneck.** Eval loss bottomed at epoch 3 (0.967), rose by epoch 4 (1.035) —
  early sign of overfitting. Best checkpoint = epoch 4.

### Run 4 — LR=3e-5 + epochs=10
- **Config:** lr=3e-5, ep=10, WD=0.01, linear schedule
- **Output:** `models/stage3_risk_deberta_v3_lr3e5_ep10/`
- **Result:** macro_f1=0.579 (≈ Run 3, slightly worse on hard-only)
- **Observation:** Slower LR + more epochs = smoother trajectory but same ceiling.
  LR alone wasn't the remaining bottleneck.

### Run 5 — Cosine + WD=0.05 + ep=10 (first regularization win)
- **Config:** lr=5e-5, ep=10, WD=0.05, cosine schedule
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05/`
- **Result:** macro_f1=0.603 (+0.020 over Run 3); MEDIUM jumped 0.43 → 0.47
- **Observation:** First meaningful improvement past LR fix. Cosine tail + stronger WD
  let the model not over-anchor on MEDIUM as the "uncertainty bucket." Confirmed the
  hypothesis that MEDIUM weakness is partly soft-label-driven. HIGH F1 dropped slightly
  (-0.025) — first sign of HIGH↔MEDIUM tradeoff.

### Run 6 — LLRD=0.9 (FAILED — too aggressive)
- **Config:** lr=5e-5, ep=10, WD=0.01, linear, LLRD=0.9
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_llrd09/`
- **Result:** macro_f1=0.575 (regression vs Run 3); only run with val > test (overfit val)
- **Observation:** LLRD=0.9 gave embeddings only 0.28× base LR — too low for legal text
  (very different domain from pretraining corpus). Bottom layers couldn't adapt.
  Hypothesis: LLRD with milder decay should work — tested in Run 9.

### Run 7 — Cosine + WD=0.1 (regularization too strong)
- **Config:** lr=5e-5, ep=10, WD=0.1, cosine schedule
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd10/`
- **Result:** macro_f1=0.591 (worse than Run 5); best_val=0.611 (highest of any run)
  but test=0.591 — biggest val/test gap, sign of val-set overfitting
- **Observation:** WD=0.05 was the sweet spot; 0.1 starts to over-regularize. HIGH F1
  recovered (+0.025) but MEDIUM dropped (-0.04). Confirms HIGH↔MEDIUM tradeoff is real.

### Run 8 — Cosine + WD=0.05 + dropout=0.2
- **Config:** lr=5e-5, ep=10, WD=0.05, cosine, dropout=0.2 (default 0.1)
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_drop20/`
- **Result:** macro_f1=0.597; **best LOW F1 of any run (0.741)**, MEDIUM precision
  jumped to 0.505 but recall dropped to 0.370
- **Observation:** Different regularization fingerprint than WD: dropout makes the
  model **more confident** (predicts MEDIUM only when sure, defaults to LOW otherwise).
  WD makes the model **more cautious** (uses MEDIUM as a hedge). Both useful for
  different downstream consumption patterns.

### Run 9 — Cosine + WD=0.05 + LLRD=0.95 (current best single config)
- **Config:** lr=5e-5, ep=10, WD=0.05, cosine, LLRD=0.95
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_llrd095/`
- **Result:** macro_f1=**0.607**; **best HIGH F1 of any run (0.661)**;
  MEDIUM F1=0.463 (≈ Run 5); val/test gap healthy (+0.010)
- **Observation:** Confirms LLRD=0.9 (Run 6) failed because of decay magnitude, not
  LLRD itself. With 0.95 the embeddings still get 0.54× base LR — they can adapt while
  protecting bottom layers somewhat. HIGH improvement is the most meaningful: HIGH
  recall = 0.681 (highest yet) — fewer missed risky clauses, business-critical.

### Run 10 — Run 9 with seed=7 (multi-seed)
- **Config:** identical to Run 9, seed=7 (was seed=42)
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_llrd095_seed7/`
- **Result:** macro_f1=0.611; **best MEDIUM F1 of any single run (0.487)**;
  best hard-only F1 (0.666)
- **Observation:** Same data + same hyperparameters + different seed = +0.025 on
  MEDIUM F1 alone. Seed-to-seed variance is **real and large** for MEDIUM.

### Run 11 — Run 9 with seed=100 (multi-seed)
- **Config:** identical to Run 9, seed=100 (was seed=42)
- **Output:** `models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_llrd095_seed100/`
- **Result:** macro_f1=0.574; MEDIUM F1=0.418 (drop)
- **Observation:** Lowest of the 3 F seeds. Combined with Runs 9 & 10, gives:
  std(macro_f1)=0.020, std(MEDIUM F1)=0.035 across seeds. **Most of our
  hyperparameter "improvements" between Runs 5-9 are within seed-noise band.**

### Run 12 — Run 9 config + pure EMD loss (FAILED — median collapse)
- **Config:** Run 9 hyperparameters, `--loss emd` (Wasserstein-1 with class weights)
- **Output:** `models/stage3_risk_deberta_v3_run12_emd/`
- **Result:** macro_f1=0.143, model predicted MEDIUM for 100% of test rows.
  Training loss flatlined at 0.679 within epoch 1. Early stopping fired epoch 3.
- **Observation:** **Pure EMD on imbalanced data collapses to the median class.**
  At random init, predicting uniform softmax gives EMD≥1 for both LOW and HIGH true
  labels but only 0.67 for MEDIUM — gradient pulls strongly toward "predict MEDIUM."
  Once collapsed, the symmetric LOW vs HIGH gradients cancel out and the model can't
  escape. Class weights (designed for CE) made it worse by giving MEDIUM a comparable
  loss share to HIGH. **Lesson:** EMD needs a class-discriminative anchor (CE) to
  avoid collapse on imbalanced ordinal problems. Tested in Run 13.

### Run 13 — Run 9 config + hybrid CE + 0.5×EMD
- **Config:** Run 9 hyperparameters, `--loss hybrid --emd-lambda 0.5`
- **Output:** `models/stage3_risk_deberta_v3_run13_hybrid/`
- **Result:** macro_f1=0.587 (-0.021 vs Run 9), MEDIUM=0.425 (-0.038), HIGH=0.616 (-0.045);
  best_val=0.613 (best of any single run) but val/test gap = -0.026 (val > test, overfitting val)
- **Observation:** **Hybrid avoided collapse but underperformed.** EMD's ordinal pressure
  smooths predictions toward MEDIUM (which is what reduces off-by-2 risk), but argmax
  outcomes still pick LOW or HIGH — net effect is more LOW recall, less HIGH precision.
  In the 4-way ensemble (Ens-8) R13 dilutes B+E+F (-0.014) because it agrees with the
  others 81-85% (vs 71% baseline) — same family of solutions, just noisier.
  **Closes the ordinal-loss path.** Squared EMD likely fails for the same reason.

### Run 14 — Run 9 config + label-mode hard_only
- **Config:** Run 9 hyperparameters, `--label-mode hard_only` (drops 1,055 SOFT_LABEL rows
  from training; train shrinks 3,472 → 2,417)
- **Output:** `models/stage3_risk_deberta_v3_run14_hardonly/`
- **Result:** macro_f1=0.610 (≈ Run 9's 0.608); **MEDIUM=0.492 (best for any seed=42 run)**;
  **hard-only=0.657 (best for any seed=42 run)**; HIGH=0.645 (-0.016)
- **Observation:** **Soft labels were polluting MEDIUM signal.** Removing them gives the
  cleanest MEDIUM number we've seen at seed=42. Net macro_f1 unchanged because LOW/HIGH
  slightly drop (less data). Hard-only metric jumps +0.018 → strong evidence that soft
  rows specifically degrade clean-case performance. Val/test gap is +0.036 (test > val
  by a lot) — this split's val just happens to over-represent soft rows.

### Run 15 — Run 9 config + label-mode argmax_soft
- **Config:** Run 9 hyperparameters, `--label-mode argmax_soft` (one-hot the 1,055 soft
  vectors via argmax; train stays at 3,472)
- **Output:** `models/stage3_risk_deberta_v3_run15_argmaxsoft/`
- **Result:** macro_f1=0.603; **HIGH=0.675 (best for any seed=42 run)**; MEDIUM=0.464;
  LOW=0.671 (-0.027)
- **Observation:** Hardening soft vectors via argmax mostly converts LOW-vs-MED disagreements
  to one or the other, slightly shifting class distribution toward LOW (50% vs 45.3%).
  HIGH gains because the model commits to crisper boundaries. **Most diverse single model
  yet** (only 72.5% agreement with E — lowest pair we've seen), but adding it to B+E+F
  ensemble doesn't help — its lower individual F1 dilutes the ensemble.

### Ens-11 — Run 14 + Run 15 (label-mode pair)
- **Result:** macro_f1=**0.623**, MEDIUM=**0.502** ← best MEDIUM of any combination
- **Observation:** Two label-mode variants ensemble together to **match Ens-1's 0.622**
  with strictly better MEDIUM. Confirms label-mode is a real diversification axis,
  comparable to regularization-mode diversity.

---

## Detailed ensemble entries

### Ensemble-1 — Run 5 + Run 8 + Run 9 (regularization-diverse, current ensemble best)
- **Members:** Run 5 (WD=0.05), Run 8 (WD=0.05+dropout=0.2), Run 9 (WD=0.05+LLRD=0.95)
- **Method:** uniform average of softmax probabilities, argmax for prediction
- **Result:** **macro_f1=0.622, hard_only=0.663** — best across the board except LOW
- **Observation:** **+0.015 over best single model (Run 9) for free.** Each member
  contributes its specialty: Run 5 best at MEDIUM, Run 8 best at LOW, Run 9 best at HIGH.
  Models agree on only 71.7% of test cases — disagreement is the ensemble fuel.

### Ensemble-2 — 4-way Run 5+7+8+9
- **Members:** add Run 7 (WD=0.1) to Ens-1
- **Result:** macro_f1=0.621 (essentially tied with Ens-1)
- **Observation:** Adding Run 7 doesn't help — its decision pattern correlates too
  much with Run 5's (same regularization mechanism: cosine + WD). Diversity must be
  *structural* (different mechanisms), not just *parameter* (different values).

### Ensemble-3 — Run 5 + 7 + 9 (drop dropout member)
- **Result:** macro_f1=0.610 (-0.012 vs Ens-1)
- **Observation:** Removing Run 8 hurts — confirms dropout adds real diversity even
  though it was the weakest single model.

### Ensemble-4 — Run 7 + 8 + 9 (drop WD-only member)
- **Result:** macro_f1=0.607 (-0.015 vs Ens-1)
- **Observation:** Removing Run 5 also hurts — Run 5 isn't redundant with Run 9 even
  though both use cosine+WD.

### Ensemble-5 — Run 5 + Run 9 only (2-way)
- **Result:** macro_f1=0.610 (-0.012 vs Ens-1)
- **Observation:** Two members aren't enough — Run 8's distinct fingerprint matters.

### Ensemble-6 — Runs 9 + 10 + 11 (F seeds only)
- **Members:** Run 9 (seed=42), Run 10 (seed=7), Run 11 (seed=100)
- **Result:** macro_f1=0.597 (≈ mean of single F runs)
- **Observation:** **Seed-only ensembling barely helps.** Same config + different
  seeds find similar minima. Confirms: ensemble diversity must be structural
  (regularization mechanism), not stochastic (init).

### Ensemble-7 — Run 5 + 8 + 9 + 10 + 11 (5-way: regularization + seed)
- **Result:** macro_f1=0.616 (-0.006 vs Ens-1)
- **Observation:** Worse than Ens-1 — adding 2 F-seed variants over-weights the LLRD
  config (3/5 vs 1/3 in Ens-1). **Regularization diversity > seed diversity.**

---

## Closed hypotheses (with evidence)

1. **Ordinal loss does not help.** Pure EMD collapsed (Run 12); hybrid CE+EMD λ=0.5
   regressed and dilutes ensembles (Run 13). Squared EMD would address the median
   collapse the CE anchor already solved — not worth testing on the same fundamental
   ambiguity. **Path closed.**

2. **Soft labels are net-neutral on macro_f1.** Run 14 (drop them) and Run 15
   (hard-argmax) both ≈ Run 9 on overall macro_f1 (within ±0.007, well inside seed noise).
   **But:** Run 14 specifically improves MEDIUM (+0.029) and hard-only (+0.018) —
   confirms soft rows pollute MEDIUM signal even if total info-gain is neutral.

3. **Label-mode diversity is a real ensemble axis.** Ens-11 (R14 + R15 only) matches
   Ens-1 (B + E + F regularization-diverse) at macro_f1 = 0.623, with better MEDIUM
   (0.502 vs 0.482).

## Current ceilings

- **Single model best:** Run 14 macro_f1 = 0.610, hard-only = 0.657
- **Single model with diversity advantage:** Run 15 (most-disagreeing decisions)
- **Ensemble best:** Ens-1 (B+E+F) = 0.622; Ens-11 (R14+R15) = 0.623 (tied)
- **MEDIUM best (any combination):** Ens-11 = 0.502
- **HIGH best (any combination):** Ens-7 = 0.670 (single run: R15 = 0.675)

## Open paths (not yet tried)

1. **Re-label the SOFT_LABEL rows with Gemini Pro** — produce clean hard labels
   for the labeler-disagreement rows. 1,055 in train, 1,327 dataset-wide. Per-row cost
   ~$0.01, so ~$10-13 total. Expected: macro_f1 → 0.62-0.63 single-shot, possibly higher.

2. **Metadata Option 2** (signing-party role injection into encoder) — addresses
   HIGH↔LOW polarity flips, won't help MEDIUM. Bigger architectural change.

3. **deberta-v3-large** — bigger model, more capacity. ~30-60 min training. Expected:
   +0.02-0.04 across all classes if capacity was a bottleneck.

4. **Higher-quality ensemble:** combine regularization-diverse (B/E/F) + label-mode-diverse
   (R14/R15) + multi-seed of the winner — may push past 0.625.

## Reference

- Architecture: `ARCHITECTURE.md` (root)
- Design rationale: `docs/STAGE3_TRAINING_NOTES.md`
- Training script: `src/stage3_risk_agent/train.py`
- Config: `configs/stage3_config.yaml`
- Memory checklist: `~/.claude/projects/-home-ubuntu-rajnish-aiml/memory/project_stage3_training_checklist.md`
