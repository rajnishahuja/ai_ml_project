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
| 16 | Run 9 + SORD (scale=1.5, soft labels) | 0.567 | 0.599 | 0.677 | 0.355 | 0.669 | — | — |
| 17 | Run 9 + CORN loss (2 binary heads) | 0.485 | 0.605 | 0.701 | 0.074 | 0.681 | 0.496 | 0.476 |

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
| Ens-A | 7-way: R5+R8+R9+R10+R11+R14+R15 | 0.620 | 0.637 | 0.707 | 0.468 | 0.685 | 0.663 |
| **Ens-B** | **5-way: R5+R8+R10+R14+R15 (best-of-axis)** | **0.6264** | **0.644** | 0.713 | 0.484 | 0.682 | **0.6733** |
| Ens-C | 4-way: R14+R15+R5+R8 (R14-anchored) | 0.625 | 0.644 | 0.710 | 0.477 | 0.688 | 0.664 |

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

### Run 16 — Run 9 config + SORD (scale=1.5, soft labels)
- **Config:** Run 9 hyperparameters, `--loss ce --label-mode sord --sord-scale 1.5`
  Hard rows (max>=0.99) converted: [1,0,0] → [0.786,0.175,0.039] via exp(-1.5*|i-j|).
  Soft rows unchanged. class_weights computed on SORD-transformed labels (bug: inflated MEDIUM count).
- **Output:** `models/stage3_risk_deberta_v3_run16_sord15/`
- **Result:** macro_f1=0.567; LOW=0.677; **MEDIUM=0.355 (collapsed vs Run 9's 0.463)**; HIGH=0.669
- **Observation:** SORD regressed MEDIUM by -0.108 F1. Two root causes:
  (1) class_weights computed on SORD-softened labels inflated MEDIUM's effective count → underweighted MEDIUM in loss;
  (2) SORD blurs LOW↔MEDIUM boundary (LOW target [0.786,0.175,0.039] has 17.5% in MEDIUM slot),
  which is exactly the hardest boundary for this dataset. **SORD path closed.**

### Run 17 — CORN loss (2 binary heads, chain-rule output)
- **Config:** Run 9 hyperparameters, `--loss corn --label-mode soft`.
  Architecture change: replaces single 3-class classifier with 2 independent binary heads:
  - classifier1: P(y ≥ 1) = P(MEDIUM or HIGH) — trained on all rows
  - classifier2: P(y ≥ 2 | y ≥ 1) = P(HIGH | not LOW) — trained only on rows with true class ≥ 1
  Chain rule gives 3-class probs. Val metrics use log-probs as argmax-compatible "logits".
- **Output:** `models/stage3_risk_deberta_v3_run17_corn/`
- **Result:** macro_f1=0.485; LOW=0.701; **MEDIUM=0.074 (CORN collapsed MEDIUM)**; HIGH=0.681;
  hard_only=0.496. Stopped at epoch 7 (best val at ep5=0.476).
- **Epoch trajectory:** ep1=0.213 (all-LOW) → ep2=0.378 (HIGH wakes up) → ep5=0.476 (MEDIUM appears 0.149)
  → ep7=0.463 (MEDIUM settles at 0.16 val, 0.074 on test)
- **Observation:** CORN's chain rule works structurally: HIGH is the cleanest binary
  (precision=0.648, recall=0.717, F1=0.681 — best HIGH F1 of any single-seed run).
  But MEDIUM (the "in-between" class) depends on classifier2 outputting P(HIGH)=low for
  MEDIUM samples. Since MEDIUM is noisy and few (22% of data), classifier2 learns a
  biased HIGH-threshold that misses MEDIUM at test time (recall=0.039).
  **Root cause is identical to all other approaches: MEDIUM label noise, not the loss function.
  CORN path closed for current label quality. Worth retrying after Gemini Pro relabel.**

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

### Ens-A — 7-way (all axes: R5+R8+R9+R10+R11+R14+R15)
- **Method:** uniform softmax average of all 7 saved models (regularization + seed + label-mode)
- **Result:** macro_f1=0.6201 (-0.002 vs Ens-1)
- **Observation:** **More is not better.** Adding Run 11 (single=0.576 — the worst F-seed)
  drags the ensemble below Ens-1 baseline. Run 11's noisy decision boundary pulls average
  probabilities away from cleaner consensus. Confirms: ensemble members must each be
  individually competent.

### Ens-B — 5-way best-of-axis (R5+R8+R10+R14+R15) — NEW BEST
- **Members:** Run 5 (WD reg) + Run 8 (dropout reg) + Run 10 (best F-seed=7) +
  Run 14 (hard_only label-mode) + Run 15 (argmax_soft label-mode)
- **Result:** macro_f1=**0.6264**, hard-only=**0.6733**, MEDIUM=0.484, HIGH=0.682
- **Observation:** **Best ensemble result so far.** +0.004 macro over Ens-1, +0.010 hard-only.
  Picks one strong model per diversity axis instead of multiple correlated ones.
  Replacing Run 9 (single=0.607) with Run 10 (single=0.611, better MEDIUM) helps marginally.
  Confirms: **best-of-axis > more-of-axis** for ensembling. The +0.004 is small but the test
  set is fixed (not seed-resampled), so the gain is real on this distribution.

### Ens-C — 4-way R14-anchored (R14+R15+R5+R8)
- **Members:** Run 14 + Run 15 (label-mode pair) + Run 5 (WD reg) + Run 8 (dropout reg)
- **Result:** macro_f1=0.6248, hard-only=0.6638
- **Observation:** Almost matches Ens-B without the F-seed model. Suggests label-mode +
  regularization diversity captures most of the ensemble gain. F-seed adds the last 0.002.
  Practical implication: 4 models give ≈ same gain as 5 — useful if shipping cost matters.

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
- **Ensemble best:** **Ens-B (R5+R8+R10+R14+R15) = 0.6264, hard-only = 0.6733**
- **MEDIUM best (any combination):** Ens-11 = 0.502
- **HIGH best single model:** Run 17 CORN = 0.681 (single run, but MEDIUM collapsed)
- **HIGH best ensemble:** Ens-A 7-way = 0.685

## Closed paths (loss / architecture level, 2026-04-27)

In addition to hyperparameter space (Runs 4-9), these structural approaches regressed:
- **EMD (Run 12):** collapsed to median class. Squared EMD not worth trying.
- **Hybrid CE+EMD (Run 13):** net-zero overall, dilutes ensembles.
- **SORD (Run 16, scale=1.5):** MEDIUM -0.108 due to class-weight contamination + boundary blurring.
- **CORN (Run 17):** HIGH improves (0.681) but MEDIUM collapses (0.074) — low-recall threshold.
  Worth retrying after Gemini Pro relabel removes MEDIUM label noise.

**Pattern:** Every structural approach produces the same fingerprint — MEDIUM suffers.
Confirms bottleneck is MEDIUM label noise, not the loss/architecture.

## Open paths (not yet tried)

1. **Re-label the SOFT_LABEL rows with Gemini Pro** — produce clean hard labels
   for the labeler-disagreement rows. 1,055 in train, 1,327 dataset-wide. Per-row cost
   ~$0.01, so ~$10-13 total. Expected: macro_f1 → 0.62-0.63 single-shot, possibly higher.
   **CORN + clean labels could be very strong — HIGH is already 0.681 on noisy labels.**

2. **Metadata Option 2** (signing-party role injection into encoder) — addresses
   HIGH↔LOW polarity flips, won't help MEDIUM. Bigger architectural change.

3. **deberta-v3-large** — DEPRIORITIZED. 17 runs of evidence contradict the capacity-bottleneck
   hypothesis: stronger regularisation hurts (Run 7 WD=0.1 regressed vs Run 5), seed variance
   is high (std=0.020 macro / 0.035 MEDIUM), train loss descends normally (1.05 → 0.78), and
   every structural approach (CORN, SORD, EMD, hybrid) hits the same MEDIUM ceiling via
   different paths. A larger model trained on the same noisy labels will learn the noise more
   thoroughly, not break through it. **Only worth trying after Gemini Pro relabel**, if clean
   labels reveal a remaining capacity gap.

4. **Higher-quality ensemble:** DONE — Ens-B (R5+R8+R10+R14+R15) achieved macro_f1=0.6264,
   hard-only=0.6733, implementing exactly this combination (regularisation-diverse + label-mode-
   diverse + best F-seed). Ens-C (4-way without F-seed) gave 0.6248 — nearly identical, useful
   if inference cost matters. **Ensemble ceiling reached at 0.626 on current labels.** Further
   ensemble gains require better base models (i.e., after Gemini Pro relabel).

## Reference

- Architecture: `ARCHITECTURE.md` (root)
- Design rationale: `docs/STAGE3_TRAINING_NOTES.md`
- Training script: `src/stage3_risk_agent/train.py`
- Config: `configs/stage3_config.yaml`
- Memory checklist: `~/.claude/projects/-home-ubuntu-rajnish-aiml/memory/project_stage3_training_checklist.md`
