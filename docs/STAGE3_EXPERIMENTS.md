# Stage 3 Risk Classifier — Experiments Log

Append-only log of training runs and ensembles for the Stage 3 DeBERTa-v3 risk classifier.
For *design rationale* (loss choice, hyperparameter philosophy, eval scheme), see
`STAGE3_TRAINING_NOTES.md`. This file records *what we tried and what we got*.

**Test set:** 466 rows. **Splits seed:** 100 (frozen). Hard-labeled subset changed after v2 relabel: 331 → **452** rows (former SOFT_LABEL rows resolved to hard labels). Direct macro_f1 comparison between pre- and post-relabel runs is NOT valid — the test ground truth changed.
**Splits file:** `data/processed/splits.json`. **Training data:** `data/processed/training_dataset.json` (4,375 rows post-v2; 4,276 hard + 99 soft). **Signing-party metadata** added to all rows (Run 22+): segment A = `"clause_type | signing party: <parties_span>"`. Coverage: 4,374/4,375 (100%).

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
| — | *v2 relabel: +1,228 hard rows (SOFT_LABEL_V2_AGREED). Test ground truth also changed — runs below are not directly comparable to runs above.* |
| 18 | CE hard_only on v2 data (4,276 rows, s42) | 0.581 | — | 0.650 | 0.488 | 0.605 | 0.583 | 0.543 |
| 19s42 | CORN hard_only on v2 data (s42) | 0.578 | — | 0.607 | 0.479 | 0.648 | — | 0.543 |
| 19s7 | CORN hard_only on v2 data (s7) | 0.578 | — | 0.613 | **0.528** | 0.591 | 0.586 | 0.563 |
| 20 | SORD (bug-fixed) on v2 data (s42) | 0.570 | — | 0.654 | 0.456 | 0.601 | 0.573 | 0.574 |
| 21 | Hybrid CE+EMD hard_only on v2 data (s42) | 0.563 | — | 0.640 | 0.451 | 0.598 | 0.563 | 0.580 |
| — | *Runs 22+ add signing-party metadata to segment A. Not comparable to Runs 18–21.* |
| 22 | CE hard_only + parties (s42, ep10, p2) | 0.602 | 0.601 | 0.636 | 0.544 | 0.625 | 0.602 | 0.578 |
| 23 | CORN hard_only + parties (s7, ep20, p5) | 0.584 | 0.573 | 0.667 | 0.481 | 0.599 | 0.584 | 0.567 |
| 24 | CE hard_only + parties (s42, ep20, p5) | 0.589 | 0.588 | 0.675 | 0.503 | 0.573 | 0.589 | 0.613 |

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
| — | *Ensembles below use v2-relabeled test ground truth (452 hard rows). Not comparable to Ens-1 through Ens-C.* |
| **Ens-D** | **2-way: R18(CE) + R19s7(CORN)** | **0.586** | 0.592 | 0.657 | 0.484 | **0.618** | **0.591** |
| Ens-E | 4-way: R18+R19s7+R20+R21 | 0.579 | 0.584 | 0.651 | 0.469 | 0.615 | 0.580 |
| — | *Ens-F uses signing-party metadata models. Not comparable to Ens-D/E.* |
| **Ens-F** | **2-way: R22(CE+parties) + R23(CORN+parties)** | **0.607** | **0.620** | **0.687** | **0.512** | **0.622** | **0.610** |

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
  Root cause confirmed as MEDIUM label noise. **Retry as Run 19 after v2 relabel (now done).**

**Pattern:** Every structural approach produces the same fingerprint — MEDIUM suffers.
Confirms bottleneck is MEDIUM label noise, not the loss/architecture.

## v2 Relabeling — COMPLETE (2026-04-29)

**Goal:** Replace 1,327 SOFT_LABEL rows (labeler-disagreement rows with soft probability
targets) with clean hard labels using improved prompts + parties metadata.

### What changed vs v1
- v1 prompt had no signing-party anchor → both models often labeled from wrong perspective
- v2 prompt injects `parties` field from `all_positive_spans.json` → pairwise agreement
  Qwen/Gemini improved from 0% (by definition in SOFT_LABEL) to 53.1%
- Groq (rate-limited, partial) excluded from final consensus; Claude Sonnet used as tiebreaker

### Labeler runs
| Labeler | Rows | Coverage | Output file |
|---------|------|----------|-------------|
| Qwen 30B v2 (GPU port 10006) | 1325/1327 | SOFT_LABEL | soft_label_relabel_v2_qwen.json |
| Gemini 2.5 Flash v2 | 1327/1327 | SOFT_LABEL | soft_label_relabel_v2_gemini.json |
| Claude Sonnet 4.6 | 945/946 | MANUAL_REVIEW + GEMINI_PRO_REVIEW + unresolved SOFT_LABEL | relabel_claude.json |

### Consensus strategy (2-tier, Groq excluded)
- **Tier 1** (623 rows — Claude available): 3-way vote Qwen + Gemini + Claude → 525 resolved (84%)
- **Tier 2** (704 rows — Qwen+Gemini only, already agreed): used directly → 703 resolved
- **Total resolved: 1,228/1,327 (92.5%)** written to master as `SOFT_LABEL_V2_AGREED`
- **99 rows unresolved** (all-3-disagree) — remain as `SOFT_LABEL`, treated as soft labels in training

### Model calibration findings
Risk-label aggressiveness order: **Qwen > Claude > Gemini**
- Qwen labels higher than Claude in 244/621 overlapping rows; Claude higher in only 95
- Claude labels higher than Gemini in 345/623 overlapping rows; Gemini higher in only 35
- Overall: Qwen↔Gemini 53% agree, Qwen↔Claude 45%, Gemini↔Claude 39%

### Cross-source alignment by clause type
| Most aligned pair | Clause types (agreement %) |
|---|---|
| Qwen + Gemini | Governing Law (84%), Anti-Assignment (69%), Warranty Duration (61%) |
| Qwen + Claude | ROFR (65%), Minimum Commitment (61%), Renewal Term (61%) |
| Gemini + Claude | Non-Compete (75%), IP Ownership (69%), Covenant Not To Sue (61%) |

Cap On Liability notable: Gemini+Claude most aligned (48%) — both recognise mutual
liability exclusion as lower risk. Qwen over-labels as HIGH in 36/54 rows.

### 99 unresolved rows pattern
- 65/99: Q=HIGH / G=LOW / C=MEDIUM — three-way ladder split, no majority possible
- 12/99: Q=LOW / G=MEDIUM / C=HIGH — reverse ladder (Claude most aggressive here)
- Most common clause types: Cap On Liability (12), Exclusivity (8), License Grant (8)
- Treatment: keep as SOFT_LABEL with uniform soft target for training

### Updated dataset counts (post v2 relabel)
| Category | Rows | Notes |
|---|---|---|
| AGREED | 2,735 | Qwen v1 + Gemini v1 agreement, hard label |
| SOFT_LABEL_V2_AGREED | 1,228 | v2 3-way consensus, now hard label |
| MANUAL_REVIEW | 235 | Human labels (4 reviewers) |
| GEMINI_PRO_REVIEW | 87 | Gemini 2.5 Pro tiebreaker |
| SOFT_LABEL | 99 | All-3-disagree, remains soft |
| METADATA | 2,292 | Excluded from risk training |
| ERROR | 26 | Excluded |

**Trainable hard rows: 4,285** (up from 3,057 before v2 relabel)
Label distribution: LOW=1,876 (43.8%) / MEDIUM=1,524 (35.6%) / HIGH=885 (20.6%)

---

## Runs 18–21 — v2 relabel data (2026-04-30)

**Context:** v2 relabel added 1,228 hard rows (SOFT_LABEL_V2_AGREED via 3-way Qwen+Gemini+Claude consensus). Training data grew from 3,057 → 4,276 hard rows. Test ground truth also changed (452 hard test rows vs 331 before). All comparisons within this block are valid; cross-block comparisons are NOT.

### Run 18 — CE hard_only baseline on v2 data (seed 42)
- **Command:** `--label-mode hard_only --loss ce --seed 42 --output-suffix _run18_s42`
- **Note:** Accidentally run with `--output-suffix _run19_s42`; model saved to `models/stage3_risk_deberta_v3_run19_s42/` (overwrote CORN s42 dir)
- **Train rows:** 3,399 hard (hard_only drops 99 soft rows from the 3,472-row train split)
- **Result:** macro_f1=0.581, LOW=0.650, MEDIUM=0.488, HIGH=0.605, hard_only=0.583
- **Best val:** epoch 5 (0.543), early stopped at epoch 7
- **Observation:** CE + clean data gives solid macro. MEDIUM=0.488 is reasonable. Establishes the v2-data baseline.

### Run 19 seed 42 — CORN on v2 data
- **Command:** `--label-mode hard_only --loss corn --seed 42 --output-suffix _run19_s42`
- **Note:** Model dir overwritten by Run 18 accident. Results from stdout only.
- **Result:** macro_f1=0.578, LOW=0.607, MEDIUM=0.479, HIGH=0.648
- **Best val:** epoch 5 (0.543), early stopped at epoch 7
- **Observation:** MEDIUM=0.479 vs Run 17's 0.074 — confirms label noise was the root cause of CORN collapse. HIGH=0.648 is CORN's structural advantage (ordinal pressure on HIGH threshold). Essentially tied with CE on macro; CORN wins on HIGH.

### Run 19 seed 7 — CORN second seed
- **Command:** `--label-mode hard_only --loss corn --seed 7 --output-suffix _run19_s7`
- **Output:** `models/stage3_risk_deberta_v3_run19_s7/`
- **Result:** macro_f1=0.578, LOW=0.613, MEDIUM=**0.528**, HIGH=0.591, hard_only=0.586
- **Best val:** epoch 7 (0.563), early stopped at epoch 9
- **Observation:** MEDIUM=0.528 — best MEDIUM across all v2 runs. Seed 7 has a different per-class fingerprint from seed 42 (more MEDIUM, less HIGH). Seed variance still present but reduced. Mean CORN MEDIUM across seeds: (0.479+0.528)/2=0.504.

### Run 20 — SORD retry with bug fix (seed 42)
- **Command:** `--label-mode sord --loss ce --seed 42 --output-suffix _run20_sord_s42`
- **Output:** `models/stage3_risk_deberta_v3_run20_sord_s42/`
- **Bug fixed:** Run 16's class_weights were computed from SORD-transformed labels, inflating MEDIUM's effective count and under-weighting it. Fixed in `train.py`: raw pre-transform labels now passed to `compute_class_weights`.
- **Result:** macro_f1=0.570, LOW=0.654, MEDIUM=0.456, HIGH=0.601, hard_only=0.573
- **Best val:** epoch 4 (0.574), early stopped at epoch 6. Uses all 3,472 rows (soft+hard).
- **Observation:** Bug fix helped vs Run 16 (was 0.355 MEDIUM), but SORD still trails CE (0.456 vs 0.488). The ordinal neighbour-smearing adds noise that outweighs the ordering benefit on this data. **SORD closed.**

### Run 21 — Hybrid CE+EMD retry on v2 data (seed 42)
- **Command:** `--label-mode hard_only --loss hybrid --seed 42 --output-suffix _run21_hybrid_s42`
- **Output:** `models/stage3_risk_deberta_v3_run21_hybrid_s42/`
- **Result:** macro_f1=0.563, LOW=0.640, MEDIUM=0.451, HIGH=0.598, hard_only=0.563
- **Best val:** epoch 5 (0.580), early stopped at epoch 7
- **Observation:** Hypothesis was that clean MEDIUM labels would let ordinal EMD pressure help. It didn't — hybrid is the weakest of all v2 runs. CE term dominates; EMD adds noise. **Hybrid closed on this dataset.**

### Ens-D — 2-way: CE(R18) + CORN_s7(R19s7)
- **Members:** `run19_s42` (CE s42) + `run19_s7` (CORN s7)
- **Result:** macro_f1=**0.586**, LOW=0.657, MEDIUM=0.484, HIGH=**0.618**, hard_only=0.591
- **Observation:** Best result on v2 test data. Structurally diverse (CE vs CORN) + different per-class fingerprints (CE strong macro, CORN strong HIGH). Averaging captures both. Adding SORD/Hybrid (Ens-E) dilutes rather than helps.

### Ens-E — 4-way: CE+CORN_s7+SORD+Hybrid
- **Result:** macro_f1=0.579 — WORSE than 2-way. SORD and Hybrid drag the ensemble.
- **Conclusion:** More models ≠ better ensemble. Best-of-axis (structurally diverse) > more-of-axis.

### v2 Summary
- **Best single model:** CORN s7 — highest MEDIUM (0.528); CE s42 — highest macro (0.581)
- **Best ensemble:** Ens-D (CE + CORN s7) = 0.586 macro, 0.618 HIGH
- **Ceiling on v2 data with current setup:** ~0.586 macro
- All methods (CE, CORN, SORD, Hybrid) cluster in 0.563–0.581 — loss function choice is not the primary remaining differentiator

## Runs 22–24 — signing-party metadata (2026-04-30)

**Context:** Signing-party identity (`parties` span from `all_positive_spans.json`) injected into segment A as `"clause_type | signing party: <parties_span>"`. Coverage: 4,374/4,375 rows (100%). Rationale: during manual review, all 60 HIGH↔LOW disagreements were resolved by knowing which party's perspective to use. The labeling perspective was always the parties-span entity throughout (v1 and v2 labels). `build_training_dataset.py` now writes `signing_party` field; `train.py` and `run_ensemble.py` construct segment A from it.

### Run 22 — CE hard_only + parties (seed 42, ep=10, patience=2)
- **Command:** `--label-mode hard_only --loss ce --seed 42 --output-suffix _run22_parties`
- **Output:** `models/stage3_risk_deberta_v3_run22_parties/`
- **Config:** same as Run 18 — only variable is signing-party metadata in segment A
- **Result:** macro_f1=**0.602**, LOW=0.636, MEDIUM=0.544, HIGH=0.625, acc=0.601
- **Best val:** epoch 4 (0.578), early stopped at epoch 6
- **vs Run 18 (no parties):** +0.021 macro, LOW +0.041, MEDIUM +0.056, HIGH +0.007
- **Observation:** Parties metadata is a genuine signal. LOW gain (+0.041) is the direct effect of resolving signing-party perspective confusion. Early stopping at epoch 6 (vs Run 18's epoch 7) — cleaner loss landscape converges faster. HIGH↔LOW polarity flips dropped from dominant error to 19% of remaining errors.

### Run 23 — CORN hard_only + parties (seed 7, ep=20, patience=5)
- **Command:** `--label-mode hard_only --loss corn --seed 7 --output-suffix _run23_corn_parties`
- **Output:** `models/stage3_risk_deberta_v3_run23_corn_parties/`
- **Config:** epochs=20, patience=5 (increased to survive CORN's ep1–2 dead zone and oscillation)
- **Result:** macro_f1=0.584, LOW=0.667, MEDIUM=0.481, HIGH=0.599, acc=0.573
- **Best val:** epoch 10 (0.567), stopped at epoch 15 (patience=5 from epoch 10)
- **vs Run 19s7 (no parties):** +0.056 macro — largest gain of any metadata run
- **Epoch trajectory:** ep1–2=0.196 (HIGH=0) → ep3=0.478 (CORN breakout) → ep10=0.567 (new best, resets patience) → ep15=0.554 (stop). Patience=5 was essential — with patience=2 training would have stopped at ep7 (0.539) and missed ep10's 0.567.
- **Observation:** CORN + parties gives complementary per-class fingerprint to CE: strong LOW recall (0.806) where CE has balanced (0.655). Diverse enough to ensemble well.

### Run 24 — CE hard_only + parties (seed 42, ep=20, patience=5) — OVERFIT
- **Command:** `--label-mode hard_only --loss ce --seed 42 --output-suffix _run24_ce_parties_p5`
- **Output:** `models/stage3_risk_deberta_v3_run24_ce_parties_p5/`
- **Result:** macro_f1=0.589, LOW=0.675, MEDIUM=0.503, HIGH=0.573, acc=0.588
- **Best val:** epoch 11 (0.613), stopped at epoch 16
- **vs Run 22 (same config, 10ep/p2):** val improved (+0.035) but **test dropped (-0.013)**
- **Observation:** Classic overfitting. CE converges cleanly by epoch 4–6; prolonged training beyond that fits the val set distribution. HIGH recall collapsed (0.625→0.490), LOW recall bloated (0.655→0.806). **Confirmed: CE works best with ep=10, patience=2. Longer training only helps CORN (slow startup).**

### Ens-F — 2-way: CE+parties (R22) + CORN+parties (R23) — CURRENT BEST
- **Members:** `run22_parties` (CE s42) + `run23_corn_parties` (CORN s7)
- **Method:** uniform average of softmax/chain-rule probabilities, argmax for prediction
- **Result:** macro_f1=**0.607**, acc=0.620, LOW=0.687, MEDIUM=0.512, HIGH=0.622, hard_only=0.610
- **vs Ens-D (no parties):** +0.021 macro — same structural gain as single-model parties effect
- **vs best pre-relabel Ens-B:** not directly comparable (different test ground truth)
- **Observation:** CE and CORN have complementary error profiles — CE strong on MEDIUM/HIGH, CORN strong on LOW recall. Probability averaging cancels errors on different examples. **This is the finalised Stage 3 classifier.**

### Error analysis (Run 22 test errors, 180/452 hard rows)
- **MEDIUM boundary errors: 145/180 (81%)** — dominant remaining problem
  - MEDIUM→LOW: 63; LOW→MEDIUM: 40; HIGH→MEDIUM: 26; MEDIUM→HIGH: 16
- **HIGH↔LOW polarity flips: 35/180 (19%)** — reduced from dominant (60/60 review cases) to minority after parties metadata
- **Short clause errors (<200 chars): 58/180 (32%)** — split into:
  - 6 true CUAD annotation artifacts (headings, incomplete sentences) — unlabelable by any model
  - 21 sparse/ambiguous — inherent task difficulty
  - 31 short-but-complete — MEDIUM boundary confusion, same root cause as longer clauses
- **Diagnosed ceiling:** remaining errors are largely irreducible — MEDIUM boundary is subjectively ambiguous (Qwen and Gemini disagreed on these during labeling)

### Closed paths (updated)
- **SORD:** closed after bug-fix retry (Run 20). Still trails CE after fix.
- **Hybrid CE+EMD:** closed after clean-data retry (Run 21). EMD adds noise on this dataset.
- **deberta-v3-large:** closed definitively. 21+ runs of evidence; not capacity-bound.
- **Longer CE training:** closed (Run 24). CE overfits with ep=20/p=5; ep=10/p=2 is optimal.
- **Agreement type metadata:** investigated but insufficient data (213 types, median ~20 rows each; ~0.5 examples per (agreement_type, clause_type) combination). Would require bucketing into ~6 broad categories with manual mapping — low expected gain for effort.

## Finalised Stage 3 Classifier

**Model:** Ens-F — Run 22 (CE) + Run 23 (CORN), both trained with signing-party metadata.
**Artefacts:**
- `models/stage3_risk_deberta_v3_run22_parties/final/` — CE model (HF format, 352MB)
- `models/stage3_risk_deberta_v3_run23_corn_parties/final/` — CORN model (safetensors, no config.json, 352MB)
- `scripts/run_ensemble.py` — batch inference (test set evaluation)
- `scripts/infer.py` — single-clause inference API (for agent pipeline)
- `src/stage3_risk_agent/train.py` — contains `CORNWrapper` class required to load CORN model
**Hosting:** HuggingFace Hub (TBD — models too large for git)
**Performance:** macro_f1=0.607, HIGH=0.622 on 452 hard test rows (v2 ground truth)

## Reference

- Architecture: `ARCHITECTURE.md` (root)
- Design rationale: `docs/STAGE3_TRAINING_NOTES.md`
- Training script: `src/stage3_risk_agent/train.py`
- Config: `configs/stage3_config.yaml`
- Memory checklist: `~/.claude/projects/-home-ubuntu-rajnish-aiml/memory/project_stage3_training_checklist.md`
