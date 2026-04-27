# Stage 3 Training — Prep Notes

> Lab-notebook style log of observations and decisions made while preparing the Stage 3
> DeBERTa risk classifier. Complements `ARCHITECTURE.md` (source of truth for design) with
> empirical findings that drove or validated the design choices. Updated as work progresses.

Created 2026-04-23 during training-prep phase.

---

## 1. Dataset shape (final)

After filtering (see §4), the training pool is:

| | Count |
|---|---|
| Total rows | 4,375 |
| Hard labels | 3,048 (CE loss) |
| Soft labels | 1,327 (KLDiv loss) |
| Unique contracts | 474 |
| Unique clause types | 36 |

Hard-label class mix: LOW 44.5% / MEDIUM 32.3% / HIGH 23.1%.

Source: `data/processed/training_dataset.json` via `scripts/build_training_dataset.py`.

---

## 2. Model choice: DeBERTa-v3-base

Chose `microsoft/deberta-v3-base` over `microsoft/deberta-base` for the Stage 3 risk
classifier. Rationale:

- **Stronger pretraining**: v3 uses ELECTRA-style replaced-token-detection instead of v1's
  masked-language-modeling. Published benchmarks show v3 consistently ahead on downstream
  classification tasks at the same parameter count.
- **Same VRAM**: both are 12-layer 768-hidden ~86M-param models. No infra cost.
- **More efficient tokenizer on legal text**: v3 uses a SentencePiece vocab of 128k vs v1's
  BPE vocab of 50k. Legal terms that v1 fragments into multiple subwords tend to be single
  tokens in v3, shortening the input sequence.

Stage 1+2 (Anushka's extraction model) stays on `deberta-base` — already trained, not
worth swapping mid-stream.

## 3. Clause length distribution

Measured with the `microsoft/deberta-v3-base` tokenizer, combined input
`tokenizer(clause_type, clause_text)` → `[CLS] clause_type [SEP] clause_text [SEP]`.

| Percentile | Tokens |
|---|---|
| Mean | 72 |
| Median | 57 |
| p75 | 88 |
| p90 | 133 |
| p95 | 168 |
| p99 | 292 |
| Max | 657 |

**Only 5 rows (0.11%) exceed 512 tokens, 0 exceed 768.** `max_length=512` with right-side
truncation is comfortably sufficient — no chunking / head+tail strategy needed.
`max_length=256` would still cover 98.5% and is a viable speed optimization if GPU time
becomes a bottleneck.

For reference, the same measurement with deberta-base tokenizer (what we would have used
on the rejected option) gave mean=87, p99=358, max=1,181, with 18 rows (0.41%) over 512.
v3's tokenizer is ~15–20% more efficient on this corpus.

---

## 4. Train/val/test split

**Strategy**: `StratifiedGroupKFold` — group by `contract` (zero leakage), stratify by
`clause_type` (ensures every type is represented in every split). 80/10/10 target.

**Final split (seed=100)**:

| Split | Rows | % | Contracts | Clause types |
|---|---|---|---|---|
| train | 3,472 | 79.4% | 379 | 36/36 |
| val | 437 | 10.0% | 48 | 36/36 |
| test | 466 | 10.6% | 47 | 36/36 |

Label distribution across splits is within ±3pp — stratification did its job.
See `data/processed/splits.json` (persisted) and `scripts/build_splits.py`.

### Why seed=100

Tried seeds `[0, 1, 7, 13, 17, 21, 42, 100, 123, 777]`. With data this scarce (36 clause
types, the rarest — Source Code Escrow — has only 13 rows), no group-and-stratify split is
mathematically perfect, and most seeds drop 1–2 rare types from val or test:

| Seed | Missing from val | Missing from test |
|---|---|---|
| 0 | Price Restrictions, Source Code Escrow | — |
| 1 | Unlimited/All-You-Can-Eat-License | — |
| 7, 13 | Source Code Escrow | — |
| 17 | — | Price Restrictions, Third Party Beneficiary |
| 21, 42 | Most Favored Nation | — |
| **100** | **none** | **none** |
| 123 | — | Source Code Escrow |
| 777 | Affiliate License-Licensor | — |

Seed=100 was the only one in the scanned set that gave full 36/36 coverage in both val and
test. This is not "seed fishing" for accuracy — it's optimizing for a concrete structural
property (every clause type represented in every split so per-type F1 can be computed).

### Why stratifying on `clause_type × final_label` doesn't work

Tempting to ask for finer stratification that also preserves LOW/MEDIUM/HIGH balance within
each clause type. Infeasible here: the smallest strata are too small to split.
- `Price Restrictions × HIGH` = 1 row
- `Most Favored Nation × HIGH` = 2 rows
- `Affiliate License-Licensor × MEDIUM` = 2 rows

Scikit-learn falls back to random when a stratum has fewer members than splits, which
defeats the point. Stratifying by `clause_type` alone (36 buckets) is robust and still
preserves LOW/MEDIUM/HIGH balance **globally** across splits via the law of large numbers.

### Known tension: group vs stratify

`StratifiedGroupKFold` cannot simultaneously satisfy both constraints perfectly when they
conflict. Our data makes this manageable because rare clause types happen to be ~1:1
with their contracts (Source Code Escrow: 13 rows / 13 distinct contracts; same for
Joint IP, Price Restrictions, etc.). Moving one contract barely disturbs rare-type
representation. Common types like Governing Law (n=436) can absorb some unevenness.

### Rare-type per-split noise

Even with the best seed, rare types have tiny per-split counts:

| Type | total | val | test |
|---|---|---|---|
| Source Code Escrow | 13 | 1 | 2 |
| Price Restrictions | 15 | 2 | 1 |
| Unlimited/All-You-Can-Eat-License | 17 | 1 | 1 |
| Most Favored Nation | 28 | 2 | 1 |

F1 on these types will be statistically noisy (a single miss swings F1 by a lot).
When reporting per-type metrics, always include the sample size and add a confidence caveat
for n<10. These are not failures of the classifier; they're hard limits of the dataset.

---

## 5. Data quality findings

### Fragment clauses (9 rows, removed)

Sanity pass surfaced 9 training rows with `<15` character `clause_text`:

```
'corporation.'   → HIGH   (Competitive Restriction Exception)
'state.'         → HIGH   (Governing Law)
'us.'            → HIGH   (Ip Ownership Assignment)
'Business.'      → HIGH   (Non-Transferable License)
'You must:'      → HIGH   (Insurance)
'initial term.'  → HIGH   (Renewal Term)
'initial term.'  → HIGH   (Notice Period To Terminate Renewal)
'120 days'       → MEDIUM (Warranty Duration)
'South Dakota'   → LOW    (Governing Law)
```

**Root cause**: these are CUAD's ground-truth annotations. The original Atticus Project
annotators highlighted these exact spans as the answer to the clause-type question. Some
are legitimate minimal answers ("South Dakota" IS the governing jurisdiction; "120 days"
IS the warranty duration). Others are annotator errors where a pronoun or a list header
got selected instead of the real clause text.

**How they evaded review**:
- They made it through Qwen+Gemini reconciliation because both labelers happened to agree
  (both conservatively labeled degenerate fragments as HIGH).
- They were never seen by human reviewers because manual review only saw HIGH↔LOW flips.
  The similar MANUAL_REVIEW fragments (MR-015 `"."`, MR-170 `"HSWI Websites)."`, MR-173,
  MR-175) were caught only because those rows had label disagreement.

**Decision**: drop them via `len(clause_text.strip()) < 15` filter in
`scripts/build_training_dataset.py`. Training a risk classifier on `"us." → HIGH`
teaches noise. All 9 rows were in the `train` split, so val/test metrics are unaffected.

**Implication for Stage 1**: DeBERTa-QA cannot "fix" these at extraction time — if it's
trained on CUAD, it learns to reproduce these annotations. The right mitigation is a
post-prediction filter at inference: short, low-confidence spans should be rejected or
flagged. Captured in `docs/STAGE1_REVIEW_NOTES.md` as item #8.

### Cross-split text duplicates (22 rows, kept)

44 duplicate groups (by `(clause_type, normalized clause_text)`) total 95 rows. Of these,
34 are confined to one split; **10 span multiple splits (22 rows, 0.50%)**.

These are boilerplate like "this agreement shall be governed by... laws of the state of
New York" — common clause patterns repeated across distinct contracts.

**Decision**: leave them. 0.50% is well within noise. Memorization-inflated test F1 on
these is negligible. We also *want* the classifier to handle boilerplate correctly;
removing it throws away real signal.

---

## 6. Soft-label construction

Confidence-weighted probability vectors for adjacent (LOW↔MEDIUM or MEDIUM↔HIGH)
disagreements between Qwen and Gemini. Qwen's `conf=0.0` artifact (11.7% of its rows)
is floored to `0.5` before normalization — labels on those rows are valid, the score
is a known model quirk, not genuine uncertainty. See ARCHITECTURE.md "Known Issue — Qwen
`conf=0.0` Artifact" for the full analysis.

Implementation: `scripts/build_training_dataset.py`. Uniform 0.5/0.5 fallback available
via `--no_conf_weight` for ablation.

---

## 7. Loss function and class weights

### Unified soft-target cross-entropy (one loss, not two)

ARCHITECTURE.md originally described "CE for hard rows, KLDiv for soft rows" — two loss
functions dispatched per-sample. At implementation time this collapses to a single,
simpler formulation:

```python
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
loss    = loss_fn(logits, soft_targets)     # soft_targets: [B, 3] float
```

`nn.CrossEntropyLoss` accepts a probability distribution as the target (since PyTorch 1.10).
Both row types flow through identically:

- Hard rows have `soft_label = [1, 0, 0]` / `[0, 1, 0]` / `[0, 0, 1]` → this reduces to
  standard CE on that class.
- Soft rows have `soft_label = [0.4, 0.6, 0]` etc. → the loss distributes proportionally
  across classes.

**Mathematical equivalence**: CE with a soft target equals the KL divergence between
target and prediction up to an additive constant (the target's entropy) that does not
affect gradients. So this is identical to the "CE + KLDiv branching" design, just cleaner
code. No dispatch logic, no risk of a branching bug.

The upstream `scripts/build_training_dataset.py` already stores both row types as 3-vectors
in the `soft_label` field, so the trainer reads a single uniform format.

### Class weights (Option A: hard-only counts)

Mild class imbalance in hard labels:

| Class | Count | % | Weight `N / (K · count)` |
|---|---|---|---|
| LOW | 1,357 | 44.5% | 0.749 |
| MEDIUM | 987 | 32.3% | 1.030 |
| HIGH | 705 | 23.1% | 1.442 |

Using `N = 3,049` (total hard rows) and `K = 3` classes. Higher weight on HIGH compensates
for its lower frequency — without this, gradient descent would drift toward "predict LOW
when uncertain," which is exactly backwards for our use case (missing a HIGH is the
expensive error).

**Computed from hard counts only** (not including soft-label mass).

Rationale:
- Class weights compensate for *optimization bias*. Hard rows are the dominant source
  of that bias — soft rows already carry built-in uncertainty in the target.
- Weights don't shift much between the two methods: Option A gives HIGH weight 1.44,
  Option B (include soft mass) gives 1.56. Within noise for our mild imbalance.
- Simpler to reason about — one clear quantity drives the weights.

**Update (post-Run 1):** Run 1 with Option A weights produced MEDIUM collapse (macro_f1=0.21).
Soft rows each contribute 0.5 to the MEDIUM target slot, so Option A underestimated MEDIUM's
effective frequency and gave it a disproportionate loss budget relative to how often the
model was correct on it. **Option B (effective_counts, including soft-label probability mass
in the class-count computation) was adopted from Run 2 onwards** and is the actual
implementation in `src/stage3_risk_agent/train.py`. The weights in the table above are the
original Option A plan values, kept for reference.

---

## 8. Hyperparameters (Section D decisions)

### Fine-tuning strategy

**Full fine-tuning** — all ~86M parameters (embeddings + 12 transformer layers +
classification head) are trainable. No LoRA, no freezing.

Rationale: DeBERTa-v3-base is base-sized, not LLM-scale; memory is not a constraint
on A100-40GB. LoRA's benefits don't kick in at this scale. Small dataset (3,472 rows)
does carry overfit risk, but weight decay + dropout + early stopping handle it.

**Layer-wise LR decay (LLRD)** — implemented and tested in Runs 6 and 9.
Run 6 (LLRD=0.9): regressed vs baseline — decay too aggressive, bottom layers got only
0.28× base LR and couldn't adapt to the legal text domain. Run 9 (LLRD=0.95): +0.004
macro over Run 5, best HIGH F1 of any single-seed run (0.661). **LLRD=0.95 is the
current baseline config** (in `configs/stage3_config.yaml`).

### Core optimizer hyperparameters

| | Value | Rationale |
|---|---|---|
| `batch_size` | 16 | Fits A100 VRAM; 217 updates/epoch on 3,472 train rows |
| `learning_rate` | 2e-5 | Within the DeBERTa-v3 paper's validated band (6e-6 to 3e-5 across tasks) |
| `warmup_ratio` | 0.1 | ~108 steps warmup; gives AdamW time to build stable gradient variance |
| `lr_scheduler_type` | linear | Post-warmup linear decay to 0; forces the model to settle |
| `epochs` | 5 (max) | Ceiling — early stopping will stop earlier if val plateaus |
| `early_stopping_patience` | 2 | Stop if val macro-F1 fails to improve for 2 consecutive epochs |
| `metric_for_best_model` | val_macro_f1 | Primary metric from Section E |
| `weight_decay` | 0.01 | AdamW default; HuggingFace Trainer excludes bias+LayerNorm automatically |
| `seed` | 42 | Independent from the `splits.json` seed (100); different random process |

### Precision — the fragile one

**Target: bf16. Fallback: fp32. Explicitly NOT fp16.**

Stage 1 experiments (Apr 2026) documented NaN issues with DeBERTa + reduced precision:

1. **transformers library bug** — `modeling_deberta_v2.py` uses `torch.finfo(dtype).min`
   as the attention mask fill value. In fp16, that's `-65504`, which overflows during
   softmax stabilization (`max-subtract` step), producing `-inf` → `NaN`. Still present
   in the currently-installed transformers 5.1.0. See `notebooks/EXPERIMENT_NOTES.md`.

2. **HF Trainer + DeBERTa-v3 interaction** — Stage 1 QA experiments hit NaN even at fp32
   with Trainer, while a manual forward pass in plain PyTorch ran cleanly (loss 6.95,
   no NaN). Root cause never fully isolated; suspects were `processing_class=tokenizer`
   and the v2 data collator. Stage 3 uses classification (not QA span prediction), which
   is a simpler head and loss — the interaction may not repeat, but we can't assume.

**De-risking plan for Stage 3:**

- Run a pre-training smoke test (plain PyTorch loop, no Trainer) — verifies the model
  loads cleanly in bf16, 10 forward+backward steps produce no NaN, gradients flow.
- If bf16 smoke test passes → proceed with bf16 (expected ~2× throughput on A100).
- If bf16 smoke test fails → rerun smoke test in fp32 to isolate the issue.
  - fp32 passes → fall back to fp32 precision (acceptable — ~30% slower, no numerical risk).
  - fp32 also fails → drop to `deberta-base` (v1, Stage 1's proven config).
- If Trainer training then NaNs after smoke test passes → rewrite as a custom PyTorch
  training loop (~80 lines), bypasses any Trainer-specific interaction.

`strict_determinism = false`. `torch.use_deterministic_algorithms(True)` costs 10-30%
throughput for bit-exact reproducibility. We accept ~0.1 F1 run-to-run wiggle in exchange.

### Smoke test — bf16 PASSED (2026-04-23)

`scripts/smoke_test_stage3.py` ran:
- deberta-v3-base + classification head (184.4M params total — v3's 128k vocab drives
  the size; ~86M backbone + ~98M embeddings)
- bf16 precision, plain PyTorch loop, no HF Trainer
- Soft-target cross-entropy loss with class weights [0.749, 1.030, 1.442]
- 10 forward+backward+optimizer steps on real clauses from the train split, batch=8

Result: no NaN/Inf in loss, logits, gradients, or classifier weights at any step.
Loss fluctuated around 1.0 (initial random-init 3-class baseline `-log(1/3) ≈ 1.099`)
which is the expected range — 10 steps is not enough to learn, just enough to verify
numerical stability.

`finfo(bf16).min = -3.39e38` was exercised in DeBERTa's attention masking on every
step without triggering NaN. The Stage 1 failure was not reproduced in this isolated
plain-PyTorch setup; the root cause there was likely HF Trainer + QA-head interaction,
not the model itself.

**Bf16 is safe to use for Stage 3 training.** The real trainer will still include a
`NaNDetector` callback as belt-and-suspenders.

### HF Trainer phase — also PASSED (extended smoke test, 2026-04-23)

After the plain-PyTorch phase passed, a second phase ran the same config through
HuggingFace `Trainer` — because Stage 1's NaN was specifically a Trainer+v3
interaction. Result: 10 `trainer.train()` steps completed cleanly, loss around
1.10-1.20 (normal 3-class random-init range), classifier weights sane post-training.

Details that may have avoided the Stage 1 trap:
- **Classification head** (simpler than Stage 1's QA span prediction)
- **No `processing_class=tokenizer`** passed to Trainer — one of the Stage 1 NaN
  suspects (per `notebooks/EXPERIMENT_NOTES.md`). We pre-tokenize into a HF
  `Dataset` and pass `data_collator=DefaultDataCollator()` directly.
- **`remove_unused_columns=False`** in `TrainingArguments` — keeps our `soft_label`
  field through the data loader; without this the Trainer strips non-model-input
  columns before `compute_loss` runs.
- **transformers 5.5.4** (may have fixed earlier v3 interaction bugs that hit 4.57)

**Decision: use HuggingFace Trainer for the real training loop.** Saves ~120 lines
vs a custom PyTorch loop, and now empirically de-risked.

---

## 9. Evaluation & metrics (Section E decisions)

Metrics are split into two tiers based on cost and purpose.

### Tier A — computed every validation epoch

Cheap to compute; drives early stopping and model selection.

| Metric | Use |
|---|---|
| **Val macro-F1** (primary) | `metric_for_best_model` in Trainer — early stopping monitor |
| Per-class precision/recall/F1 (LOW/MED/HIGH) | Diagnosis — *which* class is failing and in which direction |
| Overall accuracy | Cheap; easy sanity check |

Macro-F1 weights each class equally regardless of size. Important because LOW is
the majority class (44.5%) — a model that always predicts LOW gets 44.5% accuracy
but macro-F1 around 0.21. We want the model to handle all three classes.

**HIGH recall is the business-critical sub-metric.** Missing a HIGH clause means an
unflagged legal risk in the downstream report. Per-epoch reporting of HIGH recall
lets us spot models that are accurate-on-average but dangerously biased against HIGH.

### Tier B — computed only on the held-out test set (post-training)

Expensive or low-value during training; meaningful once at the end.

**Per-clause-type F1 (36 types)** — tells us where the model is systematically
weak. Example expected failures: Uncapped Liability F1 may be low because signing-party
ambiguity isn't resolvable from clause_text alone (the Hybrid architecture's agent path
is exactly the intended remedy — low F1 here is *expected* and motivates escalation).

Reporting includes sample size. Types with n<5 rows in test (Source Code Escrow,
Price Restrictions, Affiliate License-Licensor, etc.) are statistically noisy —
a single wrong prediction swings F1 by 20+ points. Flag with a caveat.

### Confidence calibration — load-bearing for the Hybrid architecture

The Hybrid architecture's **0.6 confidence gate** only makes sense if the model's
self-reported confidence matches its true accuracy. An over-confident model routes
wrong-but-confident predictions past the agent; an under-confident model wastes LLM
budget on already-correct predictions.

**Measurement**: **Expected Calibration Error (ECE)**. Bin predictions by confidence
(0.0-0.1, 0.1-0.2, …, 0.9-1.0); in each bin compute the mean confidence and the
actual accuracy; ECE is the weighted average of `|mean_conf - accuracy|` across
bins. Target: ECE<0.05 ideally, <0.10 acceptable.

**Visualization**: Reliability diagram. Points on the diagonal = calibrated.
Below-diagonal = overconfident. Above = underconfident.

**Fix if miscalibrated**: **temperature scaling**. Replace `softmax(logits)` with
`softmax(logits/T)` where `T` is a single scalar fit on val set to minimize NLL or
ECE. Standard technique (Guo et al., 2017). Doesn't change which class is predicted
(argmax is preserved) — only adjusts the confidence numbers. Typically reduces ECE
by 2-5×. Takes ~60 seconds to fit.

### Baselines (test set only)

Three baselines on the hard-labeled test subset (AGREED + MANUAL_REVIEW +
GEMINI_PRO_REVIEW rows in test split, ~325 rows — SOFT_LABEL rows have no single
ground-truth class, excluded from this comparison):

1. **Majority-class predictor** — always predict LOW. Macro-F1 floor ≈ 0.21. If we
   don't beat this, training failed fundamentally.
2. **Qwen-only** — use raw `qwen_label` as prediction. Shows what Qwen alone gives
   us without any merging or fine-tuning.
3. **Gemini-only** — same with `gemini_label`.

If our DeBERTa doesn't beat (2) or (3) meaningfully, the fine-tuning pipeline added
no value over either labeler alone — a signal that the merge + training effort was
not worth it. Realistic expectation: DeBERTa should beat both since it absorbs
signal from *both* labelers' agreement + the human reconciliation.

### SOFT_LABEL test rows (optional extra metric)

Test split contains ~141 SOFT_LABEL rows with no single ground-truth class. Skipped
for headline baselines (no fair comparison), but useful for a calibration signal:
compute **KL divergence between predicted softmax and the soft target vector**.
Low KL = model matches labeler uncertainty on borderline cases. Treated as a
secondary diagnostic, not a primary metric.

---

## 10. Reproducibility

All prep is deterministic given the inputs:
- `data/review/master_label_review.csv` → `scripts/build_training_dataset.py` →
  `data/processed/training_dataset.json` (4,375 rows)
- `data/processed/training_dataset.json` → `scripts/build_splits.py --seed 100` →
  `data/processed/splits.json`

Anyone on the team can regenerate both files with two commands. The split seed is pinned
in the script default, so no arguments needed.

Training seed (`42`) is separate — pinned in `configs/stage3_config.yaml`. Same code +
same data + same seed on the same GPU gives results within ~0.1 F1 of each other.

---

## 11. Training status (updated 2026-04-27)

**Training is complete.** 17 runs + 13 ensemble configurations have been executed and
documented in `docs/STAGE3_EXPERIMENTS.md`. All prep sections (A–G) are done.

Current ceilings on current label quality:
- **Single model:** Run 14 macro_f1=0.610, hard-only=0.657
- **Best HIGH single model:** Run 17 CORN = 0.681 (but MEDIUM collapsed to 0.074)
- **Ensemble:** Ens-B (R5+R8+R10+R14+R15) macro_f1=0.6264, hard-only=0.6733

Diagnosed bottleneck: **MEDIUM label noise** (SOFT_LABEL rows). Evidence: stronger
regularisation hurts (Run 7 regressed), seed variance is high (std=0.020 macro / 0.035
MEDIUM), train loss descends normally, and every structural approach regresses MEDIUM.

**Next action:** Gemini Pro relabel of the 1,055 SOFT_LABEL rows in the train split
(~$10-13 total). Script precedent: `scripts/run_gemini_pro_review.py`.
After relabel, retry CORN (Run 18) — CORN's HIGH=0.681 on noisy labels suggests the
architecture is sound; clean MEDIUM labels should allow classifier2 to recover.

See `docs/STAGE3_EXPERIMENTS.md` for full run details, closed paths, and ensemble results.
