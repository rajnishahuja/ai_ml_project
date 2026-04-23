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

## 2. Clause length distribution

Measured with `microsoft/deberta-base` tokenizer, combined input
`tokenizer(clause_type, clause_text)` → `[CLS] clause_type [SEP] clause_text [SEP]`.

| Percentile | Tokens |
|---|---|
| Mean | 87 |
| Median | 68 |
| p75 | 106 |
| p90 | 159 |
| p95 | 206 |
| p99 | 358 |
| Max | 1,181 |

**Only 18 rows (0.41%) exceed 512 tokens.** `max_length=512` with right-side truncation is
more than sufficient — no chunking / head+tail strategy needed. `max_length=256` would still
cover 97.3% and is a viable speed optimization if GPU time becomes a bottleneck.

---

## 3. Train/val/test split

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

## 4. Data quality findings

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

## 5. Soft-label construction

Confidence-weighted probability vectors for adjacent (LOW↔MEDIUM or MEDIUM↔HIGH)
disagreements between Qwen and Gemini. Qwen's `conf=0.0` artifact (11.7% of its rows)
is floored to `0.5` before normalization — labels on those rows are valid, the score
is a known model quirk, not genuine uncertainty. See ARCHITECTURE.md "Known Issue — Qwen
`conf=0.0` Artifact" for the full analysis.

Implementation: `scripts/build_training_dataset.py`. Uniform 0.5/0.5 fallback available
via `--no_conf_weight` for ablation.

---

## 6. Reproducibility

All prep is deterministic given the inputs:
- `data/review/master_label_review.csv` → `scripts/build_training_dataset.py` →
  `data/processed/training_dataset.json` (4,375 rows)
- `data/processed/training_dataset.json` → `scripts/build_splits.py --seed 100` →
  `data/processed/splits.json`

Anyone on the team can regenerate both files with two commands. The split seed is pinned
in the script default, so no arguments needed.

---

## 7. Still open (before training code)

See `memory/project_stage3_training_checklist.md` for the live list. As of 2026-04-23,
Section A (data prep) is complete. Section B (model & tokenizer), Section C (loss & signal
implementation), Section D (hyperparameters), Section E (evaluation), Section F
(post-training), and Section G (engineering) remain.
