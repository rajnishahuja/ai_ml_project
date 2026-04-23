# Stage 1+2 Review Notes — Suggestions for Alignment

> **Context**: Anushka's PR #2 (merged 2026-04-14) implemented Stage 1+2 in `src/stage1_extract_classify/`. 
> These are suggestions to align her code with the shared architecture defined in Phase 0.
> None of these block current work — they're improvements to discuss.

---

## 1. Use shared `ClauseObject` from `schema.py`

**Current**: `baseline.py` defines `BaselineClause` (8 fields), `pipeline.py` has its own `ClauseObject` copy.  
**Suggested**: Import `ClauseObject` from `src/common/schema.py` (T0.9). Stage 3 expects this as input.  
**Files**: `baseline.py:254`, `pipeline.py:45`

## 2. Use shared metrics from `utils.py`

**Current**: `evaluate.py` re-implements `normalize_answer`, `squad_em_f1`, `span_iou` (lines 101–169).  
**Suggested**: Import from `src/common/utils.py` where these already exist.  
**Files**: `evaluate.py:101-169`

## 3. Single source for `CUAD_CLAUSE_TYPES`

**Current**: Defined in `constants.py` (good), but also duplicated in `predict.py:16-31`.  
**Suggested**: `predict.py` should import from `constants.py` when implemented.  
**Files**: `constants.py`, `predict.py:16-31`

## 4. Dataset approach — resolved

**Decision**: Stick with `CUAD_v1.json` from the official CUAD GitHub repo. The `theatticusproject/cuad-qa` HuggingFace dataset uses the same underlying data but has compatibility issues with newer `datasets` library versions (deprecated loading script). No advantage to switching.

## 5. Data leakage in train/val/test split — high priority

**Current**: `preprocess_cuad.py:86` splits at the QA-pair level using `train_test_split()`. This scatters clauses from the same contract across train, val, and test sets. Since all 41 QA pairs for a contract share the same context, the model sees the same contract text during training and evaluation.  
**Impact**: Eval metrics are inflated — the model memorizes contract-specific patterns, not generalizable clause extraction. Real-world performance on unseen contracts will be lower than reported.  
**Fix**: Split at the **contract level** (group by `title` field) using `sklearn.model_selection.GroupShuffleSplit`. All QA pairs from one contract stay in the same split.  
**Files**: `preprocess_cuad.py:86-94`

## 6. Pipeline refactoring (T1.1-T1.3)

**Current**: `pipeline.py` remains monolithic (463 lines). `model.py`, `train.py`, `predict.py` are stubs.  
**Planned**: T1.1 (model.py), T1.2 (train.py), T1.3 (predict.py) — extract from pipeline.py.  
**Status**: Not started. Need to coordinate who picks these up.

## 7. Classification accuracy is misleading — high priority

**Current**: `evaluate.py` infers the predicted clause type as `pred_type = true_type if pred_text else "NO_CLAUSE"`. The clause type always comes from the question (it's part of the input, not a model output), so any non-empty answer is automatically counted as a correct type prediction.  
**Impact**: Classification accuracy in the current reports is effectively measuring "did the model return *some* answer" — not "did it classify the clause correctly". A model that returns the same boilerplate on every query would look good on this metric.  
**Fix**: Reframe the classification metric as **clause presence vs absence per type** (binary: did the model correctly predict that this clause type exists / doesn't exist in this contract). Report per-type precision/recall/F1 instead of a pooled accuracy number.  
**Files**: `evaluate.py` (classification block)

## 8. Post-prediction filter for fragment spans — medium priority

**Observed**: CUAD's ground-truth annotations include a small number of "fragment" spans — single words, pronouns, or sentence-ending punctuation highlighted as the answer to a clause-type question. Examples from CUAD_v1.json (all real ground-truth answers):

```
"us."          → Ip Ownership Assignment
"state."       → Governing Law
"corporation." → Competitive Restriction Exception
"Business."    → Non-Transferable License / License Grant
"You must:"    → Non-Compete / Exclusivity / Insurance
"initial term."→ Renewal Term / Notice Period To Terminate Renewal
```

A DeBERTa-QA model trained on CUAD will learn to reproduce these exactly when they appear in test contexts. They're not "noise" from our pipeline — they're the dataset's annotations.

**Impact**: At inference time these fragments would be emitted as legitimate clause extractions, polluting downstream Stage 3 with meaningless text. In our Stage 3 labeling pool they appeared 9 times and were filtered out via `len(clause_text.strip()) < 15` in `scripts/build_training_dataset.py` — but that's a per-downstream-stage workaround, not a fix.

**Fix (suggested)**: Add a post-prediction filter in Stage 1 `predict.py` / `evaluate.py`. If the predicted span is shorter than N characters (~15) **and** the DeBERTa start/end logits are below a confidence floor, reject it (treat as `NO_CLAUSE` or flag for review). Valid short answers like "South Dakota" for Governing Law should still pass if model confidence is high. Threshold tunable per clause type.

**Files**: `src/stage1_extract_classify/predict.py`, `src/stage1_extract_classify/evaluate.py`  
**Discovered**: 2026-04-23 during Stage 3 data sanity pass; see `docs/STAGE3_TRAINING_NOTES.md` §4 "Fragment clauses".

---

*Created 2026-04-14 during architecture review. Items #7–8 added 2026-04-23.*
