# Stage 3 LoRA Cascade Experiment Summary

**Goal:** Implement a **Top-Down Hierarchical Classification** system (Cascaded LoRA Adapters) to replace the monolithic 3-class model. This mimics human legal review by first identifying risk existence and then determining severity.

### Implementation Overview
1. **Model:** `microsoft/deberta-v3-base` + PEFT LoRA (Rank=16, Alpha=32, Target: `query_proj, key_proj, value_proj, dense`).
2. **Metadata Injection:** Signing-party identity added to Segment A: `"clause_type | signing party: <parties_span>"`.
3. **Stage 1 (Risk Gate):** Binary classifier (LOW vs NOT-LOW). 
   - **Label Map:** `LOW -> 0`, `MEDIUM/HIGH -> 1`.
   - **What to Handle:** Monitor class distribution between "Safe" (LOW) and "Risky" clauses. Balance the threshold to ensure high recall for risk without overwhelming Stage 2 with false positives.
   - **Performance:** F1: 0.75, Recall: 0.97.
4. **Stage 2 (Severity Jury):** 5-fold ensemble binary classifier (MEDIUM vs HIGH). 
   - **Label Map:** `MEDIUM -> 0`, `HIGH -> 1` (LOW rows filtered out).
   - **What to Handle:** Significant class imbalance (HIGH is the minority class). Focus on preventing "HIGH risk" clauses from being misclassified as MEDIUM, as these are the most business-critical errors.
   - **Performance (Mean):** F1: 0.48, Recall: 0.70.
5. **Reference Hyperparameters (Kaggle Baseline):**
   *Note: These values served as our stable baseline but should be treated as starting points for further optimization.*
   - **Risk Detector (Stage 1):** LR=1e-4, Ep=15, LoRA Rank=16.
   - **Severity Jury (Stage 2):** LR=5e-5, Ep=25, LoRA Rank=32 (Higher capacity for nuance).
   - **Common:** WD=0.01, MaxLength=512, Batch=16, Warmup=0.1, Focal Loss (gamma=2.0).
6. **Consensus Strategy:** Stage 2 uses "Soft Voting" by averaging probabilities across all 5 folds.
7. **Deterministic Fix:** Use `torch.manual_seed(42)` to reconstruct identical heads for local inference.

This cascade architecture allows for independent threshold tuning (e.g., maximizing recall for Stage 1) without retraining the entire system.
