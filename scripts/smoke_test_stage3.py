"""
smoke_test_stage3.py
====================
Pre-training smoke test for Stage 3 DeBERTa-v3-base risk classifier.

Goal: prove our exact training config (bf16 + deberta-v3-base + soft-target CE
with class weights) is numerically stable BEFORE we build the full trainer.

Why it matters: Stage 1 experiments (see notebooks/EXPERIMENT_NOTES.md) hit NaN
in multiple DeBERTa precision configurations — at least one due to a known
library bug (torch.finfo(fp16).min overflow in attention masking) and another
un-diagnosed v3+HF-Trainer interaction. This test uses a raw PyTorch loop —
no HuggingFace Trainer — so we isolate model/precision issues from Trainer
mysteries.

Checks (in order):
  Check 1 — Model loads cleanly in bf16; weights are sane
  Check 2 — 10 forward+backward+optimizer steps on real training data in bf16
  Check 3 — (only runs if Check 2 fails) Same in fp32, to localize the bug

On first NaN: stop, print detailed diagnostics (layer, tensor magnitudes,
finfo values in use), exit 1.

Usage:
    /home/ubuntu/miniconda3/envs/rajnish-env/bin/python3 scripts/smoke_test_stage3.py

Exit codes:
  0 → bf16 passed — safe to train in bf16
  1 → bf16 failed, fp32 passed — fall back to fp32
  2 → both failed — model or code issue; investigate before training
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Config — pull from stage3_config.yaml in spirit, hardcoded for smoke test
MODEL_NAME       = "microsoft/deberta-v3-base"
DATASET_PATH     = Path("data/processed/training_dataset.json")
SPLITS_PATH      = Path("data/processed/splits.json")
MAX_LENGTH       = 512
BATCH_SIZE       = 8                          # smaller than real training — just for smoke test
NUM_STEPS        = 10
LEARNING_RATE    = 2e-5
SEED             = 42
NUM_LABELS       = 3                          # LOW / MEDIUM / HIGH
# Class weights from STAGE3_TRAINING_NOTES.md §7 (hard-count method)
CLASS_WEIGHTS    = torch.tensor([0.749, 1.030, 1.442], dtype=torch.float32)


def banner(s):
    print(f"\n{'='*70}\n{s}\n{'='*70}")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def soft_target_ce(logits, targets, class_weights):
    """Weighted cross-entropy against soft (probability-vector) targets.

    For one-hot targets this reduces to standard CE on the true class.
    For soft targets it computes -sum_i ( w_i * target_i * log_softmax(logits)_i ).
    logits:  [B, C] float
    targets: [B, C] float — probability distribution per row, rows sum to 1
    class_weights: [C] float on same device/dtype as logits
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)   # compute softmax in fp32 for stability
    per_class = class_weights.to(log_probs.device) * targets * log_probs
    loss = -per_class.sum(dim=-1).mean()
    return loss


def check_tensor(name, t, tag=""):
    """Return True if tensor has NaN or Inf; also print a short summary."""
    with torch.no_grad():
        has_nan = torch.isnan(t).any().item()
        has_inf = torch.isinf(t).any().item()
        t_f = t.float()
        msg = (f"  {tag}{name}: shape={tuple(t.shape)} "
               f"min={t_f.min().item():.4g} max={t_f.max().item():.4g} "
               f"mean={t_f.mean().item():.4g}")
        if has_nan or has_inf:
            msg += f"  ← NaN={has_nan} Inf={has_inf}"
        print(msg)
        return has_nan or has_inf


def load_batch(tokenizer, rows, dtype_device):
    """Tokenize a list of rows into a model-ready batch."""
    enc = tokenizer(
        [r["clause_type"] for r in rows],
        [r["clause_text"] for r in rows],
        padding=True, truncation=True, max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    enc = {k: v.to(dtype_device["device"]) for k, v in enc.items()}
    targets = torch.tensor([r["soft_label"] for r in rows], dtype=torch.float32,
                           device=dtype_device["device"])
    return enc, targets


def run_phase(precision: str, tokenizer, train_rows, device):
    """Run both Check 1 and Check 2 for a given precision. Returns True on success."""
    banner(f"PRECISION: {precision}")
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]

    # finfo sanity — the known landmine
    finfo_min = torch.finfo(dtype).min
    print(f"  torch.finfo({dtype}).min = {finfo_min:.4g}")
    if precision == "bf16" and abs(finfo_min) > 1e30:
        print(f"  ← this is used by DeBERTa attention masking; bf16's huge .min "
              f"value could NaN if any arithmetic operation amplifies it")

    # Check 1 — model load
    banner(f"CHECK 1/2 [{precision}]: model loads cleanly")
    seed_everything(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        dtype=dtype,
    ).to(device)
    if check_tensor("classifier.weight", model.classifier.weight, tag="[check1] "):
        print("  ✗ classifier head NaN/Inf at init — very unexpected")
        return False
    print(f"  ✓ model loaded in {precision}, weights sane, {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Check 2 — 10 steps of real training
    banner(f"CHECK 2/2 [{precision}]: {NUM_STEPS} forward+backward steps on real data")
    optim = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    class_weights = CLASS_WEIGHTS.to(device)
    model.train()

    random.shuffle(train_rows)
    for step in range(1, NUM_STEPS + 1):
        batch_rows = train_rows[(step-1)*BATCH_SIZE : step*BATCH_SIZE]
        enc, targets = load_batch(tokenizer, batch_rows, {"device": device})

        optim.zero_grad()
        out = model(**enc)
        logits = out.logits
        loss = soft_target_ce(logits, targets, class_weights)

        # Step-level diagnostics
        print(f"\n--- step {step} ---")
        if check_tensor("logits", logits, tag="  "):
            print(f"  ✗ logits NaN/Inf after forward — FAIL")
            return False
        if check_tensor("loss", loss.unsqueeze(0), tag="  "):
            print(f"  ✗ loss NaN/Inf — FAIL (step {step})")
            return False

        loss.backward()

        # Check grads (just the classifier for speed — if any layer NaNs, this one usually does too)
        grad_has_nan = False
        for n, p in [("classifier.weight", model.classifier.weight),
                     ("classifier.bias",   model.classifier.bias)]:
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                print(f"  ✗ gradient NaN/Inf in {n}")
                grad_has_nan = True
        if grad_has_nan:
            return False

        # Optimizer step
        optim.step()

    print(f"\n  ✓ {NUM_STEPS} training steps completed with no NaN/Inf")
    return True


def main():
    banner("STAGE 3 SMOKE TEST — DeBERTa-v3-base + bf16 + soft-target CE")
    print(f"  model:   {MODEL_NAME}")
    print(f"  data:    {DATASET_PATH}")
    print(f"  splits:  {SPLITS_PATH}")
    print(f"  batch:   {BATCH_SIZE}, max_length: {MAX_LENGTH}, steps: {NUM_STEPS}")
    print(f"  seed:    {SEED}")

    if not torch.cuda.is_available():
        print("\n✗ CUDA not available — aborting. Check your Python environment.")
        sys.exit(2)
    device = torch.device("cuda:0")
    print(f"  device:  {torch.cuda.get_device_name(0)}")
    print(f"  bf16:    {torch.cuda.is_bf16_supported()}")

    # Load data once
    data = json.loads(DATASET_PATH.read_text())
    splits = json.loads(SPLITS_PATH.read_text())
    train_rns = set(splits["train"])
    train_rows = [r for r in data if r["row_num"] in train_rns]
    print(f"  train rows available: {len(train_rows)}")

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Phase 1: bf16
    if run_phase("bf16", tokenizer, list(train_rows), device):
        banner("RESULT: bf16 PASSED — safe to train in bf16")
        sys.exit(0)

    # Phase 2: fp32 fallback
    print("\n  bf16 failed — retrying in fp32 to localize...")
    torch.cuda.empty_cache()
    if run_phase("fp32", tokenizer, list(train_rows), device):
        banner("RESULT: bf16 FAILED, fp32 PASSED — fall back to fp32")
        sys.exit(1)

    banner("RESULT: BOTH FAILED — investigate before training")
    sys.exit(2)


if __name__ == "__main__":
    main()
