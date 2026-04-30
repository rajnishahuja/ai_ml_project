#!/usr/bin/env python3
"""Probability-averaging ensemble across saved Stage 3 risk classifier checkpoints.

Loads each model dir, runs softmax on the test split, averages probabilities
across models, then reports the same metrics shape as a single run.

Usage:
    python scripts/run_ensemble.py \\
        models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05 \\
        models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_drop20 \\
        models/stage3_risk_deberta_v3_lr5e5_ep10_cosine_wd05_llrd095

    python scripts/run_ensemble.py --label "B+E+F" <dirs...>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from datasets import Dataset

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.stage3_risk_agent.train import LABEL_NAMES, NUM_LABELS, CORNWrapper  # noqa: E402


def is_corn_model(model_path: Path) -> bool:
    """CORN models lack config.json in final/ — saved as raw state dicts."""
    final = model_path / "final" if (model_path / "final").exists() else model_path
    return not (final / "config.json").exists()


def load_model(model_path: Path, cfg: dict, device):
    """Load either a standard HF model or a CORNWrapper from a run output dir."""
    mp = Path(model_path)
    bp = mp / "final" if (mp / "final").exists() else mp

    if is_corn_model(mp):
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=1,
        )
        model = CORNWrapper(base)
        from safetensors.torch import load_file
        state = load_file(str(bp / "model.safetensors"), device="cpu")
        model.load_state_dict(state)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            str(bp), num_labels=NUM_LABELS,
        )
    return model.to(device)


def corn_probs(model, ds, device, batch_size: int = 32) -> np.ndarray:
    """Run CORN inference and convert 2 binary logits → 3-class probabilities."""
    model.eval()
    keep_cols = [c for c in ds.column_names if c not in ("labels", "row_num")]
    loader = DataLoader(
        ds.select_columns(keep_cols).with_format("torch"),
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
            logit1, logit2 = model._corn_logits
            p_ge1 = torch.sigmoid(logit1).squeeze(-1)
            p_ge2 = torch.sigmoid(logit2).squeeze(-1) * p_ge1
            probs = torch.stack([1 - p_ge1, p_ge1 - p_ge2, p_ge2], dim=-1)
            out.append(probs.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def load_test_split(cfg: dict, tokenizer):
    data = json.loads(Path(cfg["training_data_path"]).read_text())
    splits = json.loads(Path(cfg["splits_path"]).read_text())
    test_rows = [r for r in data if r["row_num"] in set(splits["test"])]
    def seg_a(r):
        sp = r.get("signing_party", "")
        return r["clause_type"] + (" | signing party: " + sp if sp else "")

    enc = tokenizer(
        [seg_a(r) for r in test_rows],
        [r["clause_text"] for r in test_rows],
        padding="max_length",
        truncation=True,
        max_length=cfg["max_length"],
    )
    ds = Dataset.from_dict({
        **enc,
        "labels": [r["soft_label"] for r in test_rows],
        "row_num": [r["row_num"] for r in test_rows],
    })
    return ds, test_rows


def softmax_probs(model, ds, device, batch_size: int = 32) -> np.ndarray:
    model.eval()
    keep_cols = [c for c in ds.column_names if c not in ("labels", "row_num")]
    loader = DataLoader(
        ds.select_columns(keep_cols).with_format("torch"),
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            out.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics_block(preds: np.ndarray, true: np.ndarray, hard_mask: np.ndarray):
    macro = f1_score(true, preds, average="macro", labels=[0, 1, 2], zero_division=0)
    acc = accuracy_score(true, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], zero_division=0
    )
    hard_macro = f1_score(
        true[hard_mask], preds[hard_mask],
        average="macro", labels=[0, 1, 2], zero_division=0,
    )
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "per_class": {
            lbl: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
            }
            for i, lbl in enumerate(LABEL_NAMES)
        },
        "hard_only_macro_f1": float(hard_macro),
        "n": int(len(true)),
        "n_hard": int(hard_mask.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dirs", nargs="+", help="Run output dirs (each has best_model/)")
    ap.add_argument("--label", default=None, help="Label for the ensemble in output")
    ap.add_argument("--config", default="configs/stage3_config.yaml")
    ap.add_argument("--save", default=None, help="Optional output JSON path")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    cfg_raw = yaml.safe_load(Path(args.config).read_text())
    cfg = cfg_raw.get("risk_classifier", cfg_raw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use first model's tokenizer (same base across all runs)
    first_dir = Path(args.model_dirs[0])
    tok_dir = first_dir / "final" if (first_dir / "final").exists() else first_dir
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    test_ds, _ = load_test_split(cfg, tokenizer)

    soft_targets = np.array(test_ds["labels"])
    true = np.argmax(soft_targets, axis=-1)
    hard_mask = np.array([max(sl) >= 0.99 for sl in test_ds["labels"]])

    # Per-model probs for ensembling and individual reporting
    all_probs = []
    per_model = {}
    for mdir in args.model_dirs:
        mp = Path(mdir)
        corn = is_corn_model(mp)
        print(f"Loading {mp.name} ({'CORN' if corn else 'CE/SORD/hybrid'}) ... ", end="", flush=True)
        model = load_model(mp, cfg, device)
        probs = corn_probs(model, test_ds, device, args.batch_size) if corn \
            else softmax_probs(model, test_ds, device, args.batch_size)
        all_probs.append(probs)
        ind_preds = np.argmax(probs, axis=-1)
        per_model[mp.name] = metrics_block(ind_preds, true, hard_mask)
        print(f"single macro_f1={per_model[mp.name]['macro_f1']:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Uniform average of softmax probs
    ens_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    ens_preds = np.argmax(ens_probs, axis=-1)
    ens = metrics_block(ens_preds, true, hard_mask)

    label = args.label or f"ensemble-{len(args.model_dirs)}"
    print()
    print(f"=== {label} ({len(args.model_dirs)} models) ===")
    pc = ens["per_class"]
    print(
        f"macro_f1={ens['macro_f1']:.4f}  acc={ens['accuracy']:.4f}  "
        f"hard-only={ens['hard_only_macro_f1']:.4f}"
    )
    for c in ("LOW", "MEDIUM", "HIGH"):
        print(
            f"  {c:<7} P={pc[c]['precision']:.3f}  R={pc[c]['recall']:.3f}  "
            f"F1={pc[c]['f1']:.3f}"
        )

    if args.save:
        out = {
            "label": label,
            "members": [Path(d).name for d in args.model_dirs],
            "ensemble": ens,
            "per_model": per_model,
        }
        Path(args.save).write_text(json.dumps(out, indent=2))
        print(f"\nSaved → {args.save}")


if __name__ == "__main__":
    main()
