"""
train.py — Stage 3 DeBERTa-v3-base risk classifier.

Trains the Stage 3 risk classifier per configs/stage3_config.yaml using HuggingFace
Trainer + bf16 + soft-target CE with class weights. Evaluates on held-out test set
automatically at the end.

Design references:
- ARCHITECTURE.md "Stage 3 Training Data Pipeline" and "Loss Function" sections
- docs/STAGE3_TRAINING_NOTES.md §§7-9 (loss, hyperparams, eval metrics)
- memory checklist: project_stage3_training_checklist.md

Usage:
    /home/ubuntu/miniconda3/envs/rajnish-env/bin/python3 -m src.stage3_risk_agent.train
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import transformers
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DefaultDataCollator,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

# Cut HF HTTP/loading-progress chatter from training.log; keep WARN+ for real issues.
transformers.logging.set_verbosity_error()


# -------- constants -----------------------------------------------------------

LABEL_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LABEL_NAMES = ["LOW", "MEDIUM", "HIGH"]
NUM_LABELS = 3
CONFIG_PATH = "configs/stage3_config.yaml"
MASTER_CSV_PATH = "data/review/master_label_review.csv"


# -------- config --------------------------------------------------------------

@dataclass
class Stage3Config:
    model_name: str
    output_dir: str
    training_data_path: str
    splits_path: str
    max_length: int
    class_weights_method: str
    soft_label_weighting: str
    fine_tuning: str
    llrd: bool
    batch_size: int
    learning_rate: float
    warmup_ratio: float
    lr_scheduler_type: str
    epochs: int
    weight_decay: float
    early_stopping_patience: int
    metric_for_best_model: str
    precision: str
    allow_fp32_fallback: bool
    seed: int
    strict_determinism: bool
    llrd_decay: float = 0.9   # only used when llrd: true; LR_layer = base_lr * decay^(top - layer_idx)
    dropout: float = 0.1      # DeBERTa default; override to test added regularization

    @classmethod
    def from_yaml(cls, path: str) -> "Stage3Config":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["risk_classifier"]
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in names})


# -------- loss ----------------------------------------------------------------

def soft_target_ce(logits, targets, class_weights):
    """Weighted cross-entropy with soft (probability-vector) targets.

    For one-hot targets → standard weighted CE on the true class.
    For soft targets   → distributes across classes per the target vector.
    See STAGE3_TRAINING_NOTES.md §7.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    per_class = class_weights.to(log_probs.device) * targets * log_probs
    return -per_class.sum(dim=-1).mean()


def emd_loss(logits, targets, class_weights):
    """Earth Mover's Distance (Wasserstein-1) loss for ORDINAL classes.

    Treats predictions and targets as 1D distributions over [LOW, MED, HIGH].
    Penalizes 'off by k' proportionally to k:
      truth=HIGH, pred=MED  → EMD = 1
      truth=HIGH, pred=LOW  → EMD = 2 (twice as bad)
    CE would penalize both equally; EMD respects the ordering.

    For K classes, EMD = sum_{k=0..K-2} | CDF_pred(k) - CDF_target(k) |.

    Class weights are applied per-row using the expected class weight under
    the target distribution: weight_row = sum_c (target[c] * class_weight[c]).

    NOTE: Pure EMD on imbalanced data can collapse to predicting the median
    class (Run 12 demonstrated this). Use `hybrid_ce_emd` for production runs.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_target = torch.cumsum(targets, dim=-1)
    # Drop last cumulative (always = 1, contributes 0 to abs diff).
    emd_per_row = torch.abs(cdf_pred[..., :-1] - cdf_target[..., :-1]).sum(dim=-1)
    weight_per_row = (targets * class_weights.to(probs.device)).sum(dim=-1)
    return (emd_per_row * weight_per_row).mean()


def emd_loss_unweighted(logits, targets):
    """Unweighted EMD — used as the additive term in hybrid CE+EMD."""
    probs = torch.softmax(logits.float(), dim=-1)
    cdf_pred = torch.cumsum(probs, dim=-1)
    cdf_target = torch.cumsum(targets, dim=-1)
    return torch.abs(cdf_pred[..., :-1] - cdf_target[..., :-1]).sum(dim=-1).mean()


def hybrid_ce_emd(logits, targets, class_weights, lam=0.5):
    """CE (class-weighted) + lam * EMD (unweighted).

    The CE term provides class-discriminative pressure (anchors the model away
    from the median-collapse trap that pure EMD falls into on imbalanced data).
    The EMD term adds ordinal awareness — gradients near LOW↔HIGH boundaries
    push harder than near LOW↔MED or MED↔HIGH.
    """
    return soft_target_ce(logits, targets, class_weights) + lam * emd_loss_unweighted(logits, targets)


# -------- class weights -------------------------------------------------------

def compute_class_weights(train_ds: Dataset, method: str) -> torch.Tensor:
    """Option A: hard-row counts only. weight = N / (K * count_c)."""
    if method != "hard_counts":
        raise ValueError(f"Unsupported class_weights_method: {method!r}")
    counts = [0, 0, 0]
    for sl in train_ds["labels"]:
        if max(sl) >= 0.99:  # one-hot row
            counts[sl.index(max(sl))] += 1
    N = sum(counts)
    return torch.tensor(
        [N / (NUM_LABELS * c) if c > 0 else 1.0 for c in counts],
        dtype=torch.float32,
    )


# -------- dataset prep --------------------------------------------------------

def transform_train_labels(rows, mode: str):
    """Apply label-mode transformation to training rows. Val/test are NEVER transformed.

    - 'soft' (default): no change — soft vectors used as-is for soft-target CE.
    - 'hard_only': filter out SOFT_LABEL rows (where max(soft_label) < 0.99).
                    Tests whether the labeler-disagreement rows are net helpful or harmful.
    - 'argmax_soft': convert soft vectors → one-hot via argmax. Keeps the rows but discards
                    the uncertainty info. Tests whether the SOFT vector format is the issue
                    (vs the MEDIUM anchor in the underlying labels).
    """
    if mode == "soft":
        return rows
    if mode == "hard_only":
        return [r for r in rows if max(r["soft_label"]) >= 0.99]
    if mode == "argmax_soft":
        out = []
        for r in rows:
            r2 = dict(r)
            sl = r["soft_label"]
            idx = sl.index(max(sl))
            r2["soft_label"] = [1.0 if i == idx else 0.0 for i in range(NUM_LABELS)]
            out.append(r2)
        return out
    raise ValueError(f"Unknown label-mode: {mode!r}")


def build_datasets(cfg: Stage3Config, tokenizer, label_mode: str = "soft"):
    data = json.loads(Path(cfg.training_data_path).read_text(encoding="utf-8"))
    splits = json.loads(Path(cfg.splits_path).read_text(encoding="utf-8"))

    def tokenize_split(row_nums, transform: bool = False):
        rows = [r for r in data if r["row_num"] in row_nums]
        if transform:
            rows = transform_train_labels(rows, label_mode)
        enc = tokenizer(
            [r["clause_type"] for r in rows],
            [r["clause_text"] for r in rows],
            padding="max_length",
            truncation=True,
            max_length=cfg.max_length,
        )
        return Dataset.from_dict({
            **enc,
            "labels": [r["soft_label"] for r in rows],
            "row_num": [r["row_num"] for r in rows],
        })

    return (
        tokenize_split(set(splits["train"]), transform=True),
        tokenize_split(set(splits["val"])),
        tokenize_split(set(splits["test"])),
    )


# -------- LLRD optimizer (Section D.0 / ablation) -----------------------------

def build_llrd_param_groups(model, base_lr: float, weight_decay: float, decay: float):
    """Layer-wise LR decay parameter groups for DeBERTa-v3.

    Top encoder layer (layer_{n-1}) gets base_lr; each lower layer multiplies by `decay`.
    Embeddings + rel_embeddings + encoder-level LayerNorm get the lowest LR (base_lr * decay^n).
    Classifier head + pooler stay at base_lr (newly initialized, need full speed).
    Bias / LayerNorm weights get weight_decay=0 (HF Trainer convention).
    """
    no_decay_keys = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    n_layers = model.config.num_hidden_layers
    embedding_lr = base_lr * (decay ** n_layers)

    groups = []
    seen = set()

    def add(params, lr_val, wd_val):
        if params:
            groups.append({"params": params, "lr": lr_val, "weight_decay": wd_val})

    # 1. Classifier + pooler — full LR (newly initialized layers)
    head_d, head_nd = [], []
    for n, p in model.named_parameters():
        if n.startswith(("classifier.", "pooler.")):
            seen.add(n)
            (head_nd if any(k in n for k in no_decay_keys) else head_d).append(p)
    add(head_d, base_lr, weight_decay)
    add(head_nd, base_lr, 0.0)

    # 2. Encoder layers — top-down geometric decay
    for layer_i in range(n_layers):
        layer_lr = base_lr * (decay ** (n_layers - 1 - layer_i))
        l_d, l_nd = [], []
        prefix = f"deberta.encoder.layer.{layer_i}."
        for n, p in model.named_parameters():
            if n.startswith(prefix):
                seen.add(n)
                (l_nd if any(k in n for k in no_decay_keys) else l_d).append(p)
        add(l_d, layer_lr, weight_decay)
        add(l_nd, layer_lr, 0.0)

    # 3. Embeddings + rel_embeddings + encoder.LayerNorm — lowest LR
    emb_d, emb_nd = [], []
    for n, p in model.named_parameters():
        if n in seen:
            continue
        if n.startswith(("deberta.embeddings.",
                          "deberta.encoder.rel_embeddings",
                          "deberta.encoder.LayerNorm")):
            seen.add(n)
            (emb_nd if any(k in n for k in no_decay_keys) else emb_d).append(p)
    add(emb_d, embedding_lr, weight_decay)
    add(emb_nd, embedding_lr, 0.0)

    # Sanity check
    missing = {n for n, _ in model.named_parameters()} - seen
    if missing:
        raise RuntimeError(f"LLRD param groups missing: {missing}")

    return groups


# -------- trainer subclass ----------------------------------------------------

MODEL_INPUT_KEYS = {"input_ids", "attention_mask", "token_type_ids"}


class SoftTargetCETrainer(Trainer):
    """Overrides compute_loss with our weighted soft-target CE.
    If llrd_decay is provided, also overrides create_optimizer to build per-layer LR groups.
    """

    def __init__(self, *args, class_weights, llrd_decay=None, loss_type="ce",
                 emd_lambda=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights
        self._llrd_decay = llrd_decay
        self._loss_type = loss_type
        self._emd_lambda = emd_lambda

    def create_optimizer(self):
        if self.optimizer is None and self._llrd_decay is not None:
            param_groups = build_llrd_param_groups(
                self.model,
                self.args.learning_rate,
                self.args.weight_decay,
                self._llrd_decay,
            )
            self.optimizer = torch.optim.AdamW(param_groups)
            return self.optimizer
        return super().create_optimizer()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move class weights to model device on first use
        if self._class_weights.device != model.device:
            self._class_weights = self._class_weights.to(model.device)

        targets = inputs["labels"]   # keep in inputs so prediction_step can still extract
        model_inputs = {k: v for k, v in inputs.items() if k in MODEL_INPUT_KEYS}
        outputs = model(**model_inputs)
        if self._loss_type == "emd":
            loss = emd_loss(outputs.logits, targets, self._class_weights)
        elif self._loss_type == "hybrid":
            loss = hybrid_ce_emd(outputs.logits, targets, self._class_weights,
                                  lam=self._emd_lambda)
        else:
            loss = soft_target_ce(outputs.logits, targets, self._class_weights)
        return (loss, outputs) if return_outputs else loss


# -------- per-epoch validation metrics (Tier A) ------------------------------

def compute_val_metrics(eval_pred):
    preds = np.argmax(eval_pred.predictions, axis=-1)
    true = np.argmax(eval_pred.label_ids, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], zero_division=0
    )
    out = {
        "val_macro_f1": float(f1_score(true, preds, average="macro",
                                       labels=[0, 1, 2], zero_division=0)),
        "val_accuracy": float(accuracy_score(true, preds)),
    }
    for i, lbl in enumerate(LABEL_NAMES):
        out[f"val_f1_{lbl}"] = float(f1[i])
        out[f"val_p_{lbl}"] = float(p[i])
        out[f"val_r_{lbl}"] = float(r[i])
    return out


# -------- callbacks -----------------------------------------------------------

class NaNDetector(TrainerCallback):
    """Fail loudly on first NaN/Inf training loss (belt-and-suspenders)."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        L = logs["loss"]
        if isinstance(L, (int, float)) and (L != L or abs(L) == float("inf")):
            raise ValueError(f"NaN/Inf loss at step {state.global_step}: {L}")


# -------- test-set evaluation (Tier B) ---------------------------------------

def evaluate_on_test(trainer, test_ds: Dataset, cfg: Stage3Config, logger):
    """Full test-set evaluation per STAGE3_TRAINING_NOTES.md §9.
    Writes test_metrics.json to cfg.output_dir. Calibration is handled by a
    separate post-training script — not included here.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TEST SET EVALUATION")
    logger.info("=" * 70)

    out = trainer.predict(test_ds)
    logits, soft_targets = out.predictions, out.label_ids
    preds = np.argmax(logits, axis=-1)
    true = np.argmax(soft_targets, axis=-1)

    # Overall metrics
    macro_f1 = f1_score(true, preds, average="macro", labels=[0, 1, 2], zero_division=0)
    p, r, f1, _ = precision_recall_fscore_support(
        true, preds, labels=[0, 1, 2], zero_division=0
    )
    acc = accuracy_score(true, preds)
    logger.info(f"Overall (n={len(true)}): macro_f1={macro_f1:.4f}  accuracy={acc:.4f}")
    for i, lbl in enumerate(LABEL_NAMES):
        logger.info(f"  {lbl}: P={p[i]:.4f}  R={r[i]:.4f}  F1={f1[i]:.4f}")

    # Per-clause-type F1
    data = json.loads(Path(cfg.training_data_path).read_text(encoding="utf-8"))
    row_to_type = {row["row_num"]: row["clause_type"] for row in data}
    types = np.array([row_to_type[rn] for rn in test_ds["row_num"]])

    logger.info("\nPer-clause-type F1:")
    per_type = {}
    for ct in sorted(set(types)):
        mask = types == ct
        n = int(mask.sum())
        tF1 = f1_score(
            true[mask], preds[mask], average="macro",
            labels=[0, 1, 2], zero_division=0,
        )
        caveat = "  (noisy, n<5)" if n < 5 else ""
        logger.info(f"  {ct:<40} n={n:>3}  macro_f1={tF1:.3f}{caveat}")
        per_type[ct] = {"n": n, "macro_f1": float(tF1)}

    # Baselines on hard-labeled test subset only
    hard_mask = np.array([max(sl) >= 0.99 for sl in test_ds["labels"]])
    n_hard = int(hard_mask.sum())
    logger.info(f"\nBaselines (on {n_hard} hard-labeled test rows):")

    baselines = {}
    # Majority class (always predict LOW)
    baselines["majority_LOW"] = float(f1_score(
        true[hard_mask], np.zeros(n_hard, dtype=int),
        average="macro", labels=[0, 1, 2], zero_division=0,
    ))

    # Qwen-only and Gemini-only: look up original labeler outputs from master CSV
    row_to_qwen, row_to_gemini = {}, {}
    with open(MASTER_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rn = int(row["row_num"])
            if row["qwen_label"] in LABEL_TO_INT:
                row_to_qwen[rn] = LABEL_TO_INT[row["qwen_label"]]
            if row["gemini_label"] in LABEL_TO_INT:
                row_to_gemini[rn] = LABEL_TO_INT[row["gemini_label"]]

    hard_row_nums = np.array(test_ds["row_num"])[hard_mask]
    for name, lookup in [("qwen_only", row_to_qwen), ("gemini_only", row_to_gemini)]:
        have = np.array([rn in lookup for rn in hard_row_nums])
        if have.sum() == 0:
            baselines[name] = None
            continue
        baseline_preds = np.array([lookup[rn] for rn in hard_row_nums[have]])
        baseline_true = true[hard_mask][have]
        baselines[name] = float(f1_score(
            baseline_true, baseline_preds,
            average="macro", labels=[0, 1, 2], zero_division=0,
        ))

    for k, v in baselines.items():
        logger.info(f"  {k:<20} macro_f1={v:.4f}" if v is not None
                    else f"  {k:<20} no labeler coverage")

    our_hard = f1_score(
        true[hard_mask], preds[hard_mask],
        average="macro", labels=[0, 1, 2], zero_division=0,
    )
    logger.info(f"  our model (hard only) macro_f1={our_hard:.4f}")

    # Persist
    (Path(cfg.output_dir) / "test_metrics.json").write_text(json.dumps({
        "overall": {
            "n": int(len(true)),
            "macro_f1": float(macro_f1),
            "accuracy": float(acc),
            "per_class": {
                lbl: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i])}
                for i, lbl in enumerate(LABEL_NAMES)
            },
        },
        "per_clause_type": per_type,
        "baselines": baselines,
        "our_model_hard_only_macro_f1": float(our_hard),
        "n_test_hard_labeled": n_hard,
    }, indent=2))
    logger.info(f"\nTest metrics saved to {cfg.output_dir}/test_metrics.json")


# -------- main ---------------------------------------------------------------

def parse_cli():
    p = argparse.ArgumentParser(description="Stage 3 risk classifier training")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override num_train_epochs with a fixed step count "
                        "(used for smoke runs; default = full epochs)")
    p.add_argument("--eval-steps", type=int, default=None,
                   help="If set (typically with --max-steps), run val eval every N steps "
                        "instead of once per epoch")
    p.add_argument("--output-suffix", type=str, default=None,
                   help="Append suffix to output_dir (e.g. '_smoke30') so test runs "
                        "don't overwrite full-training artifacts")
    p.add_argument("--seed", type=int, default=None,
                   help="Override training seed from config (for multi-seed runs)")
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "emd", "hybrid"],
                   help="Loss function: 'ce' (CE), 'emd' (pure EMD, may collapse), "
                        "or 'hybrid' (CE + lambda*EMD; recommended for ordinal)")
    p.add_argument("--emd-lambda", type=float, default=0.5,
                   help="Weight for the EMD term in hybrid loss (default 0.5)")
    p.add_argument("--label-mode", type=str, default="soft",
                   choices=["soft", "hard_only", "argmax_soft"],
                   help="How to treat SOFT_LABEL rows in training: keep soft (default), "
                        "drop them (hard_only), or hard-argmax the soft vectors (argmax_soft). "
                        "Val/test are never transformed.")
    return p.parse_args()


def main():
    cli = parse_cli()
    cfg = Stage3Config.from_yaml(CONFIG_PATH)
    if cli.output_suffix:
        cfg.output_dir = cfg.output_dir + cli.output_suffix
    if cli.seed is not None:
        cfg.seed = cli.seed
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(cfg.output_dir) / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 70)
    logger.info("STAGE 3 RISK CLASSIFIER — TRAINING")
    logger.info("=" * 70)

    set_seed(cfg.seed)

    logger.info(f"Loading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds, val_ds, test_ds = build_datasets(cfg, tokenizer, label_mode=cli.label_mode)
    logger.info(f"Label mode: {cli.label_mode}")
    logger.info(f"Datasets — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")

    class_weights = compute_class_weights(train_ds, cfg.class_weights_method)
    logger.info(f"Class weights ({cfg.class_weights_method}): "
                f"LOW={class_weights[0]:.4f}  MED={class_weights[1]:.4f}  HIGH={class_weights[2]:.4f}")

    logger.info(f"Loading model in {cfg.precision}: {cfg.model_name}")
    if cfg.dropout != 0.1:
        logger.info(f"Dropout override: hidden_dropout_prob={cfg.dropout}, "
                    f"attention_probs_dropout_prob={cfg.dropout}")
    dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=NUM_LABELS,
        dtype=dtype,
        hidden_dropout_prob=cfg.dropout,
        attention_probs_dropout_prob=cfg.dropout,
    )

    # Full-epoch defaults, overridden by CLI flags for smoke runs
    eval_strategy = "epoch"
    save_strategy = "epoch"
    eval_steps = None
    save_steps = None
    max_steps = -1
    num_train_epochs = cfg.epochs
    if cli.max_steps is not None:
        max_steps = cli.max_steps
        num_train_epochs = 1  # ignored when max_steps > 0; set defensively
        logger.info(f"SMOKE RUN: capping training at max_steps={max_steps}")
    if cli.eval_steps is not None:
        eval_strategy = "steps"
        save_strategy = "steps"
        eval_steps = cli.eval_steps
        save_steps = cli.eval_steps
        logger.info(f"SMOKE RUN: evaluating every {eval_steps} steps")

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        bf16=(cfg.precision == "bf16"),
        fp16=False,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=10 if cli.max_steps else 20,
        seed=cfg.seed,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    if cfg.llrd:
        logger.info(f"LLRD enabled (decay={cfg.llrd_decay}): "
                    f"top-layer LR={cfg.learning_rate:.2e}, "
                    f"embedding LR={cfg.learning_rate * (cfg.llrd_decay ** 12):.2e}")
    if cli.loss == "hybrid":
        logger.info(f"Loss function: hybrid CE + {cli.emd_lambda} * EMD")
    else:
        logger.info(f"Loss function: {cli.loss}")

    trainer = SoftTargetCETrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DefaultDataCollator(),
        compute_metrics=compute_val_metrics,
        class_weights=class_weights,
        llrd_decay=cfg.llrd_decay if cfg.llrd else None,
        loss_type=cli.loss,
        emd_lambda=cli.emd_lambda,
        callbacks=[
            NaNDetector(),
            EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience),
        ],
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    final_dir = Path(cfg.output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    logger.info(f"Best model + tokenizer saved to {final_dir}")

    evaluate_on_test(trainer, test_ds, cfg, logger)


if __name__ == "__main__":
    main()
