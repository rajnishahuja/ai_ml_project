"""
build_splits.py
===============
Builds reproducible train/val/test splits for the Stage 3 risk classifier.

Constraints:
  - Group by `contract`    → no leakage (all clauses from one contract stay in one split)
  - Stratify by clause_type → every clause type represented in every split
                              (important for rare types like Source Code Escrow, n=13)

Target ratio: 80/10/10.

Strategy:
  1. Use StratifiedGroupKFold(n_splits=10) → take 1 fold as TEST (~10%)
  2. On the remaining 90%, use StratifiedGroupKFold(n_splits=9) → take 1 fold as VAL (~10%)
  3. Remaining → TRAIN (~80%)

Output: data/processed/splits.json
  {
    "metadata": {...},
    "train": [row_num, ...],
    "val":   [row_num, ...],
    "test":  [row_num, ...]
  }

Usage:
    python scripts/build_splits.py
    python scripts/build_splits.py --seed 123
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DATASET_PATH = Path("data/processed/training_dataset.json")
SPLITS_PATH  = Path("data/processed/splits.json")

TARGET_RATIOS = {"train": 0.80, "val": 0.10, "test": 0.10}


def build_splits(seed: int = 100) -> dict:
    data = json.loads(DATASET_PATH.read_text())
    n = len(data)
    logger.info(f"Loaded {n} rows from {DATASET_PATH}")

    row_nums    = np.array([r["row_num"]     for r in data])
    clause_type = np.array([r["clause_type"] for r in data])
    contract    = np.array([r["contract"]    for r in data])

    # Step 1: carve out TEST (1 of 10 folds ≈ 10%)
    sgkf_test = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf_test.split(np.zeros(n), y=clause_type, groups=contract))

    remaining_mask = np.ones(n, dtype=bool)
    remaining_mask[test_idx] = False
    remaining_idx = np.where(remaining_mask)[0]

    # Step 2: carve out VAL from the remainder (1 of 9 ≈ 11% of remainder ≈ 10% overall)
    sgkf_val = StratifiedGroupKFold(n_splits=9, shuffle=True, random_state=seed)
    _, val_rel_idx = next(sgkf_val.split(
        np.zeros(len(remaining_idx)),
        y=clause_type[remaining_idx],
        groups=contract[remaining_idx],
    ))
    val_idx = remaining_idx[val_rel_idx]

    # Step 3: TRAIN = remaining - val
    train_mask = remaining_mask.copy()
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]

    # Sanity: no overlap, full coverage
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx)   & set(test_idx)) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == n

    # Sanity: no contract overlap across splits
    train_contracts = set(contract[train_idx])
    val_contracts   = set(contract[val_idx])
    test_contracts  = set(contract[test_idx])
    assert len(train_contracts & val_contracts) == 0, "contract leakage: train ∩ val"
    assert len(train_contracts & test_contracts) == 0, "contract leakage: train ∩ test"
    assert len(val_contracts   & test_contracts) == 0, "contract leakage: val ∩ test"

    splits = {
        "train": sorted(int(row_nums[i]) for i in train_idx),
        "val":   sorted(int(row_nums[i]) for i in val_idx),
        "test":  sorted(int(row_nums[i]) for i in test_idx),
    }

    # Verification block — print per-split stats
    print_verification(data, splits, train_idx, val_idx, test_idx, clause_type, contract)

    return {
        "metadata": {
            "seed": seed,
            "source": str(DATASET_PATH),
            "total_rows": n,
            "target_ratios": TARGET_RATIOS,
            "actual_ratios": {
                "train": round(len(train_idx)/n, 4),
                "val":   round(len(val_idx)/n, 4),
                "test":  round(len(test_idx)/n, 4),
            },
            "contracts_per_split": {
                "train": len(train_contracts),
                "val":   len(val_contracts),
                "test":  len(test_contracts),
            },
            "strategy": "StratifiedGroupKFold — group by contract, stratify by clause_type",
        },
        **splits,
    }


def print_verification(data, splits, train_idx, val_idx, test_idx, clause_type, contract):
    n = len(data)
    logger.info("=" * 60)
    logger.info(f"Total:  {n}")
    logger.info(f"Train:  {len(train_idx):>5} rows ({len(train_idx)/n*100:.2f}%)  "
                f"{len(set(contract[train_idx])):>4} contracts")
    logger.info(f"Val:    {len(val_idx):>5} rows ({len(val_idx)/n*100:.2f}%)  "
                f"{len(set(contract[val_idx])):>4} contracts")
    logger.info(f"Test:   {len(test_idx):>5} rows ({len(test_idx)/n*100:.2f}%)  "
                f"{len(set(contract[test_idx])):>4} contracts")
    logger.info("=" * 60)

    # Label distribution per split
    labels = np.array([r.get("label") or "SOFT" for r in data])
    logger.info("Label distribution per split:")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        dist = Counter(labels[idx])
        total = sum(dist.values())
        summary = "  ".join(f"{k}={v} ({v/total*100:.1f}%)" for k,v in sorted(dist.items()))
        logger.info(f"  {name:<6} {summary}")

    # Clause-type coverage: are all 36 types present in every split?
    logger.info("Clause-type coverage:")
    all_types = set(clause_type)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        present = set(clause_type[idx])
        missing = all_types - present
        logger.info(f"  {name:<6} {len(present)}/{len(all_types)} types" +
                    (f" — MISSING: {sorted(missing)}" if missing else ""))

    # Worst-case: which rare types have only 1 example in val or test?
    logger.info("Rare-type presence in val/test (types with <5 rows in either):")
    type_in_split = {
        "val":  Counter(clause_type[val_idx]),
        "test": Counter(clause_type[test_idx]),
    }
    rows = []
    for ct in all_types:
        v = type_in_split["val"].get(ct, 0)
        t = type_in_split["test"].get(ct, 0)
        if v < 5 or t < 5:
            total = int((clause_type == ct).sum())
            rows.append((ct, total, v, t))
    rows.sort(key=lambda x: x[1])
    for ct, total, v, t in rows:
        logger.info(f"  {ct:<40} total={total:>3}  val={v}  test={t}")


def main(seed: int = 42):
    result = build_splits(seed=seed)
    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLITS_PATH.write_text(json.dumps(result, indent=2))
    logger.info(f"Saved to {SPLITS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=100,
                        help="Random seed. Default=100 was chosen after scanning [0,1,7,13,17,21,42,100,123,777] "
                             "as the seed giving full 36/36 clause-type coverage in both val and test.")
    args = parser.parse_args()
    main(seed=args.seed)
