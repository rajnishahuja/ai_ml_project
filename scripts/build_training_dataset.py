"""
build_training_dataset.py
=========================
Builds the merged training dataset from master_label_review.csv.

Categories included (4,285 hard + 99 soft = 4,384 effective training rows):
  AGREED (2,735)                → hard label (qwen == gemini), one-hot soft vector
  MANUAL_REVIEW (235)           → hard label from human-filled final_label
  GEMINI_PRO_REVIEW (87)        → hard label from Gemini 2.5 Pro's final_label
  SOFT_LABEL_V2_AGREED (1,228)  → hard label from v2 3-way consensus (Qwen+Gemini+Claude)
  SOFT_LABEL (99)               → probability vector from qwen+gemini v1 outputs
                                   (all-3-disagree rows; dropped by --label-mode hard_only)

Categories excluded:
  METADATA (2,292)         → routes to report header, not risk-labeled
  ERROR (26)               → dropped (22 original labeling errors + 4 MANUAL_REVIEW fragments)

Label encoding: LOW=0, MEDIUM=1, HIGH=2

Soft-label weighting (for SOFT_LABEL rows):
  Default: confidence-weighted (qwen_confidence / gemini_confidence). Qwen's conf=0.0
  artifact (11.5% of its rows) is floored to 0.5 since the label is valid but the
  score is a known model quirk, not genuine uncertainty.
  --no_conf_weight flag falls back to uniform 50/50 weights.

Output: data/processed/training_dataset.json

Usage:
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --no_conf_weight  # equal weights for soft labels
"""

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MASTER_CSV   = Path("data/review/master_label_review.csv")
SPANS_PATH   = Path("data/processed/all_positive_spans.json")
OUTPUT_PATH  = Path("data/processed/training_dataset.json")

LABEL_TO_INT = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
INT_TO_LABEL = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}


def one_hot(label: str) -> list[float]:
    v = [0.0, 0.0, 0.0]
    v[LABEL_TO_INT[label]] = 1.0
    return v


def soft_vector(qwen_label: str, gemini_label: str,
                qwen_conf: float, gemini_conf: float,
                use_conf_weight: bool) -> list[float]:
    """Compute [LOW, MEDIUM, HIGH] probability vector for adjacent disagreements."""
    qi = LABEL_TO_INT[qwen_label]
    gi = LABEL_TO_INT[gemini_label]

    # Qwen conf=0.0 is a known model quirk (label valid, score artifact) — floor to 0.5
    if qwen_conf == 0.0:
        qwen_conf = 0.5

    if use_conf_weight and (qwen_conf + gemini_conf) > 0:
        w_q = qwen_conf / (qwen_conf + gemini_conf)
        w_g = gemini_conf / (qwen_conf + gemini_conf)
    else:
        w_q = w_g = 0.5

    v = [0.0, 0.0, 0.0]
    v[qi] += w_q
    v[gi] += w_g
    return v


def parse_conf(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def load_parties_lookup() -> dict:
    """Build contract → signing party span lookup from all_positive_spans.json."""
    spans = json.loads(SPANS_PATH.read_text(encoding="utf-8"))
    lookup = {}
    for s in spans:
        if s["clause_type"] == "Parties":
            lookup[s["contract"]] = s["clause_text"].strip()
    return lookup


def build(use_conf_weight: bool = True) -> list[dict]:
    with MASTER_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    logger.info(f"Loaded {len(rows)} rows from {MASTER_CSV}")

    parties = load_parties_lookup()
    logger.info(f"Loaded signing party spans for {len(parties)} contracts")

    dataset = []
    skipped = Counter()

    for r in rows:
        cat = r["category"]

        if cat in ("METADATA", "ERROR"):
            skipped[cat] += 1
            continue

        # Drop CUAD annotation-fragment rows. These are ground-truth answer spans
        # that are too short to carry meaningful clause content (e.g., "us.",
        # "state.", "corporation."). They slipped through the Qwen/Gemini
        # reconciliation because both labelers agreed on a cautious label, and
        # through manual review because the labelers didn't flip HIGH↔LOW.
        # See docs/STAGE3_TRAINING_NOTES.md "Fragment clauses" for detail.
        if len(r["clause_text"].strip()) < 15:
            skipped["short_fragment"] += 1
            continue

        base = {
            "row_num":        int(r["row_num"]),
            "id":             r["id"],
            "contract":       r["contract"],
            "clause_type":    r["clause_type"],
            "clause_text":    r["clause_text"],
            "signing_party":  parties.get(r["contract"], ""),
            "label_source":   cat,
        }

        if cat == "AGREED":
            label = r["qwen_label"]   # qwen == gemini for agreed rows
            conf  = (parse_conf(r["qwen_confidence"]) + parse_conf(r["gemini_confidence"])) / 2
            dataset.append({**base,
                "label":      label,
                "label_int":  LABEL_TO_INT[label],
                "soft_label": one_hot(label),
                "label_type": "hard",
                "confidence": round(conf, 4),
            })

        elif cat in ("MANUAL_REVIEW", "GEMINI_PRO_REVIEW"):
            label = r["final_label"]
            if label not in LABEL_TO_INT:
                logger.warning(f"Skipping {r['review_id']} ({cat}) — final_label={label!r}")
                skipped[f"{cat}_missing"] += 1
                continue
            dataset.append({**base,
                "label":      label,
                "label_int":  LABEL_TO_INT[label],
                "soft_label": one_hot(label),
                "label_type": "hard",
                "confidence": 1.0,
            })

        elif cat == "SOFT_LABEL_V2_AGREED":
            label = r["final_label"]
            if label not in LABEL_TO_INT:
                logger.warning(f"Skipping row_num={r['row_num']} (SOFT_LABEL_V2_AGREED) — final_label={label!r}")
                skipped["SOFT_LABEL_V2_AGREED_missing"] += 1
                continue
            dataset.append({**base,
                "label":      label,
                "label_int":  LABEL_TO_INT[label],
                "soft_label": one_hot(label),
                "label_type": "hard",
                "confidence": 0.9,   # slightly below 1.0 — consensus not unanimous in all cases
            })

        elif cat == "SOFT_LABEL":
            ql = r["qwen_label"]
            gl = r["gemini_label"]
            if ql not in LABEL_TO_INT or gl not in LABEL_TO_INT:
                skipped["SOFT_LABEL_invalid"] += 1
                continue
            qc = parse_conf(r["qwen_confidence"])
            gc = parse_conf(r["gemini_confidence"])
            sv = soft_vector(ql, gl, qc, gc, use_conf_weight)
            dataset.append({**base,
                "label":      None,
                "label_int":  None,
                "soft_label": [round(x, 4) for x in sv],
                "label_type": "soft",
                "confidence": round((qc + gc) / 2, 4),
            })

    return dataset, skipped


def main(use_conf_weight: bool = True):
    dataset, skipped = build(use_conf_weight)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")

    hard = [r for r in dataset if r["label_type"] == "hard"]
    soft = [r for r in dataset if r["label_type"] == "soft"]

    logger.info(f"Total rows: {len(dataset)} | Hard: {len(hard)} | Soft: {len(soft)}")
    logger.info(f"Skipped: {dict(skipped)}")
    logger.info(f"Hard label distribution: {dict(Counter(r['label'] for r in hard))}")

    soft_majority = [INT_TO_LABEL[r['soft_label'].index(max(r['soft_label']))] for r in soft]
    logger.info(f"Soft label majority distribution: {dict(Counter(soft_majority))}")

    logger.info(f"Saved to {OUTPUT_PATH}")

    # Sample soft label vectors
    logger.info("Sample soft label vectors:")
    from collections import defaultdict
    by_disagree = defaultdict(list)
    for r in soft:
        key = f"Qwen={r.get('qwen_label','')} Gemini={r.get('gemini_label','')}"
        by_disagree[key].append(r['soft_label'])

    with MASTER_CSV.open(encoding="utf-8") as f:
        master = {r["row_num"]: r for r in csv.DictReader(f)}

    seen = set()
    for r in soft[:200]:
        row = master.get(str(r["row_num"]), {})
        key = f"Qwen={row.get('qwen_label','')} Gemini={row.get('gemini_label','')}"
        if key not in seen:
            seen.add(key)
            logger.info(f"  {key} → {r['soft_label']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_conf_weight", action="store_true",
                        help="Use equal weights (0.5/0.5) instead of confidence weighting")
    args = parser.parse_args()
    main(use_conf_weight=not args.no_conf_weight)
