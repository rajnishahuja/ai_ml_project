"""
prepare_relabel_batch.py
========================
Formats a batch of SOFT_LABEL rows for Claude to relabel in-session.

Each row is enriched with:
  - Document name + signing party (parties metadata)
  - Expanded clause type name for acronym/confusable types
  - Per-clause-type calibration examples (1 per risk level, from AGREED rows,
    shown ONCE per clause type in the batch — not repeated per row)
  - Both Qwen and Gemini's original labels + reasoning

Usage:
    python scripts/prepare_relabel_batch.py --batch 0
    python scripts/prepare_relabel_batch.py --batch 0 --batch-size 50
    python scripts/prepare_relabel_batch.py --status
    python scripts/prepare_relabel_batch.py --reset-batch 0   # redo a batch
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

MASTER_CSV    = Path("data/review/master_label_review.csv")
SPLITS_JSON   = Path("data/processed/splits.json")
RESULTS_DIR   = Path("data/review/relabel_batches")
PROGRESS_FILE = RESULTS_DIR / "progress.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Clause types whose names are acronyms or easily confused — expand inline.
# Only types where the name alone genuinely misleads (checked against real examples).
EXPANDED_NAMES = {
    "Rofr/Rofo/Rofn": (
        "Rofr/Rofo/Rofn (Right of First Refusal / First Offer / First Negotiation)"
    ),
    "Affiliate License-Licensee": (
        "Affiliate License-Licensee (license grant TO the licensee's affiliates)"
    ),
    "Affiliate License-Licensor": (
        "Affiliate License-Licensor (license grant BY or including IP of the licensor's affiliates)"
    ),
    "Ip Ownership Assignment": (
        "Ip Ownership Assignment (IP created by one party becomes the other party's property)"
    ),
    "Cap On Liability": (
        "Cap On Liability (includes monetary caps AND time limits on bringing claims)"
    ),
}


def load_metadata(rows):
    meta = defaultdict(dict)
    for r in rows:
        if r["category"] == "METADATA":
            meta[r["contract"]][r["clause_type"]] = r["clause_text"].strip()
    return meta


def load_soft_rows(rows, splits):
    train_ids = set(splits["train"])
    val_ids   = set(splits["val"])
    test_ids  = set(splits["test"])

    soft = []
    for r in rows:
        if r["category"] != "SOFT_LABEL":
            continue
        rn = int(r["row_num"])
        if rn in train_ids:
            split = "train"
        elif rn in val_ids:
            split = "val"
        elif rn in test_ids:
            split = "test"
        else:
            split = "unknown"
        r["_split"] = split
        soft.append(r)

    soft.sort(key=lambda r: int(r["row_num"]))
    return soft


def build_agreed_examples(rows):
    """
    For each clause type, find the shortest AGREED example per risk level.
    Minimum 30 chars, truncated to 120 chars.
    Returns: {clause_type: {"LOW": text, "MEDIUM": text, "HIGH": text}}
    """
    pool = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["category"] != "AGREED":
            continue
        if r["qwen_label"] != r["gemini_label"]:
            continue
        label = r["qwen_label"]
        if label not in ("LOW", "MEDIUM", "HIGH"):
            continue
        pool[r["clause_type"]][label].append(r["clause_text"].strip())

    examples = {}
    for ct, by_label in pool.items():
        examples[ct] = {}
        for label in ("LOW", "MEDIUM", "HIGH"):
            candidates = [t for t in by_label.get(label, []) if len(t) >= 30]
            if not candidates:
                continue
            text = min(candidates, key=len)
            if len(text) > 120:
                text = text[:120].rsplit(" ", 1)[0] + "…"
            examples[ct][label] = text

    return examples


def format_batch(soft_rows, meta, agreed_examples, batch_idx, batch_size):
    start = batch_idx * batch_size
    end   = min(start + batch_size, len(soft_rows))
    batch = soft_rows[start:end]

    lines = []
    lines.append(
        f"# Relabeling batch {batch_idx} "
        f"— rows {start}–{end-1} of {len(soft_rows)} total"
    )
    lines.append("")
    lines.append(
        "Label each clause LOW / MEDIUM / HIGH from the perspective of the SIGNING PARTY "
        "(named in 'Signing party' below). They are reviewing this contract before signing "
        "and assessing whether each clause poses risk to them."
    )
    lines.append("")
    lines.append(
        "Qwen and Gemini disagreed on these rows. Read both arguments and the calibration "
        "examples, decide who is right (or whether both are wrong), and give a definitive label."
    )
    lines.append("")
    lines.append("OUTPUT: a single JSON array, one object per row, in the same order:")
    lines.append(
        '  [{"row_num": N, "label": "LOW"|"MEDIUM"|"HIGH", "reason": "...max 80 words..."}, ...]'
    )
    lines.append("Output ONLY the JSON array. No preamble, no markdown, no explanation.")
    lines.append("")
    lines.append("---")
    lines.append("")

    shown_calibration = set()

    for i, r in enumerate(batch):
        contract_meta = meta.get(r["contract"], {})
        doc_name      = contract_meta.get("Document Name", "Unknown")
        parties       = contract_meta.get("Parties", "Unknown")
        filing_co     = r["contract"].split("_")[0]
        clause_type   = r["clause_type"]
        display_type  = EXPANDED_NAMES.get(clause_type, clause_type)

        # Calibration block — shown once per clause type per batch
        if clause_type not in shown_calibration:
            shown_calibration.add(clause_type)
            ex = agreed_examples.get(clause_type, {})
            if ex:
                lines.append(
                    f"**{display_type} — calibration examples from this dataset:**"
                )
                for label in ("LOW", "MEDIUM", "HIGH"):
                    sample = ex.get(label)
                    if sample:
                        lines.append(f"  {label}: \"{sample}\"")
                    else:
                        lines.append(f"  {label}: (no example available)")
                lines.append("")

        lines.append(f"## Row {r['row_num']}  [{i+1}/{len(batch)}]")
        lines.append(f"Clause type:   {display_type}")
        lines.append(f"Document:      {doc_name}")
        lines.append(f"Drafter:       {filing_co}")
        lines.append(f"Signing party: {parties}")
        lines.append("")
        lines.append("**Clause text:**")
        lines.append(r["clause_text"].strip())
        lines.append("")
        lines.append(f"Qwen → **{r['qwen_label']}** (conf {r['qwen_confidence']})")
        if r["qwen_reason"].strip():
            lines.append(f"  {r['qwen_reason'].strip()}")
        lines.append("")
        lines.append(f"Gemini → **{r['gemini_label']}** (conf {r['gemini_confidence']})")
        if r["gemini_reason"].strip():
            lines.append(f"  {r['gemini_reason'].strip()}")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines), [int(r["row_num"]) for r in batch]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed_batches": [], "labeled_row_nums": []}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def show_status(soft_rows, batch_size):
    progress    = load_progress()
    done        = sorted(progress["completed_batches"])
    labeled     = len(progress["labeled_row_nums"])
    total       = len(soft_rows)
    n_batches   = (total + batch_size - 1) // batch_size

    print(f"Total soft-label rows : {total}")
    print(f"Batch size            : {batch_size}")
    print(f"Total batches         : {n_batches}")
    print(f"Completed batches     : {done}")
    print(f"Rows labeled          : {labeled} / {total}")
    print(f"Remaining             : {total - labeled}")

    if labeled < total:
        nxt = next((i for i in range(n_batches) if i not in done), None)
        if nxt is not None:
            print(f"\nNext batch            : --batch {nxt}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch",       type=int, default=0)
    parser.add_argument("--batch-size",  type=int, default=50)
    parser.add_argument("--status",      action="store_true")
    parser.add_argument("--reset-batch", type=int, default=None,
                        help="Remove a completed batch from progress so it can be rerun")
    args = parser.parse_args()

    with MASTER_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with SPLITS_JSON.open() as f:
        splits = json.load(f)

    meta            = load_metadata(rows)
    soft_rows       = load_soft_rows(rows, splits)
    agreed_examples = build_agreed_examples(rows)

    if args.reset_batch is not None:
        progress = load_progress()
        if args.reset_batch in progress["completed_batches"]:
            progress["completed_batches"].remove(args.reset_batch)
            # Remove the row_nums for this batch from labeled list
            n_batches = (len(soft_rows) + args.batch_size - 1) // args.batch_size
            start = args.reset_batch * args.batch_size
            end   = min(start + args.batch_size, len(soft_rows))
            batch_row_nums = {int(soft_rows[j]["row_num"]) for j in range(start, end)}
            progress["labeled_row_nums"] = [
                rn for rn in progress["labeled_row_nums"]
                if rn not in batch_row_nums
            ]
            save_progress(progress)
            print(f"Batch {args.reset_batch} removed from progress — ready to rerun.")
        else:
            print(f"Batch {args.reset_batch} was not marked complete.")
        return

    if args.status:
        show_status(soft_rows, args.batch_size)
        return

    n_batches = (len(soft_rows) + args.batch_size - 1) // args.batch_size
    if args.batch >= n_batches:
        print(f"Batch {args.batch} out of range — only {n_batches} batches total.")
        return

    progress = load_progress()
    if args.batch in progress["completed_batches"]:
        print(f"Batch {args.batch} already completed. Use --reset-batch {args.batch} to redo.")
        return

    text, row_nums = format_batch(
        soft_rows, meta, agreed_examples,
        batch_idx=args.batch, batch_size=args.batch_size
    )

    out_file = RESULTS_DIR / f"batch_{args.batch:03d}_input.txt"
    out_file.write_text(text, encoding="utf-8")

    print(text)
    print(f"\n[Saved to {out_file}]")
    print(f"[Rows in this batch: {row_nums[0]}–{row_nums[-1]}, count={len(row_nums)}]")
    print(
        f"[After labeling, run: "
        f"python scripts/save_relabel_results.py --batch {args.batch}]"
    )


if __name__ == "__main__":
    main()
