"""
run_gemini_pro_review.py
========================
Runs Gemini 2.5 Pro on 87 GEMINI_PRO_REVIEW rows in master_label_review.csv.
These are adjacent disagreements (LOW↔MEDIUM or MEDIUM↔HIGH) in three
ambiguous clause types: Uncapped Liability, Liquidated Damages,
Irrevocable Or Perpetual License.

Gemini 2.5 Pro acts as a tiebreaker, seeing both Qwen and Gemini Flash
reasoning before making its final call.

Usage:
    python scripts/run_gemini_pro_review.py
    python scripts/run_gemini_pro_review.py --dry_run    # print prompts, no API calls
    python scripts/run_gemini_pro_review.py --save_every 10
"""

import argparse
import csv
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MASTER_CSV = Path("data/review/master_label_review.csv")

REVIEW_PROMPT = """You are a senior legal risk analyst. A legal contract clause has been assessed by two AI models and they disagree. Your job is to make the final call.

=== CLAUSE DETAILS ===
Clause Type: {clause_type}
Risk Profile for this type: {clause_risk_profile}
Typical risk level seen across all contracts for this type: {typical_risk_for_type}

Clause Text:
{clause_text}

=== MODEL ASSESSMENTS ===
Model A ({qwen_label}):
{qwen_reason}

Model B ({gemini_label}):
{gemini_reason}

=== YOUR TASK ===
Risk levels:
- LOW: Standard, balanced, or favorable to the signing party. Common market practice.
- MEDIUM: Some risk present. One-sided or unusual terms that warrant attention but are negotiable.
- HIGH: Significant risk. Heavily one-sided, unusual liability exposure, or missing critical protections.

Assess risk FROM THE PERSPECTIVE OF THE PARTY SIGNING THE CONTRACT (the counterparty to the drafter — typically the customer, licensee, or non-drafting party).

Read the actual clause text carefully. Do not anchor on the clause type name alone.

Respond in JSON only — no preamble, no markdown:
{{"final_label": "LOW"|"MEDIUM"|"HIGH", "reasoning": "one or two sentences explaining your call"}}"""


def load_opus_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["category"] == "GEMINI_PRO_REVIEW"]


def load_all_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_all_rows(csv_path: Path, rows: list[dict], fieldnames: list[str]):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def call_gemini(client, prompt: str) -> dict:
    import json
    from google.genai import types

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=2000,
            response_mime_type="application/json",
        ),
    )
    # Gemini 2.5 Pro may return None for .text when thinking is active;
    # fall back to extracting text directly from the first non-thought part.
    raw = response.text
    if raw is None:
        raise ValueError(f"Empty response (finish_reason={response.candidates[0].finish_reason if response.candidates else 'unknown'})")
    result = json.loads(raw.strip())
    if result.get("final_label") not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"Invalid final_label: {result.get('final_label')!r}")
    return result


def main(dry_run: bool = False, save_every: int = 10, n_samples: int = None):
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found — check your .env file")

    client = genai.Client(api_key=api_key) if not dry_run else None

    all_rows = load_all_rows(MASTER_CSV)
    fieldnames = list(all_rows[0].keys())

    row_index = {r["row_num"]: i for i, r in enumerate(all_rows)}

    opus_rows = [r for r in all_rows if r["category"] == "GEMINI_PRO_REVIEW"]
    already_done = [r for r in opus_rows if r["final_label"] in ("LOW", "MEDIUM", "HIGH")]
    todo = [r for r in opus_rows if r["final_label"] not in ("LOW", "MEDIUM", "HIGH")]

    if n_samples:
        todo = todo[:n_samples]
    logger.info(f"GEMINI_PRO_REVIEW total: {len(opus_rows)} | Already labeled: {len(already_done)} | To do: {len(todo)}")

    errors = 0
    for i, row in enumerate(todo):
        clause_text = row["clause_text"][:3000]
        text_len = len(row["clause_text"])
        truncation_note = (
            "\nNote: The clause text above appears incomplete or very short. "
            "Weight the model assessments below more heavily in your judgment."
            if text_len < 200 else ""
        )

        prompt = REVIEW_PROMPT.format(
            clause_type=row["clause_type"],
            clause_risk_profile=row["clause_risk_profile"],
            typical_risk_for_type=row["typical_risk_for_type"],
            clause_text=clause_text + truncation_note,
            qwen_label=row["qwen_label"],
            qwen_reason=row["qwen_reason"],
            gemini_label=row["gemini_label"],
            gemini_reason=row["gemini_reason"],
        )

        if dry_run:
            print(f"\n--- Row {row['row_num']} ({row['review_id']}) | {text_len} chars ---")
            print(f"Clause text: {row['clause_text']}")
            print(f"Qwen={row['qwen_label']}: {row['qwen_reason']}")
            print(f"Gemini={row['gemini_label']}: {row['gemini_reason']}")
            continue

        try:
            result = call_gemini(client, prompt)
            final_label = result["final_label"]
            reasoning = result.get("reasoning", "")
            logger.info(f"[{i+1}/{len(todo)}] {row['review_id']} ({row['clause_type']}) "
                        f"Qwen={row['qwen_label']} Gemini={row['gemini_label']} → {final_label}")

            idx = row_index[row["row_num"]]
            all_rows[idx]["final_label"] = final_label
            all_rows[idx]["notes"] = f"Gemini-2.5-Pro: {reasoning}"
            all_rows[idx]["reviewer"] = "Gemini-2.5-Pro"

        except Exception as e:
            logger.warning(f"Failed on {row['review_id']}: {e}")
            errors += 1

        if (i + 1) % save_every == 0:
            save_all_rows(MASTER_CSV, all_rows, fieldnames)
            logger.info(f"Progress saved ({i+1}/{len(todo)} done, {errors} errors)")

    if not dry_run:
        save_all_rows(MASTER_CSV, all_rows, fieldnames)
        done_count = len(todo) - errors
        logger.info(f"Done. Labeled: {done_count} | Errors: {errors} | Saved to {MASTER_CSV}")

        from collections import Counter
        final_labels = [all_rows[row_index[r["row_num"]]]["final_label"]
                        for r in todo if all_rows[row_index[r["row_num"]]]["final_label"] in ("LOW","MEDIUM","HIGH")]
        logger.info(f"Label distribution: {dict(Counter(final_labels))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true", help="Print prompts without making API calls")
    parser.add_argument("--save_every", type=int, default=10, help="Save progress every N rows")
    parser.add_argument("--n_samples", type=int, default=None, help="Limit to N rows (for testing)")
    args = parser.parse_args()
    main(dry_run=args.dry_run, save_every=args.save_every, n_samples=args.n_samples)
