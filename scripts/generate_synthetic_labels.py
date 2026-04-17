"""
generate_synthetic_labels.py
==============================
Generates synthetic risk labels for positive CUAD clause spans using the
Claude API. Labels each clause as LOW / MEDIUM / HIGH risk with a structured
reason (risk_driver + risk_reason) and confidence score.

Key behaviours (v1):
  - Metadata clause types are skipped entirely (Option B routing).
  - Identical clause texts are labeled once and fanned to duplicate rows,
    saving ~38% of API calls vs naive per-row labeling.
  - Clause-type descriptions from Atticus CSV are injected into the prompt
    so the model has authoritative semantics, not just the type name.
  - Saves progress every 50 unique texts — safe to interrupt and resume.

Output: data/synthetic/synthetic_risk_labels.json

Usage:
    python scripts/generate_synthetic_labels.py
    python scripts/generate_synthetic_labels.py --n_samples 50   # test run
    python scripts/generate_synthetic_labels.py --input data/processed/all_positive_spans.json
"""

import csv
import json
import logging
import os
import re
import time
import argparse
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH      = "data/processed/all_positive_spans.json"
DEFAULT_OUTPUT_PATH     = "data/synthetic/synthetic_risk_labels.json"
CATEGORIES_PATH         = Path("data/reference/cuad_category_descriptions.csv")

# ---------------------------------------------------------------------------
# Metadata types — routed away from risk classifier (Option B).
# These are not labeled; they surface in the Stage 4 report header instead.
# ---------------------------------------------------------------------------

METADATA_TYPES = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

# ---------------------------------------------------------------------------
# Prompt template (v1)
# ---------------------------------------------------------------------------

RISK_LABEL_PROMPT = """You are a legal contract risk assessor. Evaluate risk FROM THE PERSPECTIVE OF THE PARTY SIGNING THE CONTRACT — the counterparty to the drafter, typically the customer, licensee, or non-drafting party.

Clause Type: {clause_type}
Clause Type Description: {clause_type_description}

Clause Text:
{clause_text}

Risk levels:
- LOW: Standard, balanced, or favorable to the signing party. Common market practice. No significant concern.
- MEDIUM: Some risk present. One-sided or unusual terms that warrant attention but are negotiable or manageable.
- HIGH: Significant risk. Heavily one-sided, unusual liability exposure, missing critical protections, or disproportionately benefits the drafter.

Calibration examples:
- LOW: Governing Law — "This Agreement shall be governed by the laws of the State of Delaware." → risk_driver: "Delaware governing law, standard commercial jurisdiction", risk_reason: "Well-understood jurisdiction with predictable outcomes; no unusual choice-of-law burden on the signing party."
- MEDIUM: Anti-Assignment — "Neither party may assign this Agreement without prior written consent, not to be unreasonably withheld." → risk_driver: "mutual consent required for assignment", risk_reason: "Limits flexibility but the 'not unreasonably withheld' standard gives the signing party a viable path to assignment."
- HIGH: Non-Compete — "For 5 years after termination, signing party shall not engage in any business competing with Company in any market worldwide." → risk_driver: "5-year worldwide non-compete in any competing business", risk_reason: "No geographic or scope limit; severely constrains the signing party's ability to operate in their own industry post-contract."

Instructions:
- Base your score on the actual clause TEXT, not just the clause type name.
- risk_driver must quote or closely paraphrase a specific phrase from the text — do NOT just restate the clause type name.
- If the clause is too short or fragmentary to assess fully, lower your confidence score.

Respond in JSON only. No preamble, no markdown, no backticks. Just the raw JSON object:
{{"risk_level": "LOW"|"MEDIUM"|"HIGH", "risk_driver": "...", "risk_reason": "...", "confidence": 0.0-1.0}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_category_descriptions(path: Path) -> dict:
    """
    Returns dict keyed by lowercased clause type name → one-line description.
    Handles utf-8-sig BOM and the 'Category: '/'Description: ' cell prefixes
    that Atticus uses in their CSV.
    """
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            name = row[0].replace("Category: ", "").strip()
            desc = row[1].replace("Description: ", "").strip() if len(row) > 1 else ""
            out[name.lower()] = desc
    return out


def normalize_text(text: str) -> str:
    """Collapse whitespace differences so identical boilerplate deduplicates."""
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Label a single clause (one API call)
# ---------------------------------------------------------------------------

def label_clause(client, clause_type: str, clause_text: str, clause_type_description: str) -> dict:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": RISK_LABEL_PROMPT.format(
                clause_type=clause_type,
                clause_type_description=clause_type_description or "No description available.",
                clause_text=clause_text[:3000],
            )
        }]
    )

    raw = response.content[0].text.strip()

    # Strip markdown backticks if the model wraps anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    result = json.loads(raw)

    if result.get("risk_level") not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"Invalid risk_level: {result.get('risk_level')!r}")
    if "risk_driver" not in result:
        raise ValueError("Missing risk_driver in response")

    return result


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def generate_synthetic_labels(
    input_path: str,
    output_path: str,
    n_samples: int = None,
    save_every: int = 50,
    sleep_seconds: float = 0.5,
) -> list[dict]:

    spans = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cat_descriptions = load_category_descriptions(CATEGORIES_PATH)
    logger.info(f"Loaded {len(spans)} spans, {len(cat_descriptions)} category descriptions")

    # Option B — skip metadata types entirely
    risk_bearing = [s for s in spans if s["clause_type"] not in METADATA_TYPES]
    skipped_meta = len(spans) - len(risk_bearing)
    logger.info(f"Metadata spans skipped: {skipped_meta} | Risk-bearing spans: {len(risk_bearing)}")

    if n_samples:
        risk_bearing = risk_bearing[:n_samples]
        logger.info(f"Test run — using {n_samples} samples")

    # Dedup: group spans by normalized clause_text
    # API is called once per unique text; label fans to all rows sharing that text.
    norm_to_spans: dict[str, list[dict]] = defaultdict(list)
    for span in risk_bearing:
        norm_to_spans[normalize_text(span["clause_text"])].append(span)

    unique_count = len(norm_to_spans)
    saved_calls  = len(risk_bearing) - unique_count
    logger.info(f"Unique texts: {unique_count} | Duplicate texts (saved calls): {saved_calls}")

    # Load existing progress if interrupted
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        results = json.loads(output_file.read_text(encoding="utf-8"))
        done_norms = {normalize_text(r["clause_text"]) for r in results if r["risk_level"] != "ERROR"}
        logger.info(f"Resuming — {len(done_norms)} unique texts already labeled")
    else:
        results = []
        done_norms = set()

    # Initialize Claude client
    try:
        import anthropic
        try:
            from kaggle_secrets import UserSecretsClient
            api_key = UserSecretsClient().get_secret("ANTHROPIC_API_KEY")
        except Exception:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in Kaggle secrets or environment")
        client = anthropic.Anthropic(api_key=api_key)
        logger.info("Claude client initialized")
    except ImportError:
        raise ImportError("anthropic not installed — run: pip install anthropic")

    # Label loop — one API call per unique normalized text
    errors = 0
    labeled_this_run = 0

    for norm_text, span_group in norm_to_spans.items():
        if norm_text in done_norms:
            continue

        representative = span_group[0]
        clause_type = representative["clause_type"]
        clause_text = representative["clause_text"]
        description = cat_descriptions.get(clause_type.lower(), "")

        try:
            label = label_clause(client, clause_type, clause_text, description)
            # Fan label to every span sharing this text
            for span in span_group:
                results.append({
                    "id":           span["id"],
                    "contract":     span.get("contract", ""),
                    "clause_type":  span["clause_type"],
                    "clause_text":  span["clause_text"],
                    "risk_level":   label["risk_level"],
                    "risk_driver":  label["risk_driver"],
                    "risk_reason":  label["risk_reason"],
                    "confidence":   label.get("confidence", 0.0),
                })
            done_norms.add(norm_text)

        except Exception as e:
            logger.warning(f"Failed on '{clause_type}' (id={representative['id']}): {e}")
            for span in span_group:
                results.append({
                    "id":           span["id"],
                    "contract":     span.get("contract", ""),
                    "clause_type":  span["clause_type"],
                    "clause_text":  span["clause_text"],
                    "risk_level":   "ERROR",
                    "risk_driver":  "",
                    "risk_reason":  str(e),
                    "confidence":   0.0,
                })
            errors += 1

        labeled_this_run += 1
        if labeled_this_run % save_every == 0:
            output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Progress: {labeled_this_run}/{unique_count} unique texts | Errors: {errors}")

        time.sleep(sleep_seconds)

    # Final save
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    risk_counts = Counter(r["risk_level"] for r in results if r["risk_level"] != "ERROR")
    logger.info(f"\nDone!")
    logger.info(f"Total rows written: {len(results)}")
    logger.info(f"LOW:    {risk_counts['LOW']}")
    logger.info(f"MEDIUM: {risk_counts['MEDIUM']}")
    logger.info(f"HIGH:   {risk_counts['HIGH']}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit to N risk-bearing spans (after metadata filter) — use for test runs")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save progress every N unique texts labeled")
    parser.add_argument("--sleep",      type=float, default=0.5,
                        help="Seconds between API calls — increase if hitting rate limits")
    args = parser.parse_args()

    generate_synthetic_labels(
        input_path=args.input,
        output_path=args.output,
        n_samples=args.n_samples,
        save_every=args.save_every,
        sleep_seconds=args.sleep,
    )
