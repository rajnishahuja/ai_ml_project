"""
generate_synthetic_labels.py
==============================
Generates synthetic risk labels for positive CUAD clause spans.
Supports two labelers via --labeler flag:
  - local  : mavenir-generic1-30b on local llama-server GPU (port 10012)
  - gemini : gemini-2.5-flash via Google Gemini API

Labels each clause as LOW / MEDIUM / HIGH risk with a structured reason
(risk_driver + risk_reason) and confidence score.

Key behaviours:
  - Metadata clause types are skipped entirely (Option B routing).
  - Identical clause texts are labeled once and fanned to duplicate rows,
    saving ~38% of API calls vs naive per-row labeling.
  - Clause-type descriptions from Atticus CSV are injected into the prompt.
  - Saves progress every 50 unique texts — safe to interrupt and resume.

Usage:
    python scripts/generate_synthetic_labels.py                          # local 30B
    python scripts/generate_synthetic_labels.py --labeler gemini         # Gemini 2.5 Flash
    python scripts/generate_synthetic_labels.py --labeler gemini --n_samples 25  # test
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

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_INPUT_PATH  = "data/processed/all_positive_spans.json"
CATEGORIES_PATH     = Path("data/reference/cuad_category_descriptions.csv")

OUTPUT_PATHS = {
    "local":  "data/synthetic/synthetic_risk_labels_qwen.json",
    "gemini": "data/synthetic/synthetic_risk_labels_gemini.json",
}

# ---------------------------------------------------------------------------
# Metadata types — routed away from risk classifier (Option B).
# ---------------------------------------------------------------------------

METADATA_TYPES = frozenset({
    "Document Name",
    "Parties",
    "Agreement Date",
    "Effective Date",
    "Expiration Date",
})

# ---------------------------------------------------------------------------
# Prompt template
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

Self-check before responding: your risk_level MUST be consistent with your risk_reason.
- If your reason describes the clause as favorable, standard, or balanced → label must be LOW.
- If your reason describes it as one-sided or unusual but manageable → label must be MEDIUM.
- If your reason describes significant exposure or heavily one-sided terms → label must be HIGH.
A label that contradicts your own reasoning is an error — reconsider before outputting.

Respond in JSON only. No preamble, no markdown, no backticks. Just the raw JSON object:
{{"risk_level": "LOW"|"MEDIUM"|"HIGH", "risk_driver": "...", "risk_reason": "...", "confidence": 0.0-1.0}}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_category_descriptions(path: Path) -> dict:
    out = {}
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
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


def parse_json_response(raw: str) -> dict:
    """Strip markdown fences if present, then parse JSON."""
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
    if result.get("risk_level") not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"Invalid risk_level: {result.get('risk_level')!r}")
    if "risk_driver" not in result:
        raise ValueError("Missing risk_driver in response")
    return result


# ---------------------------------------------------------------------------
# Labeler-specific API calls
# ---------------------------------------------------------------------------

def label_clause_local(client, prompt: str) -> dict:
    response = client.chat.completions.create(
        model="mavenir-generic1-30b",
        max_tokens=400,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return parse_json_response(response.choices[0].message.content.strip())


def label_clause_gemini(client, prompt: str) -> dict:
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=400,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return parse_json_response(response.text.strip())


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------

def init_client(labeler: str):
    if labeler == "local":
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:10012/v1", api_key="sk-no-key")
        logger.info("Local llama-server client initialized (port 10012, mavenir-generic1-30b)")
        return client
    elif labeler == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found — check your .env file")
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized (gemini-2.5-flash, temperature=0, thinking off)")
        return client
    else:
        raise ValueError(f"Unknown labeler: {labeler!r}. Choose 'local' or 'gemini'.")


# ---------------------------------------------------------------------------
# Main labeling loop
# ---------------------------------------------------------------------------

def generate_synthetic_labels(
    input_path: str,
    output_path: str,
    labeler: str = "local",
    n_samples: int = None,
    save_every: int = 50,
    sleep_seconds: float = 0.0,
) -> list[dict]:

    spans = json.loads(Path(input_path).read_text(encoding="utf-8"))
    cat_descriptions = load_category_descriptions(CATEGORIES_PATH)
    logger.info(f"Loaded {len(spans)} spans, {len(cat_descriptions)} category descriptions")

    risk_bearing = [s for s in spans if s["clause_type"] not in METADATA_TYPES]
    logger.info(f"Metadata spans skipped: {len(spans)-len(risk_bearing)} | Risk-bearing spans: {len(risk_bearing)}")

    if n_samples:
        risk_bearing = risk_bearing[:n_samples]
        logger.info(f"Test run — using {n_samples} samples")

    norm_to_spans: dict[str, list[dict]] = defaultdict(list)
    for span in risk_bearing:
        norm_to_spans[normalize_text(span["clause_text"])].append(span)

    unique_count = len(norm_to_spans)
    logger.info(f"Unique texts: {unique_count} | Duplicate texts (saved calls): {len(risk_bearing)-unique_count}")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        results = json.loads(output_file.read_text(encoding="utf-8"))
        done_norms = {normalize_text(r["clause_text"]) for r in results if r["risk_level"] != "ERROR"}
        logger.info(f"Resuming — {len(done_norms)} unique texts already labeled")
    else:
        results = []
        done_norms = set()

    client = init_client(labeler)
    label_fn = label_clause_local if labeler == "local" else label_clause_gemini

    errors = 0
    labeled_this_run = 0

    for norm_text, span_group in norm_to_spans.items():
        if norm_text in done_norms:
            continue

        representative = span_group[0]
        clause_type = representative["clause_type"]
        clause_text = representative["clause_text"]
        description = cat_descriptions.get(clause_type.lower(), "")

        prompt = RISK_LABEL_PROMPT.format(
            clause_type=clause_type,
            clause_type_description=description or "No description available.",
            clause_text=clause_text[:3000],
        )

        try:
            label = label_fn(client, prompt)
            for span in span_group:
                results.append({
                    "id":          span["id"],
                    "contract":    span.get("contract", ""),
                    "clause_type": span["clause_type"],
                    "clause_text": span["clause_text"],
                    "risk_level":  label["risk_level"],
                    "risk_driver": label["risk_driver"],
                    "risk_reason": label["risk_reason"],
                    "confidence":  label.get("confidence", 0.0),
                    "labeler":     labeler,
                })
            done_norms.add(norm_text)

        except Exception as e:
            logger.warning(f"Failed on '{clause_type}' (id={representative['id']}): {e}")
            for span in span_group:
                results.append({
                    "id":          span["id"],
                    "contract":    span.get("contract", ""),
                    "clause_type": span["clause_type"],
                    "clause_text": span["clause_text"],
                    "risk_level":  "ERROR",
                    "risk_driver": "",
                    "risk_reason": str(e),
                    "confidence":  0.0,
                    "labeler":     labeler,
                })
            errors += 1

        labeled_this_run += 1
        if labeled_this_run % save_every == 0:
            output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info(f"Progress: {labeled_this_run}/{unique_count} unique texts | Errors: {errors}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    from collections import Counter
    risk_counts = Counter(r["risk_level"] for r in results if r["risk_level"] != "ERROR")
    logger.info(f"Done! Total rows: {len(results)} | HIGH: {risk_counts['HIGH']} | MEDIUM: {risk_counts['MEDIUM']} | LOW: {risk_counts['LOW']} | Errors: {errors}")
    logger.info(f"Saved to: {output_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeler",    choices=["local", "gemini"], default="local",
                        help="'local' = 30B GPU llama-server | 'gemini' = Gemini 2.5 Flash API")
    parser.add_argument("--input",      default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output",     default=None,
                        help="Output path (default: auto-set by labeler)")
    parser.add_argument("--n_samples",  type=int, default=None,
                        help="Limit to N risk-bearing spans — use for test runs")
    parser.add_argument("--save_every", type=int, default=50,
                        help="Save progress every N unique texts labeled")
    parser.add_argument("--sleep",      type=float, default=None,
                        help="Seconds between API calls (default: 0 for local, 4.0 for gemini free tier)")
    args = parser.parse_args()

    output = args.output or OUTPUT_PATHS[args.labeler]

    # Default sleep=0 for both labelers.
    # Gemini 2.5 Flash free tier is 1,000 RPM — inference latency (~1.2s/call)
    # naturally keeps us at ~50 RPM, well within limits. No sleep needed.
    sleep = args.sleep if args.sleep is not None else 0.0

    generate_synthetic_labels(
        input_path=args.input,
        output_path=output,
        labeler=args.labeler,
        n_samples=args.n_samples,
        save_every=args.save_every,
        sleep_seconds=sleep,
    )
