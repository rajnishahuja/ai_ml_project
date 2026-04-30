"""
relabel_soft_labels_v2.py
=========================
Re-labels the 1,327 SOFT_LABEL rows using an improved prompt that anchors
on the signing party (from the Parties metadata span). Supports three labelers:
  - local  : Qwen 30B on local llama-server GPU (port 10006)
  - gemini : Gemini 2.5 Flash via Google Gemini API
  - groq   : Llama 3.3 70B via Groq API

Key improvement over v1: parties field injected into prompt, so models
can identify the signing party explicitly before labeling. Clause type
and type description are dropped — clause text makes the type clear and
type names sometimes cause confusion.

After running all labelers, use --compare to see how many rows agree across
models. With 3 labelers, majority vote (2-of-3) is used for the consensus
label. Rows where all three disagree remain for manual review.

Usage:
    # Label with Qwen (GPU, port 10006)
    python scripts/relabel_soft_labels_v2.py --labeler local

    # Label with Gemini Flash
    python scripts/relabel_soft_labels_v2.py --labeler gemini

    # Label with Llama 3.3 70B via Groq
    python scripts/relabel_soft_labels_v2.py --labeler groq

    # Test run — first 20 rows only
    python scripts/relabel_soft_labels_v2.py --labeler groq --n_samples 20

    # Compare all available outputs (2-of-3 majority vote)
    python scripts/relabel_soft_labels_v2.py --compare

    # Update master_label_review.csv with consensus labels
    python scripts/relabel_soft_labels_v2.py --compare --update-master
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MASTER_CSV   = Path("data/review/master_label_review.csv")
SPANS_JSON   = Path("data/processed/all_positive_spans.json")
OUTPUT_DIR   = Path("data/synthetic")

OUTPUT_PATHS = {
    "local":     OUTPUT_DIR / "soft_label_relabel_v2_qwen.json",
    "gemini":    OUTPUT_DIR / "soft_label_relabel_v2_gemini.json",
    "groq":      OUTPUT_DIR / "soft_label_relabel_v2_groq.json",
    "anthropic": OUTPUT_DIR / "relabel_claude.json",
}

LOCAL_PORT      = 10006
LOCAL_MODEL     = "mavenir-generic1-30b-q4_k_xl.gguf"
GROQ_MODEL      = "llama-3.3-70b-versatile"
ANTHROPIC_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT = """You are a legal contract risk assessor.

Parties to this contract:
{parties}

Clause Text:
{clause_text}

The signing party is the counterparty to the drafter — typically the customer, licensee, vendor, or non-drafting party. Identify them from the Parties field above, then assess risk SOLELY from their perspective.

Risk levels:
- LOW: Standard, balanced, or favorable to the signing party. Common market practice.
- MEDIUM: Some risk. One-sided or unusual terms warranting attention but manageable.
- HIGH: Significant risk. Heavily one-sided, unusual liability exposure, or missing critical protections.

Respond in JSON only. No preamble, no markdown, no backticks:
{{"signing_party": "...", "risk_level": "LOW"|"MEDIUM"|"HIGH", "reason": "...under 80 words..."}}"""

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_soft_label_rows() -> list[dict]:
    with MASTER_CSV.open(encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["category"] == "SOFT_LABEL"]


def load_claude_target_rows() -> list[dict]:
    """
    Rows for Claude relabeling:
      - MANUAL_REVIEW (235): human-labeled but no parties metadata at review time
      - GEMINI_PRO_REVIEW (87): Gemini-family tiebreaker, not independent
      - SOFT_LABEL unresolved (478): Qwen v2 and Gemini v2 still disagree
    """
    with MASTER_CSV.open(encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    target = [r for r in all_rows
              if r["category"] in ("MANUAL_REVIEW", "GEMINI_PRO_REVIEW")]

    # Identify unresolved SOFT_LABEL rows (Qwen v2 and Gemini v2 disagree or one is missing)
    qwen_labels, gemini_labels = {}, {}
    if OUTPUT_PATHS["local"].exists():
        data = json.loads(OUTPUT_PATHS["local"].read_text(encoding="utf-8"))
        qwen_labels = {str(r["row_num"]): r["label"] for r in data if r["label"] != "ERROR"}
    if OUTPUT_PATHS["gemini"].exists():
        data = json.loads(OUTPUT_PATHS["gemini"].read_text(encoding="utf-8"))
        gemini_labels = {str(r["row_num"]): r["label"] for r in data if r["label"] != "ERROR"}

    for r in all_rows:
        if r["category"] != "SOFT_LABEL":
            continue
        rn = r["row_num"]
        q  = qwen_labels.get(rn)
        g  = gemini_labels.get(rn)
        if not q or not g or q != g:   # missing label or disagreement = unresolved
            target.append(r)

    logger.info(
        f"Claude target: {sum(1 for r in target if r['category']=='MANUAL_REVIEW')} MANUAL_REVIEW + "
        f"{sum(1 for r in target if r['category']=='GEMINI_PRO_REVIEW')} GEMINI_PRO_REVIEW + "
        f"{sum(1 for r in target if r['category']=='SOFT_LABEL')} unresolved SOFT_LABEL"
    )
    return target


def build_parties_lookup() -> dict[str, str]:
    """Map contract name -> parties text from all_positive_spans.json."""
    spans = json.loads(SPANS_JSON.read_text(encoding="utf-8"))
    lookup = {}
    for span in spans:
        if span["clause_type"] == "Parties":
            lookup[span["contract"]] = span["clause_text"]
    return lookup


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw.strip())
    if result.get("risk_level") not in ("LOW", "MEDIUM", "HIGH"):
        raise ValueError(f"Invalid risk_level: {result.get('risk_level')!r}")
    if not result.get("reason"):
        raise ValueError("Missing reason in response")
    return result


def init_client(labeler: str):
    if labeler == "local":
        from openai import OpenAI
        client = OpenAI(base_url=f"http://localhost:{LOCAL_PORT}/v1", api_key="sk-no-key")
        logger.info(f"Local llama-server client initialized (port {LOCAL_PORT}, {LOCAL_MODEL})")
        return client
    elif labeler == "gemini":
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found — check your .env file")
        client = genai.Client(api_key=api_key)
        logger.info("Gemini client initialized (gemini-2.5-flash, temperature=0, thinking off)")
        return client
    elif labeler == "groq":
        from openai import OpenAI
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found — check your .env file")
        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        logger.info(f"Groq client initialized ({GROQ_MODEL}, temperature=0)")
        return client
    elif labeler == "anthropic":
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found — check your .env file")
        client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Anthropic client initialized ({ANTHROPIC_MODEL}, temperature=0)")
        return client
    raise ValueError(f"Unknown labeler: {labeler!r}")


def call_local(client, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=LOCAL_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_response(response.choices[0].message.content)


def call_gemini(client, prompt: str) -> dict:
    from google.genai import types
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=300,
            response_mime_type="application/json",
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return parse_response(response.text)


def call_groq(client, prompt: str) -> dict:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return parse_response(response.choices[0].message.content)


def call_anthropic(client, prompt: str) -> dict:
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return parse_response(response.content[0].text)


# ---------------------------------------------------------------------------
# Labeling loop
# ---------------------------------------------------------------------------

def run_labeler(labeler: str, n_samples: int = None, save_every: int = 50, sleep_seconds: float = 0.0):
    output_path = OUTPUT_PATHS[labeler]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    soft_rows = load_claude_target_rows() if labeler == "anthropic" else load_soft_label_rows()
    if n_samples:
        soft_rows = soft_rows[:n_samples]
        logger.info(f"Test run — using {n_samples} rows")

    parties_lookup = build_parties_lookup()
    logger.info(f"Loaded {len(soft_rows)} SOFT_LABEL rows, {len(parties_lookup)} contracts with parties metadata")

    # Resume support
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        done_row_nums = {str(r["row_num"]) for r in results if r["label"] != "ERROR"}
        logger.info(f"Resuming — {len(done_row_nums)} rows already labeled")
    else:
        results = []
        done_row_nums = set()

    client = init_client(labeler)
    call_fn = {"local": call_local, "gemini": call_gemini, "groq": call_groq,
               "anthropic": call_anthropic}[labeler]

    errors = 0
    labeled_this_run = 0
    total = len(soft_rows)

    for row in soft_rows:
        row_num = row["row_num"]
        if row_num in done_row_nums:
            continue

        contract = row["contract"]
        clause_text = row["clause_text"]
        parties = parties_lookup.get(contract, "Not available")

        prompt = PROMPT.format(
            parties=parties[:500],
            clause_text=clause_text[:3000],
        )

        try:
            result = call_fn(client, prompt)
            results.append({
                "row_num":       int(row_num),
                "label":         result["risk_level"],
                "signing_party": result.get("signing_party", ""),
                "reason":        result["reason"],
                "parties_used":  parties[:200],
            })
            done_row_nums.add(row_num)

        except Exception as e:
            logger.warning(f"row_num={row_num} failed: {e}")
            results.append({
                "row_num":       int(row_num),
                "label":         "ERROR",
                "signing_party": "",
                "reason":        str(e),
                "parties_used":  parties[:200],
            })
            errors += 1

        labeled_this_run += 1
        if sleep_seconds:
            time.sleep(sleep_seconds)
        if labeled_this_run % save_every == 0:
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            pct = 100 * labeled_this_run / total
            logger.info(f"Progress: {labeled_this_run}/{total} ({pct:.0f}%) | Errors: {errors}")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = Counter(r["label"] for r in results)
    logger.info(f"Done. LOW={counts['LOW']} MEDIUM={counts['MEDIUM']} HIGH={counts['HIGH']} ERROR={counts['ERROR']}")
    logger.info(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Async labeler (used for anthropic to run concurrent requests)
# ---------------------------------------------------------------------------

async def run_labeler_async(labeler: str, n_samples: int = None, save_every: int = 50,
                             concurrency: int = 3):
    import anthropic as _anthropic

    output_path = OUTPUT_PATHS[labeler]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = load_claude_target_rows()
    if n_samples:
        all_rows = all_rows[:n_samples]
        logger.info(f"Test run — using {n_samples} rows")

    parties_lookup = build_parties_lookup()
    logger.info(f"Loaded {len(all_rows)} target rows, {len(parties_lookup)} contracts with parties metadata")

    # Resume support
    if output_path.exists():
        results = json.loads(output_path.read_text(encoding="utf-8"))
        done_row_nums = {str(r["row_num"]) for r in results if r["label"] != "ERROR"}
        logger.info(f"Resuming — {len(done_row_nums)} rows already labeled")
    else:
        results = []
        done_row_nums = set()

    pending = [r for r in all_rows if r["row_num"] not in done_row_nums]
    total = len(pending)
    logger.info(f"Rows to label: {total} | Concurrency: {concurrency}")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found — check your .env file")
    client = _anthropic.AsyncAnthropic(api_key=api_key)

    semaphore = asyncio.Semaphore(concurrency)
    lock      = asyncio.Lock()
    completed = 0
    errors    = 0

    async def process_row(row):
        nonlocal completed, errors
        row_num     = row["row_num"]
        parties     = parties_lookup.get(row["contract"], "Not available")
        prompt      = PROMPT.format(parties=parties[:500], clause_text=row["clause_text"][:3000])

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=300,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                parsed = parse_response(response.content[0].text)
                entry = {
                    "row_num":       int(row_num),
                    "label":         parsed["risk_level"],
                    "signing_party": parsed.get("signing_party", ""),
                    "reason":        parsed["reason"],
                    "parties_used":  parties[:200],
                }
            except Exception as e:
                logger.warning(f"row_num={row_num} failed: {e}")
                entry = {
                    "row_num":       int(row_num),
                    "label":         "ERROR",
                    "signing_party": "",
                    "reason":        str(e),
                    "parties_used":  parties[:200],
                }
                async with lock:
                    errors += 1

        async with lock:
            results.append(entry)
            completed += 1
            if completed % save_every == 0:
                output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
                logger.info(f"Progress: {completed}/{total} ({100*completed/total:.0f}%) | Errors: {errors}")

    await asyncio.gather(*[process_row(row) for row in pending])
    await client.close()

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    counts = Counter(r["label"] for r in results)
    logger.info(f"Done. LOW={counts['LOW']} MEDIUM={counts['MEDIUM']} HIGH={counts['HIGH']} ERROR={counts['ERROR']}")
    logger.info(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Compare / merge  (2-of-3 majority vote across all available labelers)
# ---------------------------------------------------------------------------

LABELER_NAMES = {
    "local":     "qwen",
    "gemini":    "gemini",
    "groq":      "groq",
    "anthropic": "claude",
}


def compare(update_master: bool = False):
    """
    2-tier consensus using Qwen v2, Gemini v2, and Claude (Groq excluded).

    Tier 1 — rows where Claude has a label (946 rows):
        3-way vote: Qwen + Gemini + Claude. Majority (2-of-3) wins.
    Tier 2 — remaining SOFT_LABEL rows where Qwen+Gemini already agree (703 rows):
        Use their pairwise agreement directly.
    Rows where all 3 disagree: unresolved.
    """
    # Load only the three relevant labelers (Groq excluded)
    qwen   = {}
    gemini = {}
    claude = {}

    if OUTPUT_PATHS["local"].exists():
        qwen = {r["row_num"]: r for r in json.loads(OUTPUT_PATHS["local"].read_text(encoding="utf-8"))
                if r["label"] != "ERROR"}
        logger.info(f"qwen:   {len(qwen)} rows loaded")

    if OUTPUT_PATHS["gemini"].exists():
        gemini = {r["row_num"]: r for r in json.loads(OUTPUT_PATHS["gemini"].read_text(encoding="utf-8"))
                  if r["label"] != "ERROR"}
        logger.info(f"gemini: {len(gemini)} rows loaded")

    if OUTPUT_PATHS["anthropic"].exists():
        claude = {r["row_num"]: r for r in json.loads(OUTPUT_PATHS["anthropic"].read_text(encoding="utf-8"))
                  if r["label"] != "ERROR"}
        logger.info(f"claude: {len(claude)} rows loaded")

    if not qwen or not gemini:
        logger.error("Need at least qwen and gemini outputs. Run labelers first.")
        return

    # All SOFT_LABEL row_nums (superset)
    all_rns = set(qwen.keys()) | set(gemini.keys())

    consensus      = {}
    consensus_src  = {}

    for rn in all_rns:
        q = qwen.get(rn, {}).get("label")
        g = gemini.get(rn, {}).get("label")
        c = claude.get(rn, {}).get("label")

        if c:
            # Tier 1: 3-way vote (Qwen + Gemini + Claude)
            votes = {k: v for k, v in [("qwen", q), ("gemini", g), ("claude", c)] if v}
            vote_counts = Counter(votes.values())
            top, cnt = vote_counts.most_common(1)[0]
            if cnt >= 2:
                consensus[rn]     = top
                consensus_src[rn] = "+".join(k for k, v in votes.items() if v == top)
            else:
                consensus[rn]     = None   # all 3 disagree
                consensus_src[rn] = "none"
        elif q and g:
            # Tier 2: Qwen+Gemini pairwise
            if q == g:
                consensus[rn]     = q
                consensus_src[rn] = "qwen+gemini"
            else:
                consensus[rn]     = None
                consensus_src[rn] = "none"
        else:
            consensus[rn]     = None
            consensus_src[rn] = "none"

    agreed_rows  = {rn: lbl for rn, lbl in consensus.items() if lbl is not None}
    unresolved   = {rn for rn, lbl in consensus.items() if lbl is None}
    total        = len(consensus)

    print(f"\n=== v2 Agreement Report (Qwen + Gemini + Claude; Groq excluded) ===")
    print(f"Total rows covered:    {total}")
    print(f"Resolved (consensus):  {len(agreed_rows)} ({100*len(agreed_rows)/total:.1f}%)")
    print(f"Unresolved:            {len(unresolved)} ({100*len(unresolved)/total:.1f}%)")

    # Break down by method
    tier1_total    = sum(1 for rn in consensus if claude.get(rn, {}).get("label"))
    tier1_resolved = sum(1 for rn in agreed_rows if claude.get(rn, {}).get("label"))
    tier2_total    = total - tier1_total
    tier2_resolved = len(agreed_rows) - tier1_resolved

    print(f"\nTier 1 (3-way with Claude): {tier1_total} rows → {tier1_resolved} resolved "
          f"({100*tier1_resolved/tier1_total:.1f}%)")
    print(f"Tier 2 (Qwen+Gemini only):  {tier2_total} rows → {tier2_resolved} resolved "
          f"({100*tier2_resolved/tier2_total:.1f}% — all agreed by definition)")

    agreed_counts = Counter(agreed_rows.values())
    print(f"\nConsensus label breakdown:")
    for lbl in ["LOW", "MEDIUM", "HIGH"]:
        print(f"  {lbl}: {agreed_counts[lbl]} ({100*agreed_counts[lbl]/len(agreed_rows):.1f}%)")

    print(f"\nPairwise agreement rates (on overlapping rows):")
    pairs = [("qwen", qwen, "gemini", gemini),
             ("qwen", qwen, "claude", claude),
             ("gemini", gemini, "claude", claude)]
    for na, da, nb, db in pairs:
        both  = set(da) & set(db)
        match = sum(1 for rn in both if da[rn]["label"] == db[rn]["label"])
        print(f"  {na} vs {nb}: {match}/{len(both)} ({100*match/len(both):.1f}%)")

    # Save consensus JSON
    consensus_rows = []
    for rn in sorted(consensus.keys()):
        entry = {
            "row_num":          rn,
            "consensus_label":  consensus[rn],
            "consensus_source": consensus_src[rn],
            "qwen_label":   qwen.get(rn, {}).get("label"),
            "qwen_party":   qwen.get(rn, {}).get("signing_party"),
            "qwen_reason":  qwen.get(rn, {}).get("reason"),
            "gemini_label": gemini.get(rn, {}).get("label"),
            "gemini_party": gemini.get(rn, {}).get("signing_party"),
            "gemini_reason":gemini.get(rn, {}).get("reason"),
            "claude_label": claude.get(rn, {}).get("label"),
            "claude_party": claude.get(rn, {}).get("signing_party"),
            "claude_reason":claude.get(rn, {}).get("reason"),
        }
        consensus_rows.append(entry)

    consensus_path = OUTPUT_DIR / "soft_label_relabel_v2_consensus.json"
    consensus_path.write_text(json.dumps(consensus_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nConsensus file saved: {consensus_path}  ({len(consensus_rows)} rows)")

    if update_master:
        _update_master_csv(agreed_rows, consensus_src)


def _update_master_csv(agreed: dict, source: dict):
    """Replace SOFT_LABEL rows with majority-vote consensus labels."""
    with MASTER_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated = 0
    for row in rows:
        rn = int(row["row_num"])
        if row["category"] == "SOFT_LABEL" and rn in agreed:
            row["final_label"] = agreed[rn]
            row["category"]    = "SOFT_LABEL_V2_AGREED"
            row["reviewer"]    = source.get(rn, "v2_majority")
            row["notes"]       = f"v2 relabel majority vote: {agreed[rn]} ({source.get(rn, '')})"
            updated += 1

    with MASTER_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"master_label_review.csv updated: {updated} rows → SOFT_LABEL_V2_AGREED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeler",       choices=["local", "gemini", "groq", "anthropic"], default=None)
    parser.add_argument("--n_samples",     type=int, default=None, help="Limit rows for test runs")
    parser.add_argument("--save_every",    type=int, default=50)
    parser.add_argument("--sleep",         type=float, default=0.0, help="Seconds between API calls (use 2.0 for Groq free tier)")
    parser.add_argument("--concurrency",   type=int, default=3, help="Concurrent requests for anthropic labeler")
    parser.add_argument("--compare",       action="store_true", help="Compare both v2 outputs")
    parser.add_argument("--update-master", action="store_true", help="Write agreed labels back to master CSV")
    args = parser.parse_args()

    if args.compare:
        compare(update_master=args.update_master)
    elif args.labeler == "anthropic":
        asyncio.run(run_labeler_async("anthropic", n_samples=args.n_samples,
                                      save_every=args.save_every, concurrency=args.concurrency))
    elif args.labeler:
        run_labeler(args.labeler, n_samples=args.n_samples, save_every=args.save_every, sleep_seconds=args.sleep)
    else:
        parser.print_help()
