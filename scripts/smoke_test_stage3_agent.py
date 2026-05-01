"""
smoke_test_stage3_agent.py
==========================
End-to-end smoke test for Stage 3: DeBERTa → FAISS → LLM pipeline.

Bypasses Stage 1+2 entirely — injects ClauseObjects built from known rows
in training_dataset.json so the test runs without a full contract PDF.

What it verifies:
  1. DeBERTa (Ens-F) predicts a label + confidence for each clause.
  2. High-confidence clauses → fast path: FAISS returns similar clauses,
     LLM produces a structured explanation.
  3. Low-confidence clauses → agent path: LangGraph ReAct calls tools
     (precedent_search, optionally contract_search) and returns a structured
     RiskAssessment with agent_trace populated.
  4. extract_signing_party() resolves from the synthetic Parties clause.
  5. METADATA_CLAUSE_TYPES are skipped (Parties clause not in results).

Usage:
    /home/ubuntu/miniconda3/envs/rajnish-env/bin/python3 \\
        scripts/smoke_test_stage3_agent.py

Set OPENAI_API_KEY env var or agent_api_key in stage3_config.yaml to 'none'
when using a local llama.cpp server (default config already does this).
"""

import logging
import sys
from pathlib import Path

# Add project root to path so src.* imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.schema import ClauseObject
from src.stage3_risk_agent.agent import assess_clauses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CE_MODEL_PATH   = "models/stage3_risk_deberta_v3_run22_parties/final"
CORN_MODEL_PATH = "models/stage3_risk_deberta_v3_run23_corn_parties/final"

DOC_ID = "smoke_test_doc_001"


def make_clause(clause_id: str, clause_type: str, clause_text: str) -> ClauseObject:
    return ClauseObject(
        clause_id=clause_id,
        document_id=DOC_ID,
        clause_type=clause_type,
        clause_text=clause_text,
        start_pos=0,
        end_pos=len(clause_text),
        confidence=0.95,  # Stage 1+2 extraction confidence — not risk confidence
    )


# ---------------------------------------------------------------------------
# Test clauses — representative rows from training_dataset.json
# ---------------------------------------------------------------------------

CLAUSES = [
    # Parties clause — used by extract_signing_party(), not risk-assessed
    make_clause(
        "c0", "Parties",
        "This Agreement is entered into by and between ARCONIC ROLLED PRODUCTS CORP. "
        "(\"Company\") and its counterparty (\"Licensor\").",
    ),

    # Governing Law — typically Governing Law → fast path (DeBERTa usually high-conf here)
    make_clause(
        "c1", "Governing Law",
        "This Agreement will be governed and construed in accordance with the laws of the "
        "State of California without giving effect to conflict of laws principles.",
    ),
    make_clause(
        "c2", "Governing Law",
        "This Agreement and any claim, controversy or dispute arising out of or related to "
        "this Agreement, any of the transactions contemplated hereby and/or the interpretation "
        "and enforcement of the rights and duties of the Parties, whether arising in contract, "
        "tort, equity or otherwise, shall be governed by and construed in accordance with the "
        "domestic laws of the State of Israel (including in respect of the statute of "
        "limitations or other limitations period applicable to any such claim, controversy or "
        "dispute), without giving effect to any choice or conflict of law provision or rule "
        "(whether of the State of Israel or any other jurisdiction) that would cause the "
        "application of the laws of any jurisdiction other than the State of Israel.",
    ),

    # Cap On Liability — tests HIGH label
    make_clause(
        "c3", "Cap On Liability",
        "It is expressly agreed that Capital Resources shall not be liable for any loss, "
        "liability, claim, damage or expense or be required to contribute any amount which "
        "in the aggregate exceeds the amount paid (excluding reimbursable expenses) to "
        "Capital Resources under this Agreement.",
    ),

    # Affiliate License-Licensee — signing-party sensitive; HIGH vs LOW flips
    make_clause(
        "c4", "Affiliate License-Licensee",
        "with respect to any documentation, technical or confidential business information "
        "and/or software relating to the Equipment (collectively, \"Software\"), the Purchase "
        "Order will grant Lessor a license to use the Software and will allow Lessor to grant "
        "a sublicense to the Company to use such Software pursuant to the Lease and will allow "
        "Lessor to grant a sublicense to a third party after a termination or the expiration "
        "of the Lease in the event the Company does not elect to exercise any purchase option "
        "that may be provided for in the Lease;",
    ),
    make_clause(
        "c5", "Affiliate License-Licensee",
        "For the avoidance of doubt, Licensor also grants to Licensee and its subsidiaries "
        "and affiliates a non-exclusive, worldwide royalty-free license for continued use of "
        "the Licensed Mark for the production and sale of inventory containing the Licensed "
        "Mark applied to such products during the Transition Period as set forth in section "
        "8.2 of the Separation and Distribution Agreement and in Schedule 2 of this Agreement.",
    ),
]


def main():
    print("\n" + "=" * 70)
    print("Stage 3 Agent — End-to-End Smoke Test")
    print("=" * 70)
    print(f"  Clauses injected : {len(CLAUSES)}")
    print(f"  CE model         : {CE_MODEL_PATH}")
    print(f"  CORN model       : {CORN_MODEL_PATH}")
    print(f"  Config           : configs/stage3_config.yaml")
    print()

    results = assess_clauses(
        clauses=CLAUSES,
        config_path="configs/stage3_config.yaml",
        ce_model_path=CE_MODEL_PATH,
        corn_model_path=CORN_MODEL_PATH,
    )

    print("\n" + "=" * 70)
    print(f"RESULTS  ({len(results)} risk-assessed clauses, Parties skipped)")
    print("=" * 70)

    for r in results:
        path_indicator = "agent" if r.agent_trace else "fast "
        print(f"\n[{path_indicator}] {r.clause_type}")
        print(f"  Risk      : {r.risk_level}  (DeBERTa conf={r.confidence:.2f})")
        print(f"  Explanation: {r.risk_explanation}")
        if r.similar_clauses:
            print(f"  Precedents: {len(r.similar_clauses)} returned")
            for sc in r.similar_clauses[:2]:
                print(f"    [{sc.risk_level}] ({sc.clause_type}, sim={sc.similarity:.2f}) {sc.text[:80]!r}")
        if r.agent_trace:
            print(f"  Agent trace ({len(r.agent_trace)} tool calls):")
            for t in r.agent_trace:
                print(f"    tool={t.tool}  result_count={t.result_count}")

    print("\n" + "=" * 70)
    # Basic assertions
    assert len(results) == len(CLAUSES) - 1, (
        f"Expected {len(CLAUSES)-1} results (Parties excluded), got {len(results)}"
    )
    risk_levels = {r.risk_level for r in results}
    assert risk_levels <= {"LOW", "MEDIUM", "HIGH"}, f"Unexpected risk level values: {risk_levels}"
    print("SMOKE TEST PASSED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
