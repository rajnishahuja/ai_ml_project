"""
LLM client abstraction for Mistral-7B (and future swap-ins).

The Stage 4 report generator needs a single Mistral call: full contract text
in, JSON {summary, conclusion, action items} out. Stage 3 will eventually
share the same interface for its agent path.

Design:
  - `LLMClient` Protocol — minimal `generate_json` surface.
  - `MockLLMClient` — deterministic fixture. Returns schema-compliant JSON
    derived from the input prompts so the rest of Stage 4 (prompt builder,
    DOCX renderer, FastAPI download) can be developed and tested today
    without a real Mistral.
  - `HuggingFaceLLMClient` — STUB. Full transformers + bitsandbytes wiring
    deferred until the Mistral side is rolled out for the Stage 3 agent path.
    Until then it raises NotImplementedError; the Stage 4 node falls back
    to the Mock when this happens.

When real Mistral lands, only this file changes. Callers see the Protocol.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMClient(Protocol):
    """Minimal LLM contract used by Stage 4 (and, later, Stage 3)."""

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Run one LLM turn and return parsed JSON.

        Implementations MUST return a dict (never raise on invalid model
        output — coerce or return a safe fallback shape). Callers downstream
        should validate the keys they need.
        """
        ...


# ---------------------------------------------------------------------------
# MockLLMClient — used in dev / tests / when Mistral isn't running
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Deterministic fixture LLM. Schema-compliant JSON derived from inputs.

    The Stage 4 prompt embeds a compressed risk summary in the user prompt
    (counts + HIGH/MEDIUM clause types). This mock parses that summary back
    out and assembles a plausible report so the DOCX still looks realistic
    when no real model is loaded.
    """

    name = "mock-llm"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Return a realistic-looking summary + conclusion derived from prompt."""
        risk_signal = _extract_risk_signal(user_prompt)
        contract_excerpt = _first_meaningful_sentence(user_prompt)

        summary = (
            f"{contract_excerpt} "
            f"This contract was assessed across {risk_signal['total']} clause(s), "
            f"with {risk_signal['high']} flagged as high risk, "
            f"{risk_signal['medium']} as medium risk, and "
            f"{risk_signal['low']} as low risk."
        )

        if risk_signal["high"] > 0:
            assessment = (
                f"The contract carries elevated risk in "
                f"{risk_signal['high']} area(s); legal review of high-risk "
                f"clauses is strongly recommended before signing."
            )
        elif risk_signal["medium"] > 0:
            assessment = (
                f"The contract is broadly acceptable but contains "
                f"{risk_signal['medium']} clause(s) that warrant negotiation."
            )
        else:
            assessment = (
                "The contract appears commercially standard with no "
                "significant risk concerns identified."
            )

        high_actions = [
            f"Review and negotiate the {ctype.lower()} clause to limit "
            f"counterparty exposure."
            for ctype in risk_signal["high_types"][:5]
        ] or ["No high-priority actions required."]

        medium_actions = [
            f"Confirm the {ctype.lower()} clause is commercially reasonable "
            f"with legal counsel."
            for ctype in risk_signal["medium_types"][:5]
        ] or ["No medium-priority actions required."]

        return {
            "contract_summary":        summary,
            "overall_assessment":      assessment,
            "high_priority_actions":   high_actions,
            "medium_priority_actions": medium_actions,
        }


# ---------------------------------------------------------------------------
# HuggingFaceLLMClient — stub for real Mistral wiring
# ---------------------------------------------------------------------------

class HuggingFaceLLMClient:
    """Real Mistral via transformers + bitsandbytes 4-bit quantization.

    Wiring deferred — see configs/stage3_config.yaml for the model id and
    quantization config. When implemented, behavior should mirror the Mock:
    return a dict matching the Stage 4 prompt's documented output schema,
    never raise on parse failure.
    """

    name = "huggingface-mistral"

    def __init__(self, model_id: str = "mistralai/Mistral-7B-Instruct-v0.3"):
        self.model_id = model_id

    def generate_json(self, *args, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            "HuggingFace Mistral wiring pending — Stage 3 agent path will "
            "land first, then this client reuses that integration."
        )


# ---------------------------------------------------------------------------
# Default client — picked up by Stage 4 node. Swap by setting the module-level
# attribute (or via a config later).
# ---------------------------------------------------------------------------

DEFAULT_CLIENT: LLMClient = MockLLMClient()


# ---------------------------------------------------------------------------
# Helpers for the Mock (parsing back the structured user prompt)
# ---------------------------------------------------------------------------

_RISK_LINE_RE = re.compile(
    r"HIGH:\s*(\d+).*?MEDIUM:\s*(\d+).*?LOW:\s*(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_TYPES_LINE_RE = re.compile(
    r"(HIGH|MEDIUM)\s+clause\s+types?\s*:\s*([^\n]+)",
    re.IGNORECASE,
)


def _extract_risk_signal(user_prompt: str) -> dict[str, Any]:
    """Pull the risk-summary block back out of the prompt for the Mock.

    Tolerant: if the prompt format changes, returns zero counts so the
    Mock still produces valid JSON.
    """
    match = _RISK_LINE_RE.search(user_prompt)
    high = int(match.group(1)) if match else 0
    medium = int(match.group(2)) if match else 0
    low = int(match.group(3)) if match else 0

    high_types: list[str] = []
    medium_types: list[str] = []
    for tier, types_csv in _TYPES_LINE_RE.findall(user_prompt):
        types = [t.strip() for t in types_csv.split(",") if t.strip()]
        if tier.upper() == "HIGH":
            high_types = types
        else:
            medium_types = types

    return {
        "total": high + medium + low,
        "high": high,
        "medium": medium,
        "low": low,
        "high_types": high_types,
        "medium_types": medium_types,
    }


def _first_meaningful_sentence(user_prompt: str, max_chars: int = 220) -> str:
    """Pull the contract's opening sentence out of the user prompt body.

    Stage 4's user prompt embeds the contract text after a recognizable
    "Contract text:" header. We grab the first sentence-ish chunk of that
    block to seed the Mock summary with something the contract actually says.
    """
    marker = "Contract text:"
    idx = user_prompt.find(marker)
    if idx == -1:
        return "The contract has been reviewed."
    body = user_prompt[idx + len(marker):].strip()
    # Trim whitespace, take up to first period or max_chars
    body = " ".join(body.split())
    period = body.find(". ")
    end = period + 1 if 0 < period < max_chars else max_chars
    snippet = body[:end].strip()
    if not snippet:
        return "The contract has been reviewed."
    return snippet[0].upper() + snippet[1:] + ("." if not snippet.endswith(".") else "")
