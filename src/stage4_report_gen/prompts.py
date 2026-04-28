"""
Prompt builder for the Stage 4 Mistral call.

Stage 4 makes ONE Mistral call per contract: the entire contract text plus
a compressed risk summary go in; a JSON object with `contract_summary`,
`overall_assessment`, `high_priority_actions`, and `medium_priority_actions`
comes out. That JSON populates the report's "Contract Summary" paragraph
and "Conclusion & Recommendations" section.

Truncation: Mistral-7B has 32K-token context. We cap the contract text at
`MAX_CONTRACT_CHARS` (≈12K tokens at 4 chars/token) to leave headroom for
system prompt + risk summary + 1024 output tokens.
"""

from __future__ import annotations

from typing import Iterable

# ~12K-token budget for the contract body; leaves room for prompt + output.
MAX_CONTRACT_CHARS = 48_000


SYSTEM_PROMPT = """\
You are a senior legal-risk analyst assisting a commercial counterparty in
reviewing a contract before signing. You have been given the full contract
text and a risk-classification summary that identifies which clauses were
flagged as HIGH, MEDIUM, or LOW risk by an upstream model.

Your job is to produce a concise written briefing for a non-lawyer business
reader. Output STRICT JSON matching this schema:

{
  "contract_summary":        "<1 paragraph, ~150 words, plain English: what \
this contract is about, the parties' roles, key economic terms>",
  "overall_assessment":      "<1-2 sentences: net risk posture for the \
signing party>",
  "high_priority_actions":   ["<short imperative recommendation>", "..."],
  "medium_priority_actions": ["<short imperative recommendation>", "..."]
}

Rules:
- Output ONLY the JSON object. No preamble, no markdown fences.
- Each action item should be a concrete recommendation a contracts manager
  could act on (negotiate X, request carve-out for Y, escalate to legal).
- If a tier has no clauses, return an empty list for that tier.
- Do not invent clause types or risk levels not present in the input.
- Do not provide legal advice — frame everything as items to confirm with
  counsel.\
"""


def _format_risk_summary(grouped: dict[str, Iterable]) -> str:
    """Compress the per-tier clause buckets into a tight prompt block.

    Embeds counts and the unique clause_types per tier — enough for Mistral
    to ground its conclusion without re-scanning the full clause list.
    """
    high = list(grouped.get("HIGH", []))
    medium = list(grouped.get("MEDIUM", []))
    low = list(grouped.get("LOW", []))

    def _types(clauses) -> list[str]:
        seen: list[str] = []
        for c in clauses:
            ct = getattr(c, "clause_type", None) or (
                c.get("clause_type") if isinstance(c, dict) else None
            )
            if ct and ct not in seen:
                seen.append(ct)
        return seen

    lines = [
        f"Risk classification summary — "
        f"HIGH: {len(high)}, MEDIUM: {len(medium)}, LOW: {len(low)}",
    ]
    if high:
        lines.append("HIGH clause types: " + ", ".join(_types(high)))
    if medium:
        lines.append("MEDIUM clause types: " + ", ".join(_types(medium)))
    if low:
        lines.append("LOW clause types: " + ", ".join(_types(low)))
    return "\n".join(lines)


def build_summary_and_conclusion_prompt(
    contract_text: str,
    grouped_clauses: dict[str, Iterable],
    *,
    max_contract_chars: int = MAX_CONTRACT_CHARS,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the Stage 4 Mistral call.

    Args:
        contract_text: Full contract text as extracted by Stage 1
            (`state.contract_text`). Truncated at `max_contract_chars` to
            stay under Mistral-7B's context window.
        grouped_clauses: Output of `aggregator.group_by_risk_level()` — dict
            with keys HIGH / MEDIUM / LOW, each a list of clauses.
        max_contract_chars: Soft cap on the contract body (default 48K).

    Returns:
        (system_prompt, user_prompt) both as strings. Pass straight into
        `LLMClient.generate_json`.
    """
    text = contract_text or ""
    truncated = len(text) > max_contract_chars
    if truncated:
        text = text[:max_contract_chars].rstrip() + " [... truncated ...]"

    risk_block = _format_risk_summary(grouped_clauses)
    truncation_note = (
        "\nNote: the contract text below was truncated for length. Base the "
        "summary on the available portion; do not speculate about omitted text."
        if truncated else ""
    )

    user_prompt = (
        f"{risk_block}\n"
        f"{truncation_note}\n"
        f"\n"
        f"Contract text:\n"
        f"{text}\n"
    )
    return SYSTEM_PROMPT, user_prompt
