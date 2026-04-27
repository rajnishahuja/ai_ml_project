"""
FLAN-T5-base explanation generator for the Stage 4 risk report.

Generates plain-language explanations of why a clause is risky, using a
prompt template that combines the clause text, its type, and the Stage 3
risk level. The Stage 3 agent has already produced a structured
`risk_reason` — this module rewrites it into report-ready prose.

The model is loaded lazily via a thin wrapper class so that:
  - Unit tests / report assembly without the model still work
    (pass `model=None` to use the Stage 3 reason verbatim).
  - Heavyweight transformers imports happen only when the model is
    actually requested.

Schema-agnostic: reads attributes via `_get` so both
`src.common.schema.RiskAssessedClause` (dataclass, uses `risk_explanation`)
and `app.schemas.domain.RiskAssessedClause` (pydantic, uses `risk_reason`)
work.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(clause: Any, *names: str, default: Any = None) -> Any:
    """Read the first attribute that exists on `clause`. Mirrors aggregator."""
    for name in names:
        if hasattr(clause, name):
            value = getattr(clause, name)
            if value is not None:
                return value
        if isinstance(clause, dict) and name in clause:
            value = clause[name]
            if value is not None:
                return value
    return default


def _get_existing_reason(clause: Any) -> str:
    """Pull whatever risk reasoning Stage 3 already produced.

    Schema mismatch: dataclass uses `risk_explanation`, pydantic uses
    `risk_reason`. Try both, returning the first non-empty value.
    """
    for field in ("risk_reason", "risk_explanation"):
        value = _get(clause, field, default="")
        if value:
            return str(value)
    return ""


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = (
    "Explain in one or two clear sentences why the following contract clause "
    "is rated {risk_level} risk. Use plain English. Do not repeat the clause text.\n"
    "\n"
    "Clause type: {clause_type}\n"
    "Clause: {clause_text}\n"
    "Initial assessment: {prior_reason}\n"
    "\n"
    "Explanation:"
)


def build_explanation_prompt(clause: Any) -> str:
    """Build the FLAN-T5 prompt for one clause.

    Truncates the clause text to ~600 characters to keep the prompt within
    FLAN-T5's 512-token input window comfortably (clauses average ~250
    chars; the long tail extends past 2000).
    """
    clause_text = str(_get(clause, "clause_text", default="") or "")
    if len(clause_text) > 600:
        clause_text = clause_text[:600].rsplit(" ", 1)[0] + " ..."

    return _PROMPT_TEMPLATE.format(
        risk_level=str(_get(clause, "risk_level", default="UNKNOWN") or "UNKNOWN"),
        clause_type=str(_get(clause, "clause_type", default="Unknown type") or "Unknown type"),
        clause_text=clause_text,
        prior_reason=_get_existing_reason(clause) or "(none provided)",
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class ExplanationModel:
    """Lazy wrapper around a HuggingFace `text2text-generation` pipeline.

    Holds the FLAN-T5-base pipeline and a cached config so callers don't
    have to thread `max_length` through every call site. Construct once,
    reuse for every clause in a report.
    """

    def __init__(self, pipeline_obj: Any, max_length: int = 200) -> None:
        self._pipeline = pipeline_obj
        self.max_length = max_length

    def generate(self, prompt: str, max_length: Optional[int] = None) -> str:
        out = self._pipeline(
            prompt,
            max_length=max_length or self.max_length,
            do_sample=False,           # deterministic — reports must be reproducible
            num_beams=4,               # mild quality boost for short outputs
        )
        # transformers pipeline returns [{"generated_text": "..."}]
        if isinstance(out, list) and out:
            text = out[0].get("generated_text", "")
        else:
            text = str(out)
        return text.strip()


def load_explanation_model(
    model_name: str = "google/flan-t5-base",
    device: int = -1,
    max_length: int = 200,
) -> ExplanationModel:
    """Load FLAN-T5 as a text2text pipeline.

    Args:
        model_name: HuggingFace model id. Defaults to FLAN-T5-base
            (matches `configs/stage4_config.yaml`).
        device: -1 for CPU, 0+ for the corresponding CUDA device.
        max_length: Default max output tokens for `.generate()`.

    Returns:
        ExplanationModel ready to call `.generate(prompt)`.

    Raises:
        ImportError: if `transformers` is not installed (kept as a soft
            dependency — the rest of Stage 4 works without it).
    """
    try:
        from transformers import pipeline   # type: ignore
    except ImportError as e:
        raise ImportError(
            "transformers is required for the explainer. "
            "Install with `pip install transformers`."
        ) from e

    logger.info("Loading explanation model: %s (device=%d)", model_name, device)
    pipe = pipeline("text2text-generation", model=model_name, device=device)
    return ExplanationModel(pipe, max_length=max_length)


# ---------------------------------------------------------------------------
# Per-clause and bulk explanation
# ---------------------------------------------------------------------------

def generate_explanation(
    clause: Any,
    model: Optional[ExplanationModel] = None,
    max_length: int = 200,
) -> str:
    """Generate a plain-language explanation for one clause.

    Two modes:
      - `model` provided: run FLAN-T5 on the prompt template.
      - `model is None`: pass the Stage 3 `risk_reason` through. Lets the
        rest of Stage 4 produce a complete report without loading a model
        (useful for unit tests, the LangGraph node's first pass, and
        environments without GPU).

    Args:
        clause: A risk-assessed clause (any supported schema).
        model: Loaded `ExplanationModel`, or None to skip generation.
        max_length: Max output tokens. Ignored when `model is None`.

    Returns:
        Explanation string. Always non-empty — falls back to the existing
        risk reason or a generic line if generation produces nothing.
    """
    fallback = _get_existing_reason(clause) or (
        f"This {_get(clause, 'clause_type', default='clause')} clause was "
        f"flagged as {_get(clause, 'risk_level', default='risky')} risk."
    )

    if model is None:
        return fallback

    try:
        prompt = build_explanation_prompt(clause)
        generated = model.generate(prompt, max_length=max_length)
        return generated or fallback
    except Exception as e:                                           # noqa: BLE001
        logger.warning(
            "Explanation generation failed for clause_id=%r: %s. Using fallback.",
            _get(clause, "clause_id"),
            e,
        )
        return fallback


def generate_explanations_batch(
    clauses: list[Any],
    model: Optional[ExplanationModel] = None,
    max_length: int = 200,
) -> list[str]:
    """Generate explanations for a list of clauses.

    Sequential under the hood — FLAN-T5-base is small enough that batched
    generation gives marginal speedup, while sequential makes failure
    isolation per-clause trivial. Revisit if Stage 4 latency becomes a
    bottleneck.
    """
    return [generate_explanation(c, model=model, max_length=max_length) for c in clauses]
