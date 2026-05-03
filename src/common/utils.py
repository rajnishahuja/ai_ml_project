"""
Shared utilities: config loading, logging setup, and metric helpers.

All stages use these instead of duplicating utility code.
"""

import json
import logging
import string
from pathlib import Path
from typing import Any

import yaml


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(yaml_path: str) -> dict[str, Any]:
    """Load a YAML config file and return as dict.

    Args:
        yaml_path: Path to YAML config file (e.g. 'configs/stage1_config.yaml').

    Returns:
        Config dict.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {yaml_path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """Configure consistent logging format across all modules.

    Args:
        level: Logging level (default INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


# ---------------------------------------------------------------------------
# Shared metric helpers (used by Stage 1+2 evaluation and baseline)
# ---------------------------------------------------------------------------

def normalize_answer(text: str) -> str:
    """Normalize text for SQuAD-style comparison: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def compute_squad_em_f1(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    """Compute Exact Match and token-level F1 against a list of gold answers.

    Args:
        prediction: Predicted answer string.
        ground_truths: List of acceptable gold answer strings.

    Returns:
        (exact_match, token_f1) tuple, each in [0, 1].
    """
    if not ground_truths:
        return (1.0, 1.0) if not prediction else (0.0, 0.0)

    pred_norm = normalize_answer(prediction)
    best_em, best_f1 = 0.0, 0.0

    for truth in ground_truths:
        truth_norm = normalize_answer(truth)
        em = float(pred_norm == truth_norm)
        best_em = max(best_em, em)

        pred_tokens = pred_norm.split()
        truth_tokens = truth_norm.split()
        common = set(pred_tokens) & set(truth_tokens)

        if common:
            precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
            recall = len(common) / len(truth_tokens) if truth_tokens else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            best_f1 = max(best_f1, f1)

    return best_em, best_f1


def compute_span_iou(
    pred_text: str,
    context: str,
    true_start: int,
    true_end: int,
) -> float:
    """Compute character-level Intersection over Union between predicted and gold spans.

    Args:
        pred_text: Predicted answer text.
        context: Full contract context string.
        true_start: Gold span start character index.
        true_end: Gold span end character index.

    Returns:
        IoU score in [0, 1].
    """
    pred_start = context.find(pred_text)
    if pred_start == -1 or not pred_text:
        return 0.0
    pred_end = pred_start + len(pred_text)

    intersection = max(0, min(pred_end, true_end) - max(pred_start, true_start))
    union = max(pred_end, true_end) - min(pred_start, true_start)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# LLM factory — provider-agnostic chat model instantiation
# ---------------------------------------------------------------------------

def make_llm(cfg: dict):
    """Instantiate a LangChain chat model from config.

    Supported providers (set via cfg['llm_provider']):
      - "openai_compatible" (default): any OpenAI-compatible endpoint
            (llama.cpp, Ollama, vLLM, Azure, OpenAI). Needs agent_base_url.
      - "gemini": Google Gemini via langchain-google-genai. Needs GOOGLE_API_KEY.
      - "anthropic": Anthropic Claude via langchain-anthropic. Needs ANTHROPIC_API_KEY.

    Common config keys: agent_model, agent_max_tokens (default 512), temperature (default 0).
    """
    provider    = cfg.get("llm_provider", "openai_compatible")
    model       = cfg["agent_model"]
    max_tokens  = cfg.get("agent_max_tokens", 512)
    temperature = cfg.get("agent_temperature", 0)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Default: OpenAI-compatible endpoint (llama.cpp, Ollama, OpenAI, Azure …)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url=cfg.get("agent_base_url"),
        api_key=cfg.get("agent_api_key", "none"),
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# JSON serialization for dataclasses
# ---------------------------------------------------------------------------

def save_json(data: Any, path: str) -> None:
    """Save data to JSON file. Supports dataclasses with to_dict().

    Args:
        data: Data to serialize (dict, list, or dataclass with to_dict()).
        path: Output file path.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(data, "to_dict"):
        data = data.to_dict()
    elif isinstance(data, list):
        data = [item.to_dict() if hasattr(item, "to_dict") else item for item in data]

    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {output}")


def load_json(path: str) -> Any:
    """Load JSON file and return parsed data.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON data.
    """
    with open(path) as f:
        return json.load(f)
