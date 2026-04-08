"""
DeBERTa QA model wrapper for clause extraction + classification.

Handles model loading, tokenization, and raw prediction.
No training or evaluation logic — see train.py and evaluate.py.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class DeBERTaQAModel:
    """Wrapper around HuggingFace DeBERTa for question answering.

    Loads the model and tokenizer from a local path or HuggingFace hub.
    Provides a predict() method that returns answer spans with scores.
    """

    def __init__(self, model_path: str, device: int = -1) -> None:
        """Initialize model and tokenizer.

        Args:
            model_path: Path to fine-tuned model dir or HuggingFace model ID.
            device: -1 for CPU, 0+ for GPU index.
        """
        raise NotImplementedError

    def predict(
        self,
        question: str,
        context: str,
    ) -> dict[str, Any]:
        """Run QA prediction for a single question-context pair.

        Args:
            question: Clause-type query (e.g. "Highlight the parts related to...").
            context: Full contract text.

        Returns:
            Dict with 'answer', 'score', 'start', 'end' keys.
        """
        raise NotImplementedError

    def predict_batch(
        self,
        questions: list[str],
        contexts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Run QA prediction for a batch of question-context pairs.

        Args:
            questions: List of clause-type queries.
            contexts: List of contract texts (same length as questions).
            batch_size: Inference batch size.

        Returns:
            List of prediction dicts.
        """
        raise NotImplementedError
