"""
CUAD dataset loading and formatting for SQuAD-style QA.

Handles loading from HuggingFace, converting to flat QA format,
and providing train/test splits.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_cuad_raw(dataset_name: str = "kenlevine/CUAD") -> list[dict]:
    """Load the raw CUAD dataset from HuggingFace.

    The dataset is nested SQuAD format:
        ds[0]['data'][i]['paragraphs'][0]['qas'][j]

    Args:
        dataset_name: HuggingFace dataset ID.

    Returns:
        List of contract dicts, each with 'title' and 'paragraphs'.
    """
    raise NotImplementedError("CUAD loading not yet implemented")


def flatten_to_qa_examples(contracts: list[dict]) -> list[dict]:
    """Flatten nested CUAD structure into flat QA examples.

    Each output example has:
        - id: str
        - question: str
        - context: str (full contract text)
        - answers: {"text": list[str], "answer_start": list[int]}

    Args:
        contracts: Raw CUAD contract list from load_cuad_raw().

    Returns:
        List of flat QA example dicts.
    """
    raise NotImplementedError("QA flattening not yet implemented")


def split_dataset(
    examples: list[dict],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split QA examples into train and test sets by contract (not by QA pair).

    All 41 QA pairs from one contract stay in the same split to avoid
    data leakage (same contract text in both train and test).

    Args:
        examples: Flat QA examples from flatten_to_qa_examples().
        test_ratio: Fraction of contracts for test set.
        seed: Random seed for reproducibility.

    Returns:
        (train_examples, test_examples) tuple.
    """
    raise NotImplementedError("Dataset splitting not yet implemented")


def preprocess_for_qa(
    examples: dict[str, list],
    tokenizer: Any,
    max_length: int = 512,
    doc_stride: int = 128,
) -> dict:
    """Tokenize QA examples with sliding window for long contracts.

    Handles the HuggingFace Trainer map() interface (batched).

    Args:
        examples: Batch of examples with 'question', 'context', 'answers' keys.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Maximum token sequence length.
        doc_stride: Sliding window stride for long contexts.

    Returns:
        Tokenized features dict with start_positions and end_positions.
    """
    raise NotImplementedError("QA tokenization not yet implemented")
