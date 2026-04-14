"""
CUAD dataset loading and tokenization for SQuAD-style QA.

Uses theatticusproject/cuad-qa which provides:
  - Pre-flattened QA rows (id, title, context, question, answers)
  - Pre-split train (22,450) and test (4,182) examples
"""

import logging
from datasets import load_dataset, DatasetDict

logger = logging.getLogger(__name__)


def load_cuad_dataset(dataset_name: str = "theatticusproject/cuad-qa") -> DatasetDict:
    """Load CUAD QA dataset from HuggingFace.

    Returns a DatasetDict with 'train' and 'test' splits.
    Each example has: id, title, context, question, answers.

    Args:
        dataset_name: HuggingFace dataset ID.

    Returns:
        DatasetDict with train and test splits.
    """
    logger.info(f"Loading dataset: {dataset_name}")
    ds = load_dataset(dataset_name, trust_remote_code=True)
    logger.info(f"Loaded splits: {list(ds.keys())} | "
                f"Train: {len(ds['train'])}, Test: {len(ds['test'])}")
    return ds


def preprocess_for_qa(examples, tokenizer, max_length=512, doc_stride=128):
    """Tokenize QA examples with sliding window for long contracts.

    Works with HuggingFace dataset.map(batched=True).

    Args:
        examples: Batch dict with 'question', 'context', 'answers' keys.
        tokenizer: HuggingFace tokenizer instance.
        max_length: Max token sequence length (512 for DeBERTa).
        doc_stride: Overlap between sliding windows (128 tokens).

    Returns:
        Tokenized dict with input_ids, attention_mask, start_positions, end_positions.
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    # Step 1: Tokenize question + context with sliding window
    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",       # only truncate the context, not the question
        stride=doc_stride,              # overlap between windows
        return_overflowing_tokens=True,  # create multiple windows per long context
        return_offsets_mapping=True,     # char-to-token position mapping
        padding="max_length",
    )

    # Maps each window back to its original example index
    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    # Step 2: For each window, find where the answer is (or mark as 0)
    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        # Find which tokens belong to the context (sequence_id == 1)
        sequence_ids = tokenized.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        ctx_end = len(sequence_ids) - 1 - next(
            j for j, s in enumerate(reversed(sequence_ids)) if s == 1
        )

        # No answer in this example → point to CLS token (index 0)
        if not answer["answer_start"] or len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Answer's character positions in the original context
        char_start = answer["answer_start"][0]
        char_end = char_start + len(answer["text"][0])

        # Check if answer falls within this window's character range
        if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
            # Answer is outside this window
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Step 3: Convert character positions to token positions
        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= char_start:
            token_start += 1
        start_positions.append(token_start - 1)

        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= char_end:
            token_end -= 1
        end_positions.append(token_end + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized
