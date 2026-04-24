import json
import logging
import os
from pathlib import Path

from datasets import DatasetDict, Dataset, Features, Sequence, Value, load_from_disk
from transformers import AutoTokenizer
from datasets import concatenate_datasets
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "microsoft/deberta-base"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "384"))
STRIDE = int(os.environ.get("STRIDE", "128"))
# NEG_RATIO: number of negative windows per positive window
# Train: 1:2 recommended for CUAD
# Val:   1:2 for reliable early stopping signal
# Test:  unbalanced — natural distribution for fair evaluation
NEG_RATIO = int(os.environ.get("NEG_RATIO", "2"))
CACHE_PATH = f"./tokenized_cuad_{MAX_LENGTH}"

# Sentinel value used during preprocessing to mark "no answer" windows.
# Using -1 instead of 0 because 0 is a valid token position (maps to [CLS]),
# which causes the model to predict [CLS] for all no-answer windows.
# Sentinel is reset to 0 after window balancing, before saving.
NO_ANSWER_SENTINEL = -1


# =========================
# CACHE VALIDATION
# =========================
def cache_is_valid(path: str) -> bool:
    try:
        load_from_disk(path)
        logger.info(f"✅ Valid cache found at: {path}")
        return True
    except Exception:
        return False


# =========================
# LOAD DATASET
# =========================
def load_cuad_dataset() -> DatasetDict:
    candidates = [
        os.environ.get("CUAD_JSON", ""),
        "./CUAD_v1.json",
    ]

    json_path = next((p for p in candidates if p and Path(p).exists()), None)

    if json_path is None:
        raise FileNotFoundError("CUAD_v1.json not found")

    logger.info(f"Loading CUAD dataset from: {json_path}")

    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = {"id": [], "question": [], "context": [], "answers": []}

    for doc in raw["data"]:
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                rows["id"].append(qa["id"])
                rows["question"].append(qa["question"].strip())
                rows["context"].append(context)

                if qa.get("is_impossible") or not qa["answers"]:
                    rows["answers"].append({"text": [], "answer_start": []})
                else:
                    rows["answers"].append({
                        "text": [a["text"] for a in qa["answers"]],
                        "answer_start": [a["answer_start"] for a in qa["answers"]],
                    })

    flat = Dataset.from_dict(rows)
    logger.info(f"Loaded {len(flat)} total QA pairs")

    # =========================
    # SPLIT raw unbalanced data
    # =========================
    split = flat.train_test_split(test_size=0.10, seed=42)
    raw_train_val = split["train"]
    test = split["test"]  # natural distribution — never touched by balancing

    VAL_RATIO = 0.10 / 0.90
    split2 = raw_train_val.train_test_split(test_size=VAL_RATIO, seed=42)
    raw_train = split2["train"]
    val = split2["test"]

    logger.info(f"Pre-balance sizes → train: {len(raw_train)}, val: {len(val)}, test: {len(test)}")

    # =========================
    # BALANCE TRAIN ONLY at QA level, with 1:1 ratio
    # =========================
    logger.info("Balancing training set at QA level...")

    positives = raw_train.filter(lambda x: len(x["answers"]["answer_start"]) > 0)
    negatives = raw_train.filter(lambda x: len(x["answers"]["answer_start"]) == 0)

    logger.info(f"Train → Positives: {len(positives)}, Negatives: {len(negatives)}")

    # Sanity check — no rows should be lost or double-counted
    assert len(positives) + len(negatives) == len(raw_train), (
        f"Balancing filter leaked rows! "
        f"{len(positives)} + {len(negatives)} != {len(raw_train)}"
    )

    # 1:1 ratio at QA level
    negatives = negatives.shuffle(seed=42).select(
        range(min(len(negatives), len(positives)))
    )

    train = concatenate_datasets([positives, negatives])
    train = train.shuffle(seed=42)

    pos = sum(1 for x in train if len(x["answers"]["answer_start"]) > 0)
    neg = len(train) - pos
    logger.info(f"Balanced train → Positives: {pos}, Negatives: {neg}, Ratio: 1:{neg/pos:.2f}")

    dataset = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    logger.info(f"Final sizes → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return dataset


# =========================
# TOKENIZATION
# =========================
def preprocess_for_qa(examples: dict, tokenizer: AutoTokenizer) -> dict:
    tokenized = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    tokenized.pop("token_type_ids", None)
    answers = examples["answers"]

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):

        sequence_ids = tokenized.sequence_ids(i)
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        try:
            ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            ctx_end = len(sequence_ids) - 1 - next(
                j for j, s in enumerate(reversed(sequence_ids)) if s == 1
            )
        except StopIteration:
            # No context tokens in this window — mark as no-answer
            start_positions.append(NO_ANSWER_SENTINEL)
            end_positions.append(NO_ANSWER_SENTINEL)
            continue

        if not answer["answer_start"]:
            # QA example has no answer — true negative
            start_positions.append(NO_ANSWER_SENTINEL)
            end_positions.append(NO_ANSWER_SENTINEL)
            continue

        # CUAD may have multiple occurrences of the same clause type.
        # Find the first answer span that overlaps with this context window.
        target_char_start, target_char_end = None, None
        for a_start, a_text in zip(answer["answer_start"], answer["text"]):
            a_end = a_start + len(a_text)
            if not (offsets[ctx_start][0] > a_end or offsets[ctx_end][1] < a_start):
                target_char_start = a_start
                target_char_end = a_end
                break

        if target_char_start is None:
            # Answer exists but falls outside this window — negative window
            start_positions.append(NO_ANSWER_SENTINEL)
            end_positions.append(NO_ANSWER_SENTINEL)
            continue

        char_start = target_char_start
        char_end = target_char_end

        # Find start token: walk forward until we pass char_start, then step back
        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= char_start:
            token_start += 1
        token_start -= 1

        # Find end token: walk backward until we pass char_end, then step forward
        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= char_end:
            token_end -= 1
        token_end += 1

        # Clamp to context boundaries
        token_start = max(token_start, ctx_start)
        token_end = min(token_end, ctx_end)

        # Filter very short spans — likely tokenization noise
        if token_end - token_start < 2:
            start_positions.append(NO_ANSWER_SENTINEL)
            end_positions.append(NO_ANSWER_SENTINEL)
            continue

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized


# =========================
# WINDOW-LEVEL BALANCING
# =========================
def balance_windows(tokenized: DatasetDict) -> DatasetDict:
    """
    Balance tokenized windows at the window level for train and validation.

    Train:      balanced for good learning signal (NEG_RATIO negatives per positive)
    Validation: balanced for reliable early stopping signal (same NEG_RATIO)
    Test:       left unbalanced — natural distribution for fair final evaluation

    Uses sentinel -1 to unambiguously identify negative windows before reset.
    """

    for split in ["train", "validation"]:
        logger.info(f"Balancing {split} windows...")

        pos_windows = tokenized[split].filter(
            lambda x: x["start_positions"] != NO_ANSWER_SENTINEL,
            num_proc=None,
        )
        neg_windows = tokenized[split].filter(
            lambda x: x["start_positions"] == NO_ANSWER_SENTINEL,
            num_proc=None,
        )

        logger.info(
            f"{split} window balance before — positives: {len(pos_windows)}, "
            f"negatives: {len(neg_windows)}, "
            f"ratio: 1:{len(neg_windows)/max(len(pos_windows), 1):.0f}"
        )

        target_neg = min(len(neg_windows), len(pos_windows) * NEG_RATIO)
        neg_windows = neg_windows.shuffle(seed=42).select(range(target_neg))

        balanced = concatenate_datasets([pos_windows, neg_windows]).shuffle(seed=42)

        logger.info(
            f"{split} window balance after  — positives: {len(pos_windows)}, "
            f"negatives: {len(neg_windows)}, "
            f"total: {len(balanced)}, "
            f"ratio: 1:{NEG_RATIO}"
        )

        tokenized[split] = balanced

    # Test stays unbalanced — log for awareness
    test_pos = sum(1 for x in tokenized["test"]["start_positions"] if x != NO_ANSWER_SENTINEL)
    test_neg = len(tokenized["test"]) - test_pos
    logger.info(
        f"test window balance (unbalanced, natural) — "
        f"positives: {test_pos}, negatives: {test_neg}, "
        f"ratio: 1:{test_neg/max(test_pos,1):.0f}"
    )

    return tokenized


# =========================
# RESET SENTINEL
# =========================
def reset_sentinel(tokenized: DatasetDict) -> DatasetDict:
    """
    Convert sentinel -1 back to 0 after balancing.
    The model expects 0 for no-answer positions, not -1.
    Applied to all splits so val/test are also correctly formatted.
    """
    logger.info("Resetting sentinel values (-1 → 0)...")

    def _reset(example):
        if example["start_positions"] == NO_ANSWER_SENTINEL:
            example["start_positions"] = 0
            example["end_positions"] = 0
        return example

    for split in ["train", "validation", "test"]:
        tokenized[split] = tokenized[split].map(_reset, num_proc=None)

    # Sanity check — no -1 values should remain in any split
    for split in ["train", "validation", "test"]:
        sentinel_count = sum(
            1 for x in tokenized[split]["start_positions"]
            if x == NO_ANSWER_SENTINEL
        )
        assert sentinel_count == 0, \
            f"Sentinel reset failed in {split}: {sentinel_count} remaining"
        logger.info(f"✅ {split}: sentinel reset verified")

    return tokenized


# =========================
# MAIN
# =========================
def main():

    logger.info(f"Config — MAX_LENGTH: {MAX_LENGTH}, STRIDE: {STRIDE}, "
                f"NEG_RATIO: {NEG_RATIO}, CACHE_PATH: {CACHE_PATH}")

    if cache_is_valid(CACHE_PATH):
        logger.info("Skipping preprocessing — delete cache to rerun.")
        return

    dataset = load_cuad_dataset()

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info("Tokenizing dataset...")

    # Explicit feature schema ensures correct tensor types during training.
    # token_type_ids excluded — DeBERTa doesn't use them.
    # start_positions/end_positions use int64 to support sentinel value -1.
    qa_features = Features({
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "start_positions": Value("int64"),
        "end_positions": Value("int64"),
    })

    tokenized = dataset.map(
        preprocess_for_qa,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=None,
        remove_columns=dataset["train"].column_names,
        features=qa_features,
    )

    logger.info(f"Tokenized sizes (before window balancing) — "
                f"train: {len(tokenized['train'])}, "
                f"val: {len(tokenized['validation'])}, "
                f"test: {len(tokenized['test'])}")

    # Step 1: Balance train + val windows using sentinel to correctly identify negatives
    # Test is left unbalanced for fair final evaluation
    tokenized = balance_windows(tokenized)

    # Step 2: Reset sentinel -1 back to 0 for model compatibility
    tokenized = reset_sentinel(tokenized)

    logger.info(f"Final sizes (after window balancing) — "
                f"train: {len(tokenized['train'])}, "
                f"val: {len(tokenized['validation'])}, "
                f"test: {len(tokenized['test'])}")

    # Final sanity check on train positive/negative ratio
    pos = sum(1 for x in tokenized["train"]["start_positions"] if x > 0)
    neg = len(tokenized["train"]) - pos
    logger.info(f"Final train window ratio — positives: {pos}, negatives: {neg}, "
                f"ratio: 1:{neg/max(pos,1):.1f}")

    # Final sanity check on val positive/negative ratio
    val_pos = sum(1 for x in tokenized["validation"]["start_positions"] if x > 0)
    val_neg = len(tokenized["validation"]) - val_pos
    logger.info(f"Final val window ratio   — positives: {val_pos}, negatives: {val_neg}, "
                f"ratio: 1:{val_neg/max(val_pos,1):.1f}")

    logger.info(f"Saving dataset to: {CACHE_PATH}")
    tokenized.save_to_disk(CACHE_PATH)

    tokenizer.save_pretrained(os.path.join(CACHE_PATH, "tokenizer"))

    logger.info("✅ DONE!")


if __name__ == "__main__":
    main()
