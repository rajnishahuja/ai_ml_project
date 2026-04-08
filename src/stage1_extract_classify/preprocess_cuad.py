import json
import logging
import os
from pathlib import Path

from datasets import DatasetDict, Dataset, load_from_disk
from transformers import AutoTokenizer
"""
Preprocessing script for CUAD dataset.

Converts CUAD_v1.json → tokenized dataset for QA training.

NOTE:
- This is a one-time offline step
- Output is saved using `datasets.save_to_disk`
- Training pipeline loads preprocessed data directly
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# =========================
# CONFIG
# =========================
MODEL_NAME = "microsoft/deberta-base"
CACHE_PATH = "./tokenized_cuad"

# =========================
# CACHE VALIDATION
# =========================
def cache_is_valid(path: str) -> bool:
    """
    Checks whether a previously saved tokenized dataset exists and is loadable.
    A directory that exists but is empty or corrupted will return False.
    """
    try:
        load_from_disk(path)
        logger.info(f"✅ Valid cache found at: {path}")
        return True
    except Exception as e:
        logger.warning(f"Cache at '{path}' is invalid or corrupted ({e}). Reprocessing.")
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
        raise FileNotFoundError(
            "CUAD_v1.json not found. Set the CUAD_JSON environment variable "
            "or place the file in the current directory."
        )

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
    logger.info(f"Loaded {len(flat)} QA pairs")

    # Split: 80% train / 10% val / 10% test
    split = flat.train_test_split(test_size=0.10, seed=42)
    train_val = split["train"]
    test = split["test"]

    # 10% of total ≈ 11.1% of the 90% remainder
    VAL_RATIO = 0.10 / 0.90
    split = train_val.train_test_split(test_size=VAL_RATIO, seed=42)
    train = split["train"]
    val = split["test"]

    dataset = DatasetDict({
        "train": train,
        "validation": val,
        "test": test,
    })

    logger.info(f"Final sizes → train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return dataset

# =========================
# PREPROCESSING
# =========================
def preprocess_for_qa(examples: dict, tokenizer: AutoTokenizer) -> dict:
    """
    Tokenizes QA examples with sliding window (stride=128) and maps
    character-level answer spans to token-level start/end positions.

    CUAD answers often have duplicate annotations; we use the first
    answer only for the training span label.
    """
    tokenized = tokenizer(
        [q.strip() for q in examples["question"]],
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    answers = examples["answers"]

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):

        # Guard: missing offsets
        if offsets is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        sequence_ids = tokenized.sequence_ids(i)
        if sequence_ids is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        # Find the token range that covers the context (sequence_id == 1)
        try:
            ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
            ctx_end = len(sequence_ids) - 1 - next(
                j for j, s in enumerate(reversed(sequence_ids)) if s == 1
            )
        except StopIteration:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Unanswerable / impossible questions → label (0, 0)
        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Use the first annotation (CUAD duplicates are intentional)
        char_start = answer["answer_start"][0]
        char_end = char_start + len(answer["text"][0])

        # Guard: window doesn't cover the answer at all → label (0, 0)
        if offsets[ctx_start] is None or offsets[ctx_end] is None:
            start_positions.append(0)
            end_positions.append(0)
            continue

        if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
            start_positions.append(0)
            end_positions.append(0)
            continue

        # Walk forward until the token's start offset exceeds char_start;
        # the answer begins at the last token whose start is still <= char_start.
        token_start = ctx_start
        while (
            token_start < ctx_end
            and offsets[token_start + 1][0] <= char_start
        ):
            token_start += 1

        # Walk backward until the token's end offset is less than char_end;
        # the answer ends at the last token whose end is still >= char_end.
        token_end = ctx_end
        while (
            token_end > ctx_start
            and offsets[token_end - 1][1] >= char_end
        ):
            token_end -= 1

        start_positions.append(token_start)
        end_positions.append(token_end)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions

    return tokenized

# =========================
# MAIN
# =========================
def main():

    if cache_is_valid(CACHE_PATH):
        logger.info("Skipping preprocessing — delete the cache folder to rerun.")
        return

    dataset = load_cuad_dataset()

    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    logger.info("Starting preprocessing (this will take a few minutes)...")

    tokenized = dataset.map(
        preprocess_for_qa,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=os.cpu_count(),   # use all available CPU cores
        remove_columns=dataset["train"].column_names,
    )

    logger.info(f"Saving tokenized dataset to: {CACHE_PATH}")
    tokenized.save_to_disk(CACHE_PATH)

    # Save tokenizer alongside the dataset so Colab loads the exact same version
    tokenizer_path = os.path.join(CACHE_PATH, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to: {tokenizer_path}")

    logger.info("✅ DONE! Upload the entire folder to Google Drive / Colab.")


if __name__ == "__main__":
    main()
