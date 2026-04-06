"""
Stage 1+2: Clause Extraction & Classification (Combined)
=========================================================
Uses DeBERTa-base fine-tuned on CUAD in native QA (span-extraction) format.
One model extracts clause boundaries AND assigns clause types simultaneously.

Architecture Decision (from plan):
  - Extraction + Classification merged into single model
  - CUAD native SQuAD-style QA format: question = "Where is the X clause?"
  - Evaluated on two dimensions: extraction quality + classification quality
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DefaultDataCollator,
    pipeline,
)

from sklearn.metrics import accuracy_score, f1_score, classification_report

import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

from constants import CUAD_CLAUSE_TYPES, CUAD_QUESTION_TEMPLATES, QUESTION_TO_CLAUSE_TYPE

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ClauseObject:
    """Output object for a single extracted + classified clause."""
    clause_id: str
    clause_text: str
    clause_type: str
    start_pos: int
    end_pos: int
    confidence: float
    document_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class ExtractionResult:
    """Full result for one contract document."""
    document_id: str
    clauses: list = field(default_factory=list)

    def to_dict(self):
        return {
            "document_id": self.document_id,
            "clauses": [c.to_dict() if isinstance(c, ClauseObject) else c for c in self.clauses],
        }

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------

def load_cuad_dataset() -> DatasetDict:
    """
    Load CUAD from HuggingFace in native SQuAD-style QA format.
    CUAD: 510 contracts, 13K annotations, 41 categories. License: CC BY 4.0.
    """
    logger.info("Loading CUAD dataset from HuggingFace…")
    dataset = load_dataset("theatticusproject/cuad")
    logger.info(f"CUAD loaded. Splits: {list(dataset.keys())}")
    return dataset


def preprocess_for_qa(examples, tokenizer, max_length=512, doc_stride=128):
    """
    Tokenize CUAD examples in SQuAD QA format.
    Handles long contracts via sliding window (doc_stride).

    Each example: { question, context, answers: {text, answer_start} }
    Returns tokenized features with start/end position labels.
    """
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    tokenized = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    answers = examples["answers"]

    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        # Locate the sequence (context) tokens
        sequence_ids = tokenized.sequence_ids(i)
        ctx_start = next(j for j, s in enumerate(sequence_ids) if s == 1)
        ctx_end = len(sequence_ids) - 1 - next(
            j for j, s in enumerate(reversed(sequence_ids)) if s == 1
        )

        # No answer → CLS token
        if not answer["answer_start"] or len(answer["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        char_start = answer["answer_start"][0]
        char_end = char_start + len(answer["text"][0])

        # Check if answer is within this window
        if offsets[ctx_start][0] > char_end or offsets[ctx_end][1] < char_start:
            start_positions.append(0)
            end_positions.append(0)
            continue

        token_start = ctx_start
        while token_start <= ctx_end and offsets[token_start][0] <= char_start:
            token_start += 1
        start_positions.append(max(ctx_start, token_start - 1))

        token_end = ctx_end
        while token_end >= ctx_start and offsets[token_end][1] >= char_end:
            token_end -= 1
        end_positions.append(min(ctx_end, token_end + 1)) 

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    return tokenized


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def fine_tune_deberta(
    model_name: str = "microsoft/deberta-base",
    output_dir: str = "./models/stage1_2_deberta",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    doc_stride: int = 128,
    fp16: bool = True,
):
    """
    Fine-tune DeBERTa-base on CUAD QA task.

    Model choice (from plan):
      - DeBERTa-base (184M params): superior disentangled attention,
        5.4x better than BERT on CUAD benchmarks. VRAM: 8-16 GB.
      - Alternative: Legal-BERT (110M) — swap model_name if preferred.

    Training config targets Azure ML compute or A100 backup.
    """
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    dataset = load_cuad_dataset()

    # CUAD has no native val split — carve 10% off train
    train_val = dataset["train"].train_test_split(test_size=0.10, seed=42)
    dataset = DatasetDict({
        "train":      train_val["train"],
        "validation": train_val["test"],   # held-out val (~2,200 examples)
        "test":       dataset["test"],     # never touched until final eval
    })

    logger.info("Preprocessing dataset…")
    fn_kwargs = {"tokenizer": tokenizer, "max_length": max_length, "doc_stride": doc_stride}
    tokenized = dataset.map(
        preprocess_for_qa,
        fn_kwargs=fn_kwargs,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    logger.info(f"Tokenized sizes — train: {len(tokenized['train'])}, "
            f"val: {len(tokenized['validation'])}, test: {len(tokenized['test'])}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=fp16 and torch.cuda.is_available(),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",  # swap to "wandb" if desired
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DefaultDataCollator(),
    )

    logger.info("Starting fine-tuning…")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Model saved to {output_dir}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

class ClauseExtractorClassifier:
    """
    Stage 1+2 inference pipeline.
    Runs all 41 CUAD query templates against a contract and returns
    a list of ClauseObject instances — one per detected clause.

    Usage:
        extractor = ClauseExtractorClassifier("./models/stage1_2_deberta")
        clauses = extractor.extract(contract_text, doc_id="contract_001")
        print(json.dumps([c.to_dict() for c in clauses], indent=2))
    """

    def __init__(self, model_path: str, device: int = -1):
        """
        Args:
            model_path: Path to fine-tuned model dir (or HuggingFace hub name).
            device: -1 for CPU, 0+ for GPU index.
        """
        logger.info(f"Loading Stage 1+2 model from: {model_path}")
        self.qa_pipeline = pipeline(
            "question-answering",
            model=model_path,
            tokenizer=model_path,
            device=device,
            handle_impossible_answer=True,  # returns "" when clause absent
        )
        self.clause_types = CUAD_CLAUSE_TYPES
        self.question_templates = CUAD_QUESTION_TEMPLATES

    def extract(
        self,
        contract_text: str,
        doc_id: str = "unknown",
        confidence_threshold: float = 0.2,
    ) -> list[ClauseObject]:
        """
        Run all 41 clause-type queries against the contract.

        Adds:
        - Exact + overlap-based deduplication
        - Keeps highest-confidence span when overlaps occur
        """
        clauses = []
        inputs = [
            {"question": self.question_templates[ct], "context": contract_text}
            for ct in self.clause_types
        ]

        logger.info(f"Running {len(inputs)} queries for doc: {doc_id}")
        results = self.qa_pipeline(inputs, batch_size=16)

        for idx, (clause_type, result) in enumerate(zip(self.clause_types, results)):
            answer_text = result.get("answer", "").strip()
            score = result.get("score", 0.0)

            if not answer_text or score < confidence_threshold:
                continue

            # Use offsets
            start = result.get("start", -1)
            end = result.get("end", -1)

            if start == -1:
                start = contract_text.find(answer_text)
                end = start + len(answer_text) if start != -1 else -1
                if start == -1:
                    logger.warning(
                        f"Could not locate answer text in doc '{doc_id}' "
                        f"for clause '{clause_type}': '{answer_text[:50]}'"
                    )
                    continue

            new_clause = ClauseObject(
                clause_id=f"{doc_id}_{clause_type.replace(' ', '_')}_{idx:04d}",
                clause_text=answer_text,
                clause_type=clause_type,
                start_pos=start,
                end_pos=end,
                confidence=round(score, 4),
                document_id=doc_id,
            )

            # ---------------------------
            # 🔧 Deduplication logic
            # ---------------------------
            keep = True
            to_remove = []

            for i, existing in enumerate(clauses):
                # Exact duplicate span
                if (
                    existing.start_pos == new_clause.start_pos
                    and existing.end_pos == new_clause.end_pos
                ):
                    if new_clause.confidence > existing.confidence:
                        to_remove.append(i)
                    else:
                        keep = False
                    continue

                # Overlap check
                overlap = min(existing.end_pos, new_clause.end_pos) - max(existing.start_pos, new_clause.start_pos)

                if overlap > 0:
                    # If significant overlap (>50%), treat as duplicate
                    smaller_len = min(
                        existing.end_pos - existing.start_pos,
                        new_clause.end_pos - new_clause.start_pos,
                    )
                    if smaller_len > 0 and overlap / smaller_len > 0.5:
                        if new_clause.confidence > existing.confidence:
                            to_remove.append(i)
                        else:
                            keep = False

            # Remove weaker overlapping clauses
            for idx_to_remove in reversed(to_remove):
                clauses.pop(idx_to_remove)

            if keep:
                clauses.append(new_clause)

        # Sort by position
        clauses.sort(key=lambda c: c.start_pos)
        logger.info(f"Extracted {len(clauses)} clauses from {doc_id}")
        return clauses

    def extract_from_file(self, file_path: str, **kwargs) -> ExtractionResult:
        """
        Convenience method: read a .txt contract file and extract clauses.
        For PDF/DOCX, run preprocessing first (see preprocess_contract()).
        """
        text = Path(file_path).read_text(encoding="utf-8")
        doc_id = Path(file_path).stem
        clauses = self.extract(text, doc_id=doc_id, **kwargs)
        return ExtractionResult(document_id=doc_id, clauses=clauses)


# ---------------------------------------------------------------------------
# Contract preprocessing (PDF / DOCX → plain text)
# ---------------------------------------------------------------------------

def preprocess_contract(file_path: str) -> str:
    """
    Convert PDF or DOCX contract to clean plain text.
    Install: pip install pdfplumber python-docx
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")
        text_parts = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        return "\n\n".join(text_parts)

    elif ext in (".docx", ".doc"):
        try:
            from docx import Document
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    elif ext == ".txt":
        return Path(file_path).read_text(encoding="utf-8")

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use PDF, DOCX, or TXT.")


# ---------------------------------------------------------------------------
# Evaluation (two-dimensional, as specified in plan)
# ---------------------------------------------------------------------------
def evaluate_stage1_2(
    model_path: str,
    test_data_path: Optional[str] = None,
    output_path: str = "./results/stage1_2_eval.json",
    confidence_threshold: float = 0.2,
):
    """
    Evaluate the pipeline on two dimensions from the same model output:

    Dimension 1 — Extraction Quality:
        Exact Match (EM) and Token-level F1 (HuggingFace SQuAD metric)

    Dimension 2 — Classification Quality:
        Accuracy, Macro F1, and per-category analysis

    Uses HuggingFace evaluate library.
    """
    squad_metric = evaluate.load("squad")

    # Load CUAD test split if no custom test data
    if test_data_path is None:
        dataset = load_cuad_dataset()
        test_examples = dataset["test"]
    else:
        with open(test_data_path) as f:
            test_examples = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path)
    qa = pipeline("question-answering", model=model, tokenizer=tokenizer,
                  handle_impossible_answer=True)

    predictions, references = [], []
    pred_types, true_types = [], []

    for example in test_examples:
        result = qa({"question": example["question"], "context": example["context"]})
        pred_text = result.get("answer", "").strip()
        true_answers = example.get("answers", {}).get("text", [])

        # Dimension 1: SQuAD EM / F1
        predictions.append({
            "id": example["id"],
            "prediction_text": pred_text,
            "no_answer_probability": 1.0 - result["score"],
        })
        references.append({
            "id": example["id"],
            "answers": example["answers"],
        })

        # Dimension 2: classification — fixed to account for absent clauses
        true_type = _infer_clause_type_from_question(example["question"])

        true_has_clause = bool(true_answers and true_answers[0].strip())
        pred_has_clause = bool(pred_text and result["score"] >= confidence_threshold)

        true_type_label = true_type if true_has_clause else "NO_CLAUSE"
        pred_type = true_type if pred_has_clause else "NO_CLAUSE"

        true_types.append(true_type_label)
        pred_types.append(pred_type)

    # --- Dimension 1 ---
    squad_results = squad_metric.compute(predictions=predictions, references=references)

    # --- Dimension 2 ---
    class_accuracy = accuracy_score(true_types, pred_types)
    class_macro_f1 = f1_score(true_types, pred_types, average="macro", zero_division=0)
    class_report = classification_report(true_types, pred_types, zero_division=0)

    results = {
        "extraction": {
            "exact_match": squad_results["exact_match"],
            "f1": squad_results["f1"],
        },
        "classification": {
            "accuracy": class_accuracy,
            "macro_f1": class_macro_f1,
            "per_class_report": class_report,
        },
    }

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation results:\n{json.dumps(results, indent=2)}")
    logger.info(f"Results saved to {output_path}")
    return results

def _infer_clause_type_from_question(question: str) -> str:
    """
    Infer clause type from a CUAD question string.
    Uses exact reverse lookup against known question templates first,
    falls back to substring matching only if no exact match found.
    """
    # Exact match — robust against substring ambiguity
    if question in QUESTION_TO_CLAUSE_TYPE:
        return QUESTION_TO_CLAUSE_TYPE[question]

    # Fallback — substring match for custom or slightly modified questions
    question_lower = question.lower()
    for ct in CUAD_CLAUSE_TYPES:
        if ct.lower() in question_lower:
            return ct

    return "Unknown"

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1+2: Clause Extraction & Classification")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Fine-tune DeBERTa on CUAD")
    train_parser.add_argument("--model", default="microsoft/deberta-base")
    train_parser.add_argument("--output_dir", default="./models/stage1_2_deberta")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch_size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--no_fp16", action="store_true")

    # Infer
    infer_parser = subparsers.add_parser("infer", help="Run inference on a contract file")
    infer_parser.add_argument("--model_path", required=True)
    infer_parser.add_argument("--contract_file", required=True)
    infer_parser.add_argument("--output_file", default="clauses.json")
    infer_parser.add_argument("--threshold", type=float, default=0.2)
    infer_parser.add_argument("--device", type=int, default=-1)

    # Evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on CUAD test set")
    eval_parser.add_argument("--model_path", required=True)
    eval_parser.add_argument("--output_path", default="./results/stage1_2_eval.json")

    args = parser.parse_args()

    if args.command == "train":
        fine_tune_deberta(
            model_name=args.model,
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=not args.no_fp16,
        )

    elif args.command == "infer":
        contract_text = preprocess_contract(args.contract_file)
        extractor = ClauseExtractorClassifier(args.model_path, device=args.device)
        result = extractor.extract(contract_text, doc_id=Path(args.contract_file).stem,
                                   confidence_threshold=args.threshold)
        output = [c.to_dict() for c in result]
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(output)} clauses to {args.output_file}")

    elif args.command == "evaluate":
        evaluate_stage1_2(
            model_path=args.model_path,
            output_path=args.output_path,
        )

    else:
        parser.print_help()
