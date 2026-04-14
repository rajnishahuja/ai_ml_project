import sys
sys.path.insert(0, '/home/rajnish/ai_apps/aiml')

from src.common.data_loader import load_cuad_dataset, preprocess_for_qa
from transformers import AutoTokenizer

# Step 1: Load dataset
ds = load_cuad_dataset()
print(f"Train: {len(ds['train']):,}  Test: {len(ds['test']):,}")

# Step 2: Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Step 3: Tokenize the training set
tokenized_train = ds["train"].map(
    preprocess_for_qa,
    fn_kwargs={"tokenizer": tokenizer, "max_length": 512, "doc_stride": 128},
    batched=True,
    remove_columns=ds["train"].column_names,
)

print(f"\nBefore tokenization: {len(ds['train']):,} examples")
print(f"After tokenization:  {len(tokenized_train):,} windows (sliding window expansion)")
print(f"Columns: {tokenized_train.column_names}")
print(f"Sample start_positions (first 10): {tokenized_train['start_positions'][:10]}")
print(f"Sample end_positions (first 10):   {tokenized_train['end_positions'][:10]}")
