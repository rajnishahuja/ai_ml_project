import sys
sys.path.insert(0, '/home/rajnish/ai_apps/aiml')

from src.common.data_loader import load_cuad_dataset, preprocess_for_qa
from transformers import AutoTokenizer

# Load dataset and tokenizer
ds = load_cuad_dataset()
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# Tokenize just 2 examples to see what happens
small_batch = ds["train"].select(range(2))
result = preprocess_for_qa(small_batch, tokenizer)

print(f"Input keys: {list(result.keys())}")
print(f"Number of windows created: {len(result['input_ids'])}")
print(f"Tokens per window: {len(result['input_ids'][0])}")
print(f"Start positions: {result['start_positions']}")
print(f"End positions: {result['end_positions']}")

# Verify: decode the answer span from the first window
first_start = result["start_positions"][0]
first_end = result["end_positions"][0]
if first_start > 0:
    answer_tokens = result["input_ids"][0][first_start:first_end + 1]
    print(f"\nDecoded answer: '{tokenizer.decode(answer_tokens)}'")
    print(f"Original answer: '{small_batch[0]['answers']['text'][0]}'")
