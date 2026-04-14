"""
Learn: How sliding window tokenization works for QA models.

We'll use a simple story (not legal text) to see exactly what happens
when we tokenize a question + long context for a QA model.
"""
import sys
sys.path.insert(0, '/home/rajnish/ai_apps/aiml')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")

# =========================================================================
# PART 1: What is a token?
# =========================================================================
print("=" * 60)
print("PART 1: What is a token?")
print("=" * 60)

sentence = "The quick brown fox jumps over the lazy dog"
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.encode(sentence, add_special_tokens=False)

print(f"\nText:      '{sentence}'")
print(f"Characters: {len(sentence)}")
print(f"Tokens:     {tokens}")
print(f"Token IDs:  {token_ids}")
print(f"Token count: {len(tokens)}")
print(f"\nRatio: {len(sentence)} chars / {len(tokens)} tokens = ~{len(sentence)/len(tokens):.1f} chars per token")

# Longer words get split into sub-tokens
print("\n--- Sub-token example ---")
word = "Indemnification"
tokens = tokenizer.tokenize(word)
print(f"'{word}' → {tokens}  ({len(tokens)} tokens for 1 word)")

# =========================================================================
# PART 2: Question + Context together
# =========================================================================
print("\n" + "=" * 60)
print("PART 2: How question + context get combined")
print("=" * 60)

question = "What color is the cat?"
context = "Alice has a black cat named Whiskers. Bob has a white dog named Rex."
answer_text = "black"
answer_start = context.index("black")  # character position = 12

print(f"\nQuestion: '{question}'")
print(f"Context:  '{context}'")
print(f"Answer:   '{answer_text}' at character {answer_start}")

# Tokenize them together
encoded = tokenizer(question, context, return_offsets_mapping=True)
tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

print(f"\nCombined tokens ({len(tokens)} total):")
seq_ids = encoded.sequence_ids()
for i, (tok, seq_id, offset) in enumerate(zip(tokens, seq_ids, encoded["offset_mapping"])):
    role = {None: "SPECIAL", 0: "QUESTION", 1: "CONTEXT"}[seq_id]
    print(f"  [{i:2d}] {tok:<20} seq_id={str(seq_id):<5} offset={offset}  ({role})")

# =========================================================================
# PART 3: Why we need sliding windows
# =========================================================================
print("\n" + "=" * 60)
print("PART 3: Why we need sliding windows")
print("=" * 60)

# Let's make a context that's too long for a small window
# We'll use max_length=30 tokens to make it visible (instead of 512)
short_context = (
    "Alice has a black cat named Whiskers. "
    "Bob has a white dog named Rex. "
    "Charlie has a green parrot named Polly. "
    "Diana has a brown horse named Star. "
    "Eve has a golden fish named Bubbles."
)
question = "What color is the horse?"
answer_text = "brown"
answer_start = short_context.index("brown")

print(f"\nQuestion: '{question}'")
print(f"Context:  '{short_context}'")
print(f"Answer:   '{answer_text}' at character {answer_start}")

# Without sliding window — just truncate
truncated = tokenizer(question, short_context, max_length=30, truncation=True)
trunc_tokens = tokenizer.convert_ids_to_tokens(truncated["input_ids"])
print(f"\n--- Without sliding window (max_length=30, just truncate) ---")
print(f"Tokens: {trunc_tokens}")
print(f"Decoded: '{tokenizer.decode(truncated['input_ids'], skip_special_tokens=True)}'")
print("⚠️  'brown horse' is CUT OFF — the model can never find the answer!")

# With sliding window
print(f"\n--- With sliding window (max_length=30, stride=10) ---")
windowed = tokenizer(
    question,
    short_context,
    max_length=30,
    truncation="only_second",
    stride=10,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)

sample_map = windowed.pop("overflow_to_sample_mapping")
offsets_all = windowed.pop("offset_mapping")

print(f"Number of windows created: {len(windowed['input_ids'])}\n")

for w in range(len(windowed["input_ids"])):
    tokens = tokenizer.convert_ids_to_tokens(windowed["input_ids"][w])
    seq_ids = windowed.sequence_ids(w)

    # Find context token range
    ctx_tokens = [(i, tok) for i, (tok, s) in enumerate(zip(tokens, seq_ids)) if s == 1]

    if ctx_tokens:
        first_ctx = ctx_tokens[0][0]
        last_ctx = ctx_tokens[-1][0]
        char_from = offsets_all[w][first_ctx][0]
        char_to = offsets_all[w][last_ctx][1]
        ctx_text = short_context[char_from:char_to]
    else:
        char_from, char_to = 0, 0
        ctx_text = ""

    # Does the answer fall in this window?
    answer_end = answer_start + len(answer_text)
    has_answer = (char_from <= answer_start and char_to >= answer_end)

    print(f"  Window {w+1}: chars {char_from}–{char_to} | "
          f"{'✅ ANSWER HERE' if has_answer else '❌ no answer'}")
    print(f"    Context slice: '...{ctx_text[:60]}...'")
    print()

# =========================================================================
# PART 4: Finding the answer's TOKEN position
# =========================================================================
print("=" * 60)
print("PART 4: Converting character position → token position")
print("=" * 60)

# Use the window that has the answer
for w in range(len(windowed["input_ids"])):
    seq_ids = windowed.sequence_ids(w)
    ctx_start_tok = next(j for j, s in enumerate(seq_ids) if s == 1)
    ctx_end_tok = len(seq_ids) - 1 - next(
        j for j, s in enumerate(reversed(seq_ids)) if s == 1
    )

    offsets = offsets_all[w]
    answer_end = answer_start + len(answer_text)

    # Check if answer is in this window
    if offsets[ctx_start_tok][0] > answer_end or offsets[ctx_end_tok][1] < answer_start:
        continue

    # Found it! Walk through tokens to find the answer
    print(f"\nWindow {w+1} contains the answer. Let's find which tokens:")
    print(f"  Answer chars: {answer_start}–{answer_end}")
    print(f"  Context tokens {ctx_start_tok}–{ctx_end_tok}:\n")

    tokens = tokenizer.convert_ids_to_tokens(windowed["input_ids"][w])
    found_start = None
    found_end = None

    for j in range(ctx_start_tok, ctx_end_tok + 1):
        char_range = offsets[j]
        is_answer = (char_range[0] >= answer_start and char_range[1] <= answer_end and char_range != (0, 0))
        marker = " ◄── ANSWER" if is_answer else ""
        print(f"    token[{j:2d}] = '{tokens[j]:<15}' covers chars {char_range[0]:3d}–{char_range[1]:3d}{marker}")
        if is_answer and found_start is None:
            found_start = j
        if is_answer:
            found_end = j

    print(f"\n  → Answer is at token positions {found_start}–{found_end}")
    print(f"  → Decoded: '{tokenizer.decode(windowed['input_ids'][w][found_start:found_end+1])}'")
    print(f"  → Original: '{answer_text}'")
    break

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("""
1. TOKENIZE: Split text into sub-word tokens (~4 chars each)
2. COMBINE:  [CLS] question [SEP] context [SEP] — single sequence
3. WINDOW:   If context is too long, slide across it in overlapping chunks
4. LOCATE:   For each window, find which TOKEN positions the answer occupies
5. LABEL:    start_position + end_position → what the model learns to predict

At inference: model reads a window and predicts start/end token positions.
We decode those tokens back to text → that's the extracted clause.
""")
