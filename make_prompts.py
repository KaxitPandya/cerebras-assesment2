"""Generate prompts from GSM8K dataset."""
from datasets import load_dataset

# Load GSM8K test split (MIT licensed)
ds = load_dataset("openai/gsm8k", "main", split="test")

# Get first 25 questions
prompts = [ex["question"] for ex in ds.select(range(25))]

# Save with delimiter
with open("prompts.txt", "w", encoding="utf-8") as f:
    f.write("\n\n---\n\n".join(prompts))

print(f"Saved {len(prompts)} prompts to prompts.txt")
print(f"First prompt: {prompts[0][:100]}...")
