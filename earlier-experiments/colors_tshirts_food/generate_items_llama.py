import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import itertools
import pandas as pd
import torch
import transformers
import re
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

remember = False
num_iter = 100
# max_memory = {0: "12GiB", "cpu": "400GiB"}
MAX_ATTEMPTS = num_iter * 5  # Safeguard against infinite loops

# ── Variables to sweep ────────────────────────────────────────────────────────

TEMPERATURES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
# TEMPERATURES = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]

# Price pairs applied as (price of option 1, price of option 2).
PRICE_PAIRS = [
    None,  # No price
    (10, 10),  # Same prices
    (50, 50),
    (10, 15),  # Different prices
    (15, 10),
    (20, 10),
    (10, 20),
    (20, 50),
    (50, 20)
]

# ── Word pairs (Type B only — no unavailable word) ────────────────────────────
# Each tuple is (word1, word2). Both orderings are generated automatically.

BASE_PAIRS = [
    # Colored t-shirts
    ("red t-shirt", "blue t-shirt"),
    ("yellow t-shirt", "blue t-shirt"),
    ("yellow t-shirt", "red t-shirt"),

    # Food
    ("chicken", "pasta"),
]


def build_conditions(pairs):
    """
    For each pair, generate both orderings.
    Returns a list of (word1, word2, None) tuples (Type B only).
    """
    conditions = []
    for w1, w2 in pairs:
        conditions.append((w1, w2, None))
        conditions.append((w2, w1, None))
    return conditions


CONDITIONS = build_conditions(BASE_PAIRS)

# ─────────────────────────────────────────────────────────────────────────────

output_name = model_id.split("/")[-1].replace("-", "_")
remember_tag = "with_memory" if remember else "no_memory"

# ── Auth & Model loading (done once) ─────────────────────────────────────────
login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    # max_memory=max_memory,
    use_safetensors=True,
)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_terminators(pipeline):
    raw = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    return [t for t in raw if t is not None]


def get_keywords_found(text):
    """Extract ALL identifiable keywords found in the text."""
    KEYWORDS = ["red", "blue", "yellow", "chicken", "pasta"]
    found_kws = set()

    for kw in KEYWORDS:
        # Check if the keyword exists as an isolated word
        if re.search(r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])', text):
            found_kws.add(kw)

    return list(found_kws)


def get_valid_choice(text, candidates):
    """
    Tries to match a candidate using ONLY the first 60 characters.
    If both options are mentioned within these 60 characters, it rejects the output.
    """
    if not text:
        return None

    # 1. Force truncate to 60 characters FIRST
    text_lower = str(text).lower()[:60]

    # 2. Try exact full phrase match first (e.g., "red t-shirt")
    found_candidates = [c for c in candidates if c.lower() in text_lower]

    if len(found_candidates) > 1:
        return None  # Both exact phrases mentioned in 60 chars (IGNORE)
    if len(found_candidates) == 1:
        return found_candidates[0]  # Exactly one found (ACCEPT)

    # 3. Fall back to keyword extraction on the 60-character text
    found_kws = get_keywords_found(text_lower)

    # Match the found keywords back to our specific candidates
    relevant_matches = []
    for c in candidates:
        for kw in found_kws:
            if kw in c.lower() and c not in relevant_matches:
                relevant_matches.append(c)

    if len(relevant_matches) > 1:
        return None  # Keywords for both options were mentioned (IGNORE)
    if len(relevant_matches) == 1:
        return relevant_matches[0]  # Keyword for exactly one option found (ACCEPT)

    return None  # Neither option was found (IGNORE)


def sanitize_filename(s):
    """Replace characters unsafe for filenames."""
    return s.replace(" ", "_").replace("-", "")


def build_prompt(word1, word2, unavailable, price_pair=None):
    """
    Returns (prompt_string, filename_extra_tag).
    unavailable is always None here (Type B only), kept for API compatibility.
    """
    if price_pair is not None:
        price_1, price_2 = price_pair
        w1_str = f"{word1} ${price_1}"
        w2_str = f"{word2} ${price_2}"
        price_tag = f"_{price_1}_{price_2}"
    else:
        w1_str = word1
        w2_str = word2
        price_tag = ""

    prompt = (
        f"Choose between {w1_str} and {w2_str}. "
        f"Do not write full sentences. Do not explain your choice. "
        f"Reply with only one of them."
    )
    extra = f"_items{price_tag}"

    return prompt, extra


def run_single(word1, word2, unavailable, temp, price_pair, terminators):
    """Run iterations until num_iter valid responses are generated."""
    candidates = [word1, word2]
    prompt, extra = build_prompt(word1, word2, unavailable, price_pair)

    w1_safe = sanitize_filename(word1)
    w2_safe = sanitize_filename(word2)
    order_str = f"{w1_safe}_or_{w2_safe}"
    file_path = f"{order_str}_{remember_tag}_{output_name}_{temp}{extra}.csv"

    if os.path.exists(file_path):
        print(f"\n  [SKIP – file exists] {file_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"  Type B (plain binary)")
    print(f"  word1={word1!r}  word2={word2!r}  temp={temp}  prices={price_pair}")
    print(f"  prompt: {prompt}")
    print(f"  output: {file_path}")
    print(f"{'=' * 70}")

    responses = []

    if remember:
        messages = [
            {"role": "system", "content": "You are a consistent chatbot that remembers previous turns."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    valid_count = 0
    attempts = 0

    while valid_count < num_iter and attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"  Valid {valid_count}/{num_iter} (Attempt {attempts}/{MAX_ATTEMPTS})")

        try:
            outputs = pipeline(
                messages,
                max_new_tokens=30,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                generation_config=None,
            )

            output = outputs[0]["generated_text"][-1]["content"].strip()

            # Use our new strict function
            choice = get_valid_choice(output, candidates)

            if choice is not None:
                print(f"    → {output}  [matched: {choice!r}]")
                valid_count += 1
                responses.append({
                    "iteration": valid_count,
                    "attempt": attempts,
                    "response": output,
                    "matched_choice": choice,
                    "experiment_type": "B_plain",
                    "unavailable": "",
                })

                if remember:
                    messages.append({"role": "assistant", "content": output})
                    if valid_count < num_iter:
                        messages.append({"role": "user", "content": prompt})
            else:
                print(f"    → [INVALID, IGNORING]: {output}")

        except Exception as e:
            print(f"    🌦️ Error at attempt {attempts}: {e}")

        time.sleep(0.5)

    if valid_count == 0:
        print(f"  ❌ Failed to generate any valid responses — CSV not saved.")
        return

    pd.DataFrame(responses).to_csv(file_path, index=False)

    if valid_count < num_iter:
        print(f"  ⚠️  Only {valid_count}/{num_iter} valid iterations saved (hit max attempts).")
    else:
        print(f"  ✅ Saved {valid_count} valid iterations: {file_path}")


# ── Main sweep ────────────────────────────────────────────────────────────────
terminators = get_terminators(pipeline)

combinations = list(itertools.product(TEMPERATURES, CONDITIONS, PRICE_PAIRS))
total = len(combinations)

print(f"\nConditions: {len(CONDITIONS)} total (all Type B, both orderings)")
print(f"Running {total} combinations "
      f"({len(TEMPERATURES)} temps × {len(CONDITIONS)} conditions × {len(PRICE_PAIRS)} price pairs)\n")

print("Pairs (both orderings):")
for w1, w2, _ in CONDITIONS:
    print(f"  {w1!r} vs {w2!r}")
print()

failed = []

for idx, (temp, (word1, word2, unavailable), price_pair) in enumerate(combinations, 1):
    print(f"\n[{idx}/{total}]", end="")
    try:
        run_single(word1, word2, unavailable, temp, price_pair, terminators)
    except Exception as e:
        print(
            f"\n  ❌ Combination (temp={temp}, word1={word1!r}, word2={word2!r}, "
            f"prices={price_pair}) failed.\n  Reason: {e}"
        )
        failed.append((temp, word1, word2, price_pair))

print("\n\nAll done! ✅")
if failed:
    print(f"\n⚠️  {len(failed)} combination(s) failed and were skipped:")
    for temp, w1, w2, pp in failed:
        print(f"    temp={temp}, {w1!r} or {w2!r}, prices={pp}")
else:
    print("All combinations completed successfully.")