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
max_memory = {0: "12GiB", "cpu": "400GiB"}
MAX_ATTEMPTS = num_iter * 5  # Safeguard against infinite loops

# ── Variables to sweep ────────────────────────────────────────────────────────

TEMPERATURES = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.5]

WORDS = ["imphdy", "mkzlyg", "gdwaab"]

# Price pairs applied as (price of option 1, price of option 2).
# Add None to the list if you want to run combinations without prices.
PRICE_PAIRS = [(10, 10)]

# ── Build experiment conditions ───────────────────────────────────────────────

def build_conditions(words):
    """
    Returns a list of (word1, word2, unavailable) tuples covering both
    experiment types:

    Type A — with unavailable word (6 conditions):
        For each word as the unavailable one, both orderings of the other two.
        e.g. ('imphdy', 'mkzlyg', 'gdwaab'), ('mkzlyg', 'imphdy', 'gdwaab'), ...

    Type B — plain binary choice, no unavailable word (6 conditions):
        All pairs × both orderings, unavailable=None.
        e.g. ('imphdy', 'mkzlyg', None), ('mkzlyg', 'imphdy', None), ...
    """
    conditions = []

    # Type A: unavailable word present
    for unavailable in words:
        active = [w for w in words if w != unavailable]
        conditions.append((active[0], active[1], unavailable))
        conditions.append((active[1], active[0], unavailable))

    # Type B: plain binary, all unique pairs × both orderings
    for w1, w2 in itertools.combinations(words, 2):
        conditions.append((w1, w2, None))
        conditions.append((w2, w1, None))

    return conditions

CONDITIONS = build_conditions(WORDS)

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
    max_memory=max_memory,
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


def is_valid_response(text, candidates):
    """
    Returns True if one of the two active candidate words is found in the
    response. The unavailable word is never in candidates, so it can never
    pass — no extra filtering needed.
    """
    if not text:
        return False

    text = str(text).lower()

    # 1. Exact match
    if text in candidates:
        return True

    # 2. Word-boundary regex
    for c in candidates:
        if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', text):
            return True

    # 3. Plain substring
    for c in candidates:
        if c in text:
            return True

    # 4. Fuzzy match (handles slight typos)
    best_score = 0.0
    for word in re.split(r'[^a-zA-Z]+', text):
        if not word:
            continue
        for c in candidates:
            score = SequenceMatcher(None, word.lower(), c).ratio()
            if score > best_score:
                best_score = score

    return best_score >= 0.75


def build_prompt(word1, word2, unavailable, price_pair=None):
    """
    Returns (prompt_string, filename_extra_tag).

    With prices, the two active words get a price each; the unavailable word
    (if present) is mentioned without a price.

    Type A + prices:
        "Choose from word1 ($p1) and word2 ($p2). Unavailable is unavailable
         at the moment. Reply with only one of the two available options."
        filename tag: _random_no_<unavailable>_<p1>_<p2>

    Type A, no prices:
        "Choose from word1 and word2. Unavailable is unavailable at the moment.
         Reply with only one of the two available options."
        filename tag: _random_no_<unavailable>

    Type B + prices:
        "Choose between word1 ($p1) and word2 ($p2). Reply with only one of the two."
        filename tag: _random_<p1>_<p2>

    Type B, no prices:
        "Choose between word1 and word2. Reply with only one of the two."
        filename tag: _random
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

    if unavailable is not None:
        prompt = (
            f"Choose between {w1_str} and {w2_str}. "
            f"{unavailable.capitalize()} is unavailable at the moment. "
            f" Reply with only one of the two."
        )
        extra = f"_random_no_{unavailable}{price_tag}"
    else:
        prompt = (
            f"Choose between {w1_str} and {w2_str}. "
            f"Reply with only one of the two."
        )
        extra = f"_random{price_tag}"

    return prompt, extra


def run_single(word1, word2, unavailable, temp, price_pair, terminators):
    """Run iterations until num_iter valid responses are generated."""
    candidates = [word1, word2]
    prompt, extra = build_prompt(word1, word2, unavailable, price_pair)

    order_str = f"{word1}_or_{word2}"
    file_path = f"{order_str}_{remember_tag}_{output_name}_{temp}{extra}.csv"

    if os.path.exists(file_path):
        print(f"\n  [SKIP – file exists] {file_path}")
        return

    exp_type = f"Type A (unavailable={unavailable!r})" if unavailable else "Type B (plain binary)"
    print(f"\n{'=' * 70}")
    print(f"  {exp_type}")
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

            if is_valid_response(output, candidates):
                print(f"    → {output}")
                valid_count += 1
                responses.append({
                    "iteration": valid_count,
                    "attempt": attempts,
                    "response": output,
                    "experiment_type": "A_unavailable" if unavailable else "B_plain",
                    "unavailable": unavailable if unavailable else "",
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

n_type_a = sum(1 for c in CONDITIONS if c[2] is not None)
n_type_b = sum(1 for c in CONDITIONS if c[2] is None)

print(f"\nConditions: {len(CONDITIONS)} total  "
      f"({n_type_a} Type A with unavailable word, {n_type_b} Type B plain binary)")
print(f"Running {total} combinations "
      f"({len(TEMPERATURES)} temps × {len(CONDITIONS)} conditions × {len(PRICE_PAIRS)} price pairs)\n")

print("Type A (word1, word2, unavailable):")
for c in CONDITIONS:
    if c[2] is not None:
        print(f"  {c}")
print("Type B (word1, word2, no unavailable):")
for c in CONDITIONS:
    if c[2] is None:
        print(f"  {c[:2]}")
print()

failed = []

for idx, (temp, (word1, word2, unavailable), price_pair) in enumerate(combinations, 1):
    print(f"\n[{idx}/{total}]", end="")
    try:
        run_single(word1, word2, unavailable, temp, price_pair, terminators)
    except Exception as e:
        print(
            f"\n  ❌ Combination (temp={temp}, word1={word1!r}, word2={word2!r}, "
            f"unavailable={unavailable!r}, prices={price_pair}) failed.\n  Reason: {e}"
        )
        failed.append((temp, word1, word2, unavailable, price_pair))

print("\n\nAll done! ✅")
if failed:
    print(f"\n⚠️  {len(failed)} combination(s) failed and were skipped:")
    for temp, w1, w2, unav, pp in failed:
        print(f"    temp={temp}, {w1} or {w2}, unavailable={unav!r}, prices={pp}")
else:
    print("All combinations completed successfully.")