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


                                                                            ###########################################
                                                                ##########     CHANGE???                                #########################
                                                                            ###########################################


TEMPERATURES = [0.2, 0.5, 1.0, 1.5, 2.0, 2.5]

ORDERS = [
    'chicken or pasta',
    'pasta or chicken',
]

# Price pairs applied as (price of option 1, price of option 2).
# Add None to the list if you want to run combinations without prices.
PRICE_PAIRS = [(5, 20), (20, 5), (10, 15), (15,10), (5, 10), (10, 5)]

# ─────────────────────────────────────────────────────────────────────────────

output_name = model_id.split("/")[-1].replace("-", "_")
remember_tag = "with_memory" if remember else "no_memory"

# ── Auth & Model loading (done once) ──────────────────────────────────────────
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


def is_valid_response(text):
    """Check if 'chicken' or 'pasta' is in the response using exact, regex, or fuzzy matching."""
    if not text:
        return False

    candidates = ["chicken", "pasta"]
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


def build_prompt(order, price_pair=None):
    """Return the prompt string and the extra filename tag for this combination."""
    order_split = order.split(' or ')

    if price_pair is not None:
        price_1, price_2 = price_pair
        prompt = (
            f"You are on a flight and can choose between {order}. "
            f"The {order_split[0]} costs ${price_1} and the {order_split[1]} costs ${price_2}. "
            f"Reply with only one of the two, without any punctuation."
        )
        extra = f"_flight_{price_1}_{price_2}"
    else:
        prompt = f"You are on a flight and can choose between {order}. Reply with only one of the two, without any punctuation."
        extra = "_flight"

    return prompt, extra


def run_single(order, temp, price_pair, terminators):
    """Run iterations until num_iter valid responses are generated."""
    prompt, extra = build_prompt(order, price_pair)
    file_path = f"{order}_{remember_tag}_{output_name}_{temp}{extra}.csv"

    if os.path.exists(file_path):
        print(f"\n  [SKIP – file exists] {file_path}")
        return

    print(f"\n{'=' * 70}")
    print(f"  order={order!r}  temp={temp}  prices={price_pair}")
    print(f"  prompt: {prompt}")
    print(f"  output: {file_path}")
    print(f"{'=' * 70}")

    responses = []

    # Setup initial messages
    if remember:
        messages = [
            {"role": "system", "content": "You are a consistent chatbot that remembers previous turns."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = [{"role": "user", "content": prompt}]

    valid_count = 0
    attempts = 0

    # Ensure exactly `num_iter` correct responses are collected
    while valid_count < num_iter and attempts < MAX_ATTEMPTS:
        attempts += 1
        print(f"  Valid {valid_count}/{num_iter} (Attempt {attempts}/{MAX_ATTEMPTS})")

        try:
            outputs = pipeline(
                messages,
                max_new_tokens=20,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                generation_config=None,
            )

            output = outputs[0]["generated_text"][-1]["content"].strip()

            if is_valid_response(output):
                print(f"    → {output}")
                valid_count += 1
                responses.append({"iteration": valid_count, "attempt": attempts, "response": output})

                if remember:
                    messages.append({"role": "assistant", "content": output})
                    if valid_count < num_iter:
                        messages.append({"role": "user", "content": prompt})
            else:
                # Discard invalid output and try again with the same message history
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
combinations = list(itertools.product(TEMPERATURES, ORDERS, PRICE_PAIRS))
total = len(combinations)

print(f"\nRunning {total} combinations "
      f"({len(TEMPERATURES)} temps × {len(ORDERS)} orders × {len(PRICE_PAIRS)} price pairs)\n")

failed = []

for idx, (temp, order, price_pair) in enumerate(combinations, 1):
    print(f"\n[{idx}/{total}]", end="")
    try:
        run_single(order, temp, price_pair, terminators)
    except Exception as e:
        print(f"\n  ❌ Combination (temp={temp}, order={order!r}, prices={price_pair}) "
              f"failed entirely and was skipped.\n  Reason: {e}")
        failed.append((temp, order, price_pair))

print("\n\nAll done! ✅")
if failed:
    print(f"\n⚠️  {len(failed)} combination(s) failed and were skipped:")
    for temp, order, price_pair in failed:
        print(f"    temp={temp}, order={order!r}, prices={price_pair}")
else:
    print("All combinations completed successfully.")