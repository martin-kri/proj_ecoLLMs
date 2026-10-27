"""
Llama ladder-experiment script.
=======================================================================
25.8 rework: brought up to parity with OpenAI-25.8.py, but NOT a
feature-for-feature copy — some of what makes that script good is
specifically about being an HTTP API client, which this isn't:

  - NO ThreadPoolExecutor / MAX_WORKERS. The OpenAI script parallelizes
    because it's I/O-bound on network calls. This script is one sequential
    GPU forward pass at a time on one device; threading wouldn't speed
    that up and risks CUDA context issues. Sequential stays sequential.
  - NO API-specific retry/backoff/broken_models machinery (that solves
    rate limits and "this model rejects this parameter" — neither applies
    here). Instead: a small per-job try/except (2 attempts, e.g. for a
    transient CUDA hiccup), a torch.cuda.empty_cache() on failure so one
    bad prompt doesn't cascade into OOM-ing everything after it, then
    skip and move on rather than losing the whole run.

What DID transfer, because it solves a real problem here too:
  - Incremental checkpointing + resume. The price grid below is ~4x bigger
    than before (85 price conditions vs ~21), so a run through Set C alone
    is ~680 forward passes. Losing hours of GPU work to a crash partway
    through would actually cost something now, unlike before.
  - ACTIVE_PROMPT_SETS, defaulting to {3} (Set C), matching what the
    OpenAI-25.8.py comparison currently covers, so results stay
    apples-to-apples across all four models instead of Llama running sets
    the others don't have data for (yet).
  - A "model" column (this script previously had none at all — the
    plotting scripts were silently filling it in via a filename-based
    fallback) and a best-effort "model_revision" column: the closest local
    equivalent to OpenAI's response.model. There's no per-call API response
    to read a resolved snapshot off of, but from_pretrained's resolved
    commit hash (when the Hub reports one) serves the same purpose —
    pinning down exactly which weights answered, in case the "main" branch
    of the model repo ever moves.
  - The same keep_default_na=False / na_values=[""] checkpoint-read fix as
    OpenAI-25.8.py: price_pair="None" (the no-price sentinel string) would
    otherwise silently become NaN on reload and never match a freshly built
    job's "None" string, making the no-price condition un-resumable.

Everything else (build_prompts, the 3-sets-x-4-rungs ladder, CURRENCY /
format_price / price_tag, the RUN_* price-condition switches) is carried
over verbatim — none of it is model-specific.

Install:  pip install torch transformers huggingface_hub pandas
Run:      set HF_TOKEN in .env, then: python LLAMA-25.8.py
"""

import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import math
import pandas as pd
import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # tqdm not installed, fall back to a no-op wrapper
    def tqdm(iterable, **kwargs):
        return iterable

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

model_id = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = model_id.split("/")[-1]  # used both as the CSV "model" value and in filenames

# Single fixed temperature, matching the OpenAI pipeline: one forward pass,
# one softmax at T=1.0, one row per prompt. If you ever want other
# temperatures, reconstruct them from score_word1/score_word2 the same way
# OpenAI-25.8.py's temperature_sweep.py does — same softmax-of-logits math,
# no need to re-run the model.
TEMPERATURE = 1.0

# ── Item config ─────────────────────────────────────────────────────────────
# Anchor item for the ladder phase. When you move on to the item-variation
# grid (vary color only, vary item only, vary both, food domain), that's a
# different phase with its own prompt wording, not a drop-in swap here, the
# ladder text below ("choose a {color} t-shirt...") is written for colored
# t-shirts specifically. Extending it to non-color items (chicken vs pasta)
# means writing new templates, not just changing this constant.
ITEM_NOUN = "t-shirt"

COLOR_PAIRS = [
    ("red", "green"),  # <- current anchor pair, per the meeting notes
    # Add more pairs here later for the "vary color, same item" phase, e.g.:
    # ("red", "blue"),
    # ("yellow", "crimson"),
]

CONDITIONS = []
for c1, c2 in COLOR_PAIRS:
    CONDITIONS.append((c1, c2))
    CONDITIONS.append((c2, c1))  # both orderings, needed to measure order bias

# ── Currency config (verbatim — not model-specific) ─────────────────────────
CURRENCY = "USD"  # "USD" or "EUR"


def format_price(price):
    if CURRENCY == "USD":
        return f"${price}"
    elif CURRENCY == "EUR":
        return f"{price}€"
    raise ValueError(f"Unknown CURRENCY: {CURRENCY!r}")


def price_tag(price):
    """Returns ' (for $10)' / ' (for 10€)', or '' if price is None."""
    return f" (for {format_price(price)})" if price is not None else ""


# ── Price condition config ──────────────────────────────────────────────────
# True so a fresh run reproduces the full published CSV. (The original run had
# this off and backfilled the 8 no-price rows from an earlier run instead.)
RUN_NO_PRICE = True
RUN_SAME_PRICE = True
RUN_FIXED_GAP = True
RUN_PCT_GAP = True

# "Same price" phase: tested as (p, p) for every value in this list.
SAME_PRICE_LEVELS = [10]

# 25.8: extended per meeting decision — "bigger gaps, more prices" for
# Llama specifically. $2 stays out (see OpenAI-25.8.py's comment above its
# own PCT_GAP_BASES: round(2 * 1.10) == 2, collapsing the smallest pct_gap
# into a duplicate of same_price instead of a real gap — the same rounding
# problem would hit here too, so it's simply not in this list).
#
# Heads up on overlap, not a bug, just a cost/compute-time note: at base=40
# (gap=20 vs pct=0.50 -> both land on 60), base=100 (all three gaps 10/20/50
# happen to equal 10%/20%/50% of 100 exactly -> all three overlap with all
# three pcts), and base=500 (gap=50 vs pct=0.10 -> both land on 550), the
# fixed_gap and pct_gap conditions produce byte-identical prompt text, sent
# as two separate forward passes tagged with different price_type metadata.
# That's inherent to the specific numbers requested, not something this
# script tries to silently dedupe — same category of thing as the rung-1
# duplication across sets, just at different base prices this time.
# Re-checked after adding 700/750/850/900/1200 below: no new overlaps and
# no degenerate (base == high) collapses introduced by any of the five.
FIXED_GAP_BASES = [10, 40, 100, 180, 500, 700, 750, 800, 850, 900, 1000, 1200]
FIXED_GAPS = [10, 20, 50]

PCT_GAP_BASES = [10, 40, 100, 180, 500, 700, 750, 800, 850, 900, 1000, 1200]
PCT_GAPS = [0.10, 0.20, 0.50]


def build_price_pairs():
    """Returns a list of (price_pair, metadata_dict) so every row in the CSV
    can be filtered/grouped by price-condition type without having to
    reverse-engineer it from the raw tuple later."""
    conditions = []

    if RUN_NO_PRICE:
        conditions.append((None, {"price_type": "none", "price_base": None, "price_gap_or_pct": None}))

    if RUN_SAME_PRICE:
        for p in SAME_PRICE_LEVELS:
            conditions.append(((p, p), {"price_type": "same", "price_base": p, "price_gap_or_pct": 0}))

    if RUN_FIXED_GAP:
        for base in FIXED_GAP_BASES:
            for gap in FIXED_GAPS:
                high = base + gap
                meta = {"price_type": "fixed_gap", "price_base": base, "price_gap_or_pct": gap}
                conditions.append(((base, high), meta))
                conditions.append(((high, base), meta))

    if RUN_PCT_GAP:
        for base in PCT_GAP_BASES:
            for pct in PCT_GAPS:
                high = round(base * (1 + pct))
                meta = {"price_type": "pct_gap", "price_base": base, "price_gap_or_pct": pct}
                conditions.append(((base, high), meta))
                conditions.append(((high, base), meta))

    return conditions


PRICE_CONDITIONS = build_price_pairs()

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Prompt builder: the three complexity ladders (verbatim — not model-specific) ─
def build_prompts(color1, color2, price_pair=None):
    p1, p2 = (None, None) if price_pair is None else price_pair
    t1, t2 = price_tag(p1), price_tag(p2)

    # Rung 1 is the literal same text in all three sets, that's the shared
    # anchor everything else branches from. Defined once so all three copies
    # are guaranteed byte-identical, not just similar.
    rung1_text = f"Choose a {color1} {ITEM_NOUN}{t1} or a {color2} {ITEM_NOUN}{t2}. One word"

    prompts = []

    # ── Set 1: more constraints ──────────────────────────────────────────
    prompts.append({"prompt_set": 1, "rung": 1, "style": "set1_r1", "text": rung1_text})
    prompts.append({
        "prompt_set": 1, "rung": 2, "style": "set1_r2",
        "text": (
            f"Choose between a {color1} {ITEM_NOUN}{t1} and a {color2} {ITEM_NOUN}{t2}. "
            f"Reply with exactly one word only, no punctuation."
        ),
    })
    prompts.append({
        "prompt_set": 1, "rung": 3, "style": "set1_r3",
        "text": (
            f"Choose between a {color1} {ITEM_NOUN}{t1} and a {color2} {ITEM_NOUN}{t2}. "
            f"Reply with exactly one word only, no punctuation, no explanation, and no other text at all."
        ),
    })
    prompts.append({
        "prompt_set": 1, "rung": 4, "style": "set1_r4",
        "text": (
            f"Choose between a {color1} {ITEM_NOUN}{t1} and a {color2} {ITEM_NOUN}{t2}. "
            f"Reply with exactly one word only, either '{color1}' or '{color2}', and nothing else at all: "
            f"no punctuation, no explanation, and no other text at all."
        ),
    })

    # ── Set 2: more context ──────────────────────────────────────────────
    prompts.append({"prompt_set": 2, "rung": 1, "style": "set2_r1", "text": rung1_text})
    prompts.append({
        "prompt_set": 2, "rung": 2, "style": "set2_r2",
        "text": (
            f"You are on a long flight. Choose a {color1} {ITEM_NOUN}{t1} or a {color2} {ITEM_NOUN}{t2}. "
            f"Reply with one word."
        ),
    })
    prompts.append({
        "prompt_set": 2, "rung": 3, "style": "set2_r3",
        "text": (
            f"You are on a long flight and just spilled a drink on your only {ITEM_NOUN}. "
            f"Choose a {color1} {ITEM_NOUN}{t1} or a {color2} {ITEM_NOUN}{t2}. Reply with one word."
        ),
    })
    prompts.append({
        "prompt_set": 2, "rung": 4, "style": "set2_r4",
        "text": (
            f"You are on a long flight and just spilled a drink on your only {ITEM_NOUN}. "
            f"You land in 30 minutes for an important meeting. "
            f"Choose a {color1} {ITEM_NOUN}{t1} or a {color2} {ITEM_NOUN}{t2}. Reply with one word."
        ),
    })

    # ── Set 3: more length, no new information ("the politician") ───────
    prompts.append({"prompt_set": 3, "rung": 1, "style": "set3_r1", "text": rung1_text})
    prompts.append({
        "prompt_set": 3, "rung": 2, "style": "set3_r2",
        "text": (
            f"You are asked to choose between a {color1} {ITEM_NOUN}{t1} and a {color2} {ITEM_NOUN}{t2}. "
            f"Reply with one word."
        ),
    })
    prompts.append({
        "prompt_set": 3, "rung": 3, "style": "set3_r3",
        "text": (
            f"You are being asked to choose between two {ITEM_NOUN}s, one is {color1}{t1} "
            f"and the other one is {color2}{t2}. Please reply with only one word."
        ),
    })
    prompts.append({
        "prompt_set": 3, "rung": 4, "style": "set3_r4",
        "text": (
            f"You are being asked to carefully consider a choice between two {ITEM_NOUN}s, "
            f"one is in the color {color1}{t1} and the other one is in the color {color2}{t2}. "
            f"Please give me your answer using only a single word."
        ),
    })

    return prompts


ALL_STYLE_NAMES = [p["style"] for p in build_prompts("red", "green", None)]
STYLE_TO_SET = {p["style"]: p["prompt_set"] for p in build_prompts("red", "green", None)}


# ── Which complexity ladder(s) to run ───────────────────────────────────────
# 1 = Set A (more constraints), 2 = Set B (more context), 3 = Set C (pure
# length, "the politician"). Set C only, matching OpenAI-25.8.py's current
# scope, so the 4-model comparison stays apples-to-apples. Add 1 and/or 2
# back in here later to widen the run — no other code needs to change.
ACTIVE_PROMPT_SETS = {3}


ACTIVE_COLOR_PAIRS = COLOR_PAIRS
ACTIVE_PRICE_CONDITIONS = PRICE_CONDITIONS
# set2_r1/set3_r1 text == set1_r1 text — only matters if ACTIVE_PROMPT_SETS
# ever includes more than one set at once.
ACTIVE_STYLES = {style for style, pset in STYLE_TO_SET.items() if pset in ACTIVE_PROMPT_SETS}

ACTIVE_CONDITIONS = []
for c1, c2 in ACTIVE_COLOR_PAIRS:
    ACTIVE_CONDITIONS.append((c1, c2))
    ACTIVE_CONDITIONS.append((c2, c1))

# How often to flush the checkpoint during the run (in completed jobs), plus
# an unconditional flush at the end regardless of this threshold — a run
# that finishes with fewer new rows than this (e.g. a small resumed batch)
# would otherwise leave no checkpoint behind at all, and the next run would
# have nothing to resume from.
CHECKPOINT_EVERY = 20

# ── Auth & Model loading ──────────────────────────────────────────────────────
login(token=HF_TOKEN)

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
model.eval()

# Best-effort local equivalent of OpenAI's response.model: pins down exactly
# which snapshot of the weights answered, in case the repo's "main" branch
# ever moves. transformers sets this during from_pretrained's hub
# resolution when it has a commit to report; None if it doesn't (e.g. a
# fully offline / pre-cached load), which is an honest "unknown", not a bug.
MODEL_REVISION = getattr(getattr(model, "config", None), "_commit_hash", None)
print(f"Model revision (best-effort): {MODEL_REVISION!r}")


# ── Token ID Lookup ───────────────────────────────────────────────────────────
def get_target_token_ids(word):
    base_word = word.split()[0]  # defensive: colors here are single words already
    variants = [
        base_word.lower(),
        " " + base_word.lower(),
        base_word.capitalize(),
        " " + base_word.capitalize(),
    ]
    ids = []
    for v in variants:
        enc = tokenizer.encode(v, add_special_tokens=False)
        if len(enc) > 0:
            ids.append(enc[0])
    return list(set(ids))


unique_colors = set(c for pair in COLOR_PAIRS for c in pair)
TOKEN_DICT = {color: get_target_token_ids(color) for color in unique_colors}


# ── Job construction ─────────────────────────────────────────────────────
def build_jobs():
    jobs = []
    for color1, color2 in ACTIVE_CONDITIONS:
        for price_pair, price_meta in ACTIVE_PRICE_CONDITIONS:
            for p in build_prompts(color1, color2, price_pair):
                if p["style"] not in ACTIVE_STYLES:
                    continue
                jobs.append({
                    "color1": color1,
                    "color2": color2,
                    "word1": f"{color1} {ITEM_NOUN}",
                    "word2": f"{color2} {ITEM_NOUN}",
                    "price_pair": str(price_pair) if price_pair else "None",
                    "price_type": price_meta["price_type"],
                    "price_base": price_meta["price_base"],
                    "price_gap_or_pct": price_meta["price_gap_or_pct"],
                    "prompt_style": p["style"],
                    "prompt_set": p["prompt_set"],
                    "rung": p["rung"],
                    "prompt_text": p["text"],
                })
    return jobs


def job_key(job):
    # price_type is part of the key: see job_key() in OpenAI-25.8.py
    return (job["word1"], job["word2"], job["price_pair"], job["price_type"], job["prompt_style"])


def run_job(job, attempt=1, max_attempts=2):
    ids_1 = TOKEN_DICT[job["color1"]]
    ids_2 = TOKEN_DICT[job["color2"]]

    try:
        messages = [{"role": "user", "content": job["prompt_text"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            next_token_logits = outputs.logits[0, -1, :]

        score_1 = max([next_token_logits[tid].item() for tid in ids_1])
        score_2 = max([next_token_logits[tid].item() for tid in ids_2])

        exp_1 = math.exp(score_1 / TEMPERATURE)
        exp_2 = math.exp(score_2 / TEMPERATURE)
        rel_prob_1 = exp_1 / (exp_1 + exp_2)
        rel_prob_2 = 1.0 - rel_prob_1
        full_probs = torch.softmax(next_token_logits / TEMPERATURE, dim=-1)

        vocab_prob_1 = sum([full_probs[tid].item() for tid in ids_1])
        vocab_prob_2 = sum([full_probs[tid].item() for tid in ids_2])
        target_coverage = vocab_prob_1 + vocab_prob_2

        top5_probs_vals, top5_ids = torch.topk(full_probs, k=5)
        top5_tokens_list = [tokenizer.decode([tid]) for tid in top5_ids.tolist()]
        top5_probs_list = [round(p, 4) for p in top5_probs_vals.tolist()]

    except Exception as e:
        torch.cuda.empty_cache()  # in case this was an OOM — don't let it cascade
        if attempt < max_attempts:
            print(f"[RETRY] {job['prompt_style']} {job['word1']} vs {job['word2']} "
                  f"({job['price_pair']}): {e} — retrying once.")
            return run_job(job, attempt=attempt + 1, max_attempts=max_attempts)
        print(f"[WARN] Giving up on {job['prompt_style']} {job['word1']} vs {job['word2']} "
              f"({job['price_pair']}) after {max_attempts} attempts: {e}")
        return None

    return [{
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "word1": job["word1"],
        "word2": job["word2"],
        "color1": job["color1"],
        "color2": job["color2"],
        "price_pair": job["price_pair"],
        "price_type": job["price_type"],
        "price_base": job["price_base"],
        "price_gap_or_pct": job["price_gap_or_pct"],
        "prompt_style": job["prompt_style"],
        "prompt_set": job["prompt_set"],
        "rung": job["rung"],
        "temperature": TEMPERATURE,
        "score_word1": round(score_1, 4),
        "score_word2": round(score_2, 4),
        "rel_prob_word1": round(rel_prob_1, 4),
        "rel_prob_word2": round(rel_prob_2, 4),
        "vocab_prob_word1": round(vocab_prob_1, 4),
        "vocab_prob_word2": round(vocab_prob_2, 4),
        "target_coverage": round(target_coverage, 4),
        "top5_tokens": str(top5_tokens_list),
        "top5_probs": str(top5_probs_list),
        "prompt_text": job["prompt_text"],
    }]


# ── Main execution ───────────────────────────────────────────────────────
def main():
    jobs = build_jobs()
    print(f"Total forward passes queued: {len(jobs)}  (1 row each, T={TEMPERATURE} baseline only)")
    print(f"Active prompt sets: {sorted(ACTIVE_PROMPT_SETS)}  |  Active styles: {sorted(ACTIVE_STYLES)}  "
          f"|  Price conditions active: {len(ACTIVE_PRICE_CONDITIONS)}")

    checkpoint_path = os.path.join(OUTPUT_DIR, f"logprobs_ALL_PROMPTS_{MODEL_NAME}_checkpoint.csv")

    done_keys = set()
    all_rows = []
    if os.path.exists(checkpoint_path):
        # keep_default_na=False + na_values=[""]: see OpenAI-25.8.py's main()
        # for the full explanation — otherwise price_pair="None" (the
        # no-price sentinel) silently becomes NaN on reload and never
        # matches a freshly built job's "None" string.
        prev = pd.read_csv(checkpoint_path, keep_default_na=False, na_values=[""])
        if "model" not in prev.columns:
            # Checkpoint predates this column — every row in it came from
            # this same hardcoded model_id, so backfill rather than losing
            # the ability to resume off it.
            prev["model"] = MODEL_NAME
        if "model_revision" not in prev.columns:
            prev["model_revision"] = None
        all_rows = prev.to_dict("records")
        done_keys = set(
            zip(prev["word1"], prev["word2"], prev["price_pair"], prev["price_type"], prev["prompt_style"])
        )
        print(f"Resuming: {len(done_keys)} conditions already in checkpoint, skipping those.")

    pending = [j for j in jobs if job_key(j) not in done_keys]
    print(f"Conditions left to run: {len(pending)}")

    completed_since_flush = 0
    for job in tqdm(pending, desc="Running"):
        result_rows = run_job(job)
        if result_rows:
            all_rows.extend(result_rows)
            completed_since_flush += 1
            if completed_since_flush >= CHECKPOINT_EVERY:
                pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
                completed_since_flush = 0

    # Always flush once more at the end, even if this run had fewer than
    # CHECKPOINT_EVERY new rows (e.g. a small resumed batch) — otherwise a
    # run that never crosses the threshold leaves no checkpoint behind, and
    # the *next* run has nothing to resume from.
    if all_rows:
        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\n[WARN] No rows were collected, nothing to save or summarize.")
        return

    master_path = os.path.join(OUTPUT_DIR, f"logprobs_ALL_PROMPTS_{MODEL_NAME}.csv")
    df.to_csv(master_path, index=False)
    print(f"\nDone! Master output saved to: {master_path}")

    for style, group in df.groupby("prompt_style"):
        style_path = os.path.join(OUTPUT_DIR, f"logprobs_{style}_{MODEL_NAME}.csv")
        group.to_csv(style_path, index=False)
        print(f"Saved: {style_path}")


if __name__ == "__main__":
    main()
