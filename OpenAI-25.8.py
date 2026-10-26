"""
OpenAI ladder-experiment script.
=======================================================================
25.8 revision:
  - Added full "gpt-5.4" to MODELS (nano/mini were already here). Pilot-
    tested on 25.8 (4 calls, PILOT_MODE=True, MODELS temporarily just
    ["gpt-5.4"]): reasoning_effort="none" and logprobs/top_logprobs were
    both accepted with no fatal errors, and top_logprobs self-adjusted
    20 -> 5 on the first call, same cap as nano/mini — see
    TOP_LOGPROBS_STARTING_CAP below, now set to the confirmed value.
  - Dropped price_base=2 from FIXED_GAP_BASES/PCT_GAP_BASES: with a 10%
    pct_gap, round(2 * 1.10) == 2, so that condition silently collapsed
    into a duplicate of same_price instead of a real gap. See the comment
    above PCT_GAP_BASES.
  - Added ACTIVE_PROMPT_SETS so a run can target only some of the 3
    complexity ladders instead of all 12 styles — see the comment above it.
    Currently set to {3} (Set C) only.
  - CSV output is now T=1.0 baseline ONLY, one row per condition, not the
    old 13-value TEMPERATURES sweep. Other temperatures are reconstructed
    from this baseline's score_word1/score_word2 by a separate script
    (temperature_sweep.py) that does the exact same rescale math as a
    post-processing pass over the CSV — no re-querying the API. This keeps
    the master CSV from ballooning to 13x its row count when you only ever
    look at one temperature at a time in practice.
  - reasoning_effort is now a real per-job dimension (REASONING_EFFORTS
    list) instead of a hardcoded "none" string, and every row now also
    carries prompt_tokens/completion_tokens/reasoning_tokens/total_tokens
    from the response's usage object (extract_usage()). This is what
    actually answers "is reasoning_effort='none' avoiding hidden reasoning
    cost" instead of just inferring it from full_answer not looking
    truncated — reasoning_tokens will show a real (usually 0) number.
    Checkpoint loading backfills reasoning_effort="none" for any older
    checkpoint CSV that predates this column, so existing data isn't
    orphaned; those older rows just won't have usage numbers.
  - Added "gpt-5.6-luna" to MODELS (nano/mini/gpt-5.4 were already here) —
    UNCONFIRMED, never pilot-tested. Priced like the nano tier
    ($0.20/$1.20 per 1M in/out as of this writing), so its top_logprobs
    starting cap is set to 5 as an informed guess, not a verified one —
    self-adjusts down further if wrong, same as every other model here.
  - Price grid extended to match LLAMA-25.8.py exactly: FIXED_GAP_BASES /
    PCT_GAP_BASES grew from 5 bases x 1 gap size to 12 bases x 3 gap sizes
    each, so the two scripts produce directly comparable data. See the
    comment above PCT_GAP_BASES for the (unchanged) overlap notes.
  - MAX_COMPLETION_TOKENS raised 5 -> 16, and query_with_retry now detects
    the "model output limit was reached" error specifically and retries
    with a bumped per-call token budget instead of identically repeating a
    doomed call — two rows (gpt-5.4-mini and gpt-5.4, different prices/
    styles) got silently dropped by this on the first full 12-base run
    before the fix. See OUTPUT_LIMIT_PHRASES below.

This is a MERGE, not a new design:

  - The API-calling machinery (single call per condition at T=1.0 + full
    13-temperature reconstruction in post-processing, top_logprobs cap
    auto-adjustment, retry/backoff, broken-model detection, threaded
    execution, checkpointing) is carried over UNCHANGED from the original
    OpenAI port script (openai_logprob_experiment.py).

  - The experiment DESIGN (what prompts exist, what gets varied, which
    switches control what runs) is carried over from the HuggingFace
    Llama-3.1-8B-Instruct ladder script: ITEM_NOUN + COLOR_PAIRS, the
    CURRENCY switch, the RUN_NO_PRICE / RUN_SAME_PRICE / RUN_FIXED_GAP /
    RUN_PCT_GAP price-condition switches, and the "3 sets x 4 rungs"
    complexity-ladder prompt builder.

  So: same infra, new experiment. Below are the specific things that
  changed or needed a decision in the merge.

WHAT'S IDENTICAL TO THE ORIGINAL OPENAI SCRIPT (see that script for the
full rationale, only summarized here):
  1. One API call per condition at temperature=1.0, top_p=1.0. logprob
     differences at T=1 equal raw logit differences (shared normalizer
     cancels), so other temperatures CAN be reconstructed in Python
     afterward without separate API calls — this script itself only ever
     saves the T=1.0 row though; see temperature_sweep.py for the sweep.
  2. top_logprobs is capped at 5 for all three gpt-5.4 variants (nano/mini/
     full — confirmed empirically for all three now, docs say 20).
     `top_logprobs_cap` self-adjusts down per model if a call ever reports
     a stricter limit.
  3. Word matching is STRING-based (`get_word_variant_strings` /
     `token_matches_word`), not tokenizer-based, because the API doesn't
     expose a tokenizer. The HF script matched on real token ids; this
     can't, so results aren't perfectly apples-to-apples with the HF
     numbers even for identical prompt text.
  4. PILOT_MODE defaults to True. Confirm `logprobs` / `top_logprobs` /
     `reasoning_effort="none"` are accepted, and eyeball costs, before
     flipping it to False and running the full grid.

WHAT CHANGED FROM THE HF LADDER SCRIPT:
  - CURRENCY / format_price / price_tag, the RUN_* price switches, and
    build_prompts() (the 3 sets x 4 rungs ladder) are copied verbatim.
    They're currency/wording logic, nothing about them is
    model-specific, so they didn't need to change at all.
  - TOKEN_DICT (tokenizer-based) is gone; replaced by the string-based
    variant matching from the OpenAI script (point 3 above). Matching is
    done on the bare color word (e.g. "red"), same as the HF script's
    TOKEN_DICT was keyed by color, not by "red t-shirt".
  - The HF script had no cost concern, so it just ran everything
    COLOR_PAIRS x PRICE_CONDITIONS x all 12 ladder prompts. This script
    reintroduces PILOT_MODE + ACTIVE_* subsetting from the OpenAI script
    so you can dry-run cheaply first (see the "Which price-condition
    switches" and "cost-control" answers you gave — PILOT_MODE stays,
    only RUN_NO_PRICE is on, ITEM_NOUN stays a single constant rather
    than a list).

ONE COST GOTCHA WORTH KNOWING: rung 1 is byte-identical across all three
sets (set1_r1 == set2_r1 == set3_r1, same for nothing else). If you turn
on all 12 styles, you are sending that exact same prompt 3x per
condition — that's intentional in the HF script (a shared anchor point
across ladders), but here it's 3x the API cost for identical text. Use
ACTIVE_STYLES to drop the duplicates (e.g. keep only set1_r1) if that
redundancy isn't something you want to pay for.

Install:  pip install openai pandas tqdm
Run:      export OPENAI_API_KEY=sk-...
          python openai_ladder_experiment.py
"""

import os
import re
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

import pandas as pd
from openai import (
    OpenAI,
    RateLimitError,
    BadRequestError,
    APIError,
    APIConnectionError,
    APITimeoutError,
)

try:
    from tqdm import tqdm
except ImportError:  # tqdm not installed, fall back to a no-op wrapper
    def tqdm(iterable, **kwargs):
        return iterable

# ── API config ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)

MODELS = ["gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.4", "gpt-5.6-luna"]

# Which reasoning_effort value(s) to run. "none" is the current baseline (all
# existing nano/mini/gpt-5.4 data was run at this value, hardcoded). Add
# "minimal" (or whichever levels a model exposes) here later to compare
# against it — every extra value multiplies total API calls by
# len(REASONING_EFFORTS), since each is a full separate job grid, not a
# free reconstruction like the temperature sweep. top_logprobs_cap is kept
# per-model only (not per reasoning_effort) on the assumption that the cap
# is a property of the model's API surface, not of this parameter — revisit
# if a future effort level ever reports a different cap than "none" did.
REASONING_EFFORTS = ["none"]

MAX_WORKERS = 6          # concurrent requests; lower this if you hit rate limits
MAX_RETRIES = 5
SEED = 42                # best-effort determinism, not guaranteed by the API
MAX_COMPLETION_TOKENS = 16  # only content[0] (the first generated token) is
                            # actually used for scoring; a few extra tokens
                            # just make the CSV easier to eyeball by hand.
                            # 25.8: raised from 5 after two full-grid runs hit
                            # "Could not finish the message because max_tokens
                            # or model output limit was reached" — even with
                            # reasoning_effort="none", the API can apparently
                            # still spend a token or two on hidden reasoning
                            # before the visible answer token, and 5 was too
                            # tight to survive that occasionally. This costs
                            # nothing extra for the ~99% of calls that only
                            # ever use ~4 tokens anyway (billed on tokens
                            # actually used, not the cap) — see OUTPUT_LIMIT_
                            # PHRASES below for what happens if 16 still isn't
                            # enough some day.

# gpt-5.4-nano/mini/full all confirmed at 5 (general docs say 20; each one
# self-adjusted down to 5 on its first call the first time it was run here).
# gpt-5.6-luna is UNCONFIRMED — never queried yet. It's documented as
# "roughly corresponds to the nano tier", so 5 is an informed guess, not a
# verified one; either way this only ever self-adjusts downward if the API
# reports a stricter cap, so a wrong guess costs one retry, not a bad run.
DEFAULT_TOP_LOGPROBS = 5
TOP_LOGPROBS_STARTING_CAP = {"gpt-5.4-nano": 5, "gpt-5.4-mini": 5, "gpt-5.4": 5, "gpt-5.6-luna": 5}
top_logprobs_cap = {m: TOP_LOGPROBS_STARTING_CAP.get(m, DEFAULT_TOP_LOGPROBS) for m in MODELS}
top_logprobs_cap_lock = threading.Lock()

OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Only the T=1.0 baseline is queried and saved here. Other temperatures are
# reconstructed later, from this baseline, by temperature_sweep.py — see
# docstring. Not a list because this script no longer sweeps at all.
BASELINE_TEMPERATURE = 1.0


# ── Item config (from the HF ladder script) ────────────────────────────────
# Kept as a single ITEM_NOUN + COLOR_PAIRS, same as the HF script — not
# generalized to a list of item nouns. Extending to non-color items (e.g.
# chicken vs pasta) means writing new prompt templates in build_prompts(),
# not just editing this constant.
ITEM_NOUN = "t-shirt"

COLOR_PAIRS = [
    ("red", "green"),  # <- current anchor pair
    # Add more pairs here later for the "vary color, same item" phase, e.g.:
    # ("red", "blue"),
    # ("yellow", "crimson"),
]

CONDITIONS = []
for c1, c2 in COLOR_PAIRS:
    CONDITIONS.append((c1, c2))
    CONDITIONS.append((c2, c1))  # both orderings, needed to measure order bias


# ── Currency config (verbatim from the HF script) ──────────────────────────
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


# ── Price condition config (verbatim from the HF script) ──────────────────
RUN_NO_PRICE = True
RUN_SAME_PRICE = True
RUN_FIXED_GAP = True
RUN_PCT_GAP = True

SAME_PRICE_LEVELS = [10]

# $2 base intentionally excluded: with a 10% pct_gap, round(2 * 1.10) ==
# round(2.2) == 2, so the "10%" pct_gap condition collapsed to (2, 2) - a
# duplicate of the same_price condition, not a real gap. Dropped from both
# lists (not just pct_gap) so fixed_gap and pct_gap keep the same set of
# bases for apples-to-apples comparison. The plotting script
# (one-by-one_order-bias-26.8.py) already filters price_base == 2.0 out
# separately, this just stops paying to generate it in the first place.
#
# 25.8: extended to match LLAMA-25.8.py's price grid exactly (same 12 bases,
# same 3 gap sizes each) for a like-for-like comparison. Same overlap note
# as that script: at base=40 (gap=20 vs pct=0.50 -> both land on 60),
# base=100 (all three gaps happen to equal 10%/20%/50% of 100 exactly ->
# all three overlap with all three pcts), and base=500 (gap=50 vs pct=0.10
# -> both land on 550), fixed_gap and pct_gap send byte-identical prompt
# text as two separate API calls tagged with different price_type metadata.
# Verified: no degenerate (base == high) collapses anywhere in this list.
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


# ── Prompt builder: the three complexity ladders (verbatim from the HF script) ─
def build_prompts(color1, color2, price_pair=None):
    p1, p2 = (None, None) if price_pair is None else price_pair
    t1, t2 = price_tag(p1), price_tag(p2)

    # Rung 1 is the literal same text in all three sets — the shared anchor
    # everything else branches from. See the cost note in the module
    # docstring: turning on all three sets' rung 1 means 3 identical API
    # calls per condition.
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
# length, "the politician").
ACTIVE_PROMPT_SETS = {3}


# ── PILOT_MODE toggle (from the OpenAI script, re-applied to the new config) ─
# Flip to False once a pilot run confirms logprobs/reasoning_effort/temperature
# are all accepted for both models, and the cost of a full run looks sane.
PILOT_MODE = False

if PILOT_MODE:
    ACTIVE_COLOR_PAIRS = COLOR_PAIRS[:1]            # just the current anchor pair
    ACTIVE_PRICE_CONDITIONS = PRICE_CONDITIONS[:1]  # just the first enabled price condition
    # One shared anchor rung, one max-constraint rung, both from the active set(s) only.
    ACTIVE_STYLES = {
        f"set{s}_r1" for s in ACTIVE_PROMPT_SETS
    } | {
        f"set{s}_r4" for s in ACTIVE_PROMPT_SETS
    }
    ACTIVE_MODELS = MODELS
    ACTIVE_REASONING_EFFORTS = REASONING_EFFORTS[:1]
else:
    ACTIVE_COLOR_PAIRS = COLOR_PAIRS
    ACTIVE_PRICE_CONDITIONS = PRICE_CONDITIONS
    # set2_r1/set3_r1 text == set1_r1 text (see cost note above) — only matters
    # if ACTIVE_PROMPT_SETS ever includes more than one set at once.
    ACTIVE_STYLES = {style for style, pset in STYLE_TO_SET.items() if pset in ACTIVE_PROMPT_SETS}
    ACTIVE_MODELS = MODELS
    ACTIVE_REASONING_EFFORTS = REASONING_EFFORTS

ACTIVE_CONDITIONS = []
for c1, c2 in ACTIVE_COLOR_PAIRS:
    ACTIVE_CONDITIONS.append((c1, c2))
    ACTIVE_CONDITIONS.append((c2, c1))


# ── Word-variant matching (string-based, since the API gives no tokenizer) ─
# Same utility as the original OpenAI port script. The crimson->red fallback
# is a Llama-specific quirk carried over defensively; it's inert unless you
# add a "crimson" pair to COLOR_PAIRS, at which point re-check it's still
# needed for GPT-5.4 before trusting it.
def get_word_variant_strings(word):
    base = word.split()[0]
    variants = {base.lower(), base.capitalize(), base.upper()}

    if base.lower() == "crimson":
        variants.update({"cr", "Cr", "crim", "Crim", "red", "Red"})
    elif base.lower() in ["red", "blue", "yellow", "pink", "green"]:
        variants.update({base[0].lower(), base[0].upper()})

    return variants


_STRIP_CHARS = ' \t\n"\'.,!?:;'


def token_matches_word(token_str, variants):
    cleaned = token_str.strip(_STRIP_CHARS)
    return cleaned.casefold() in {v.casefold() for v in variants}


# ── Model call + score extraction (unchanged from the OpenAI port script) ──
broken_models = set()
broken_models_lock = threading.Lock()

RANGE_ERROR_RE = re.compile(r"must be (?:less than or equal to|<=)\s*(\d+)", re.IGNORECASE)

FATAL_PHRASES = (
    "is not supported with",
    "is not supported for",
    "does not support",
    "not supported with this model",
    "unsupported_parameter",
    "unsupported parameter",
)

# Seen twice on a full-grid run (gpt-5.4-mini and gpt-5.4, different prices,
# different styles — not tied to one specific prompt): the model spends its
# whole max_completion_tokens budget on hidden reasoning before it ever
# emits a visible answer token, and the API errors instead of truncating.
# Retrying the identical call is pointless (same prompt, same budget, same
# outcome every time) — bump the budget for that one call instead, same
# spirit as the top_logprobs self-adjustment below.
OUTPUT_LIMIT_PHRASES = ("model output limit was reached", "could not finish the message")
OUTPUT_LIMIT_BUMP = 20
OUTPUT_LIMIT_CEILING = 100


def query_model(model, prompt_text, reasoning_effort, max_completion_tokens=None):
    n = top_logprobs_cap[model]
    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt_text}],
        max_completion_tokens=max_completion_tokens or MAX_COMPLETION_TOKENS,
        temperature=1.0,   # unscaled: makes returned logprobs a valid proxy for raw logits
        top_p=1.0,          # no nucleus truncation
        reasoning_effort=reasoning_effort,
        logprobs=True,
        top_logprobs=n,
        seed=SEED,
    ), n


def query_with_retry(model, prompt_text, reasoning_effort):
    last_err = None
    key = (model, reasoning_effort)
    max_tokens_this_call = MAX_COMPLETION_TOKENS
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(model, prompt_text, reasoning_effort, max_tokens_this_call)
        except BadRequestError as e:
            msg = str(e)
            low = msg.lower()

            range_match = RANGE_ERROR_RE.search(msg)
            if range_match and "top_logprobs" in low:
                new_cap = int(range_match.group(1))
                with top_logprobs_cap_lock:
                    if top_logprobs_cap[model] > new_cap:
                        top_logprobs_cap[model] = new_cap
                        print(f"[ADJUST] {model}: top_logprobs cap lowered to {new_cap}, retrying with the new value.")
                continue

            if any(t in low for t in OUTPUT_LIMIT_PHRASES) and max_tokens_this_call < OUTPUT_LIMIT_CEILING:
                max_tokens_this_call = min(max_tokens_this_call + OUTPUT_LIMIT_BUMP, OUTPUT_LIMIT_CEILING)
                print(f"[ADJUST] {model}: hit the output-limit error at max_completion_tokens="
                      f"{max_tokens_this_call - OUTPUT_LIMIT_BUMP}, retrying this call at {max_tokens_this_call}.")
                continue

            if any(t in low for t in FATAL_PHRASES):
                with broken_models_lock:
                    if key not in broken_models:
                        broken_models.add(key)
                        print("\n" + "=" * 70)
                        print(f"[STOP] {model} (reasoning_effort={reasoning_effort!r}) rejected one of our parameters outright:\n{msg}")
                        print("This is the real 'Plan B' scenario, not a fixable range error.")
                        print(f"No more calls will be sent to {model} at reasoning_effort={reasoning_effort!r}.")
                        print("Come back and we'll switch to a forced-choice / structured-output approach instead.")
                        print("=" * 70 + "\n")
                return None
            last_err = e
            time.sleep(2 ** attempt)
        except (RateLimitError, APIConnectionError, APITimeoutError, APIError) as e:
            last_err = e
            wait = 2 ** attempt
            time.sleep(wait)
    print(f"[WARN] Giving up on a call to {model} (reasoning_effort={reasoning_effort!r}) after {MAX_RETRIES} retries: {last_err}")
    return None


def extract_scores(resp, color1, color2):
    if resp is None:
        return None
    choice = resp.choices[0]
    if choice.logprobs is None or not choice.logprobs.content:
        return None

    first_pos = choice.logprobs.content[0]
    candidates = {first_pos.token: first_pos.logprob}
    for alt in (first_pos.top_logprobs or []):
        candidates.setdefault(alt.token, alt.logprob)

    variants1 = get_word_variant_strings(color1)
    variants2 = get_word_variant_strings(color2)

    def best_logprob(variants):
        matches = [lp for tok, lp in candidates.items() if token_matches_word(tok, variants)]
        return max(matches) if matches else None

    score1 = best_logprob(variants1)
    score2 = best_logprob(variants2)

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top5_tokens = [t for t, _ in ranked]
    top5_probs = [round(math.exp(lp), 4) for _, lp in ranked]

    full_answer = "".join(
        c.token for c in choice.logprobs.content
    ) if choice.logprobs.content else ""

    return {
        "score1": score1,
        "score2": score2,
        "sampled_token": first_pos.token,
        "full_answer": full_answer,
        "top5_tokens": str(top5_tokens),
        "top5_probs": str(top5_probs),
    }


def extract_usage(resp):
    """Pulls token accounting off the response, in particular
    reasoning_tokens — the number that actually matters for reasoning-model
    cost, since it's billed as output even when reasoning_effort="none" and
    nothing reasoning-shaped shows up in full_answer. None for any field the
    API didn't report, so the CSV shows a blank rather than a fabricated 0."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {"prompt_tokens": None, "completion_tokens": None, "reasoning_tokens": None, "total_tokens": None}
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details is not None else None,
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def rescale_over_temperatures(score1, score2, temps):
    rows = []
    for temp in temps:
        if score1 is None or score2 is None:
            rows.append((None, None))
            continue
        if temp > 0:
            e1 = math.exp(score1 / temp)
            e2 = math.exp(score2 / temp)
            rel1 = e1 / (e1 + e2)
        else:
            if score1 > score2:
                rel1 = 1.0
            elif score2 > score1:
                rel1 = 0.0
            else:
                rel1 = 0.5
        rows.append((round(rel1, 4), round(1 - rel1, 4)))
    return rows


# ── Job construction ─────────────────────────────────────────────────────
def build_jobs():
    jobs = []
    for model in ACTIVE_MODELS:
        for reasoning_effort in ACTIVE_REASONING_EFFORTS:
            for color1, color2 in ACTIVE_CONDITIONS:
                for price_pair, price_meta in ACTIVE_PRICE_CONDITIONS:
                    for p in build_prompts(color1, color2, price_pair):
                        if p["style"] not in ACTIVE_STYLES:
                            continue
                        jobs.append({
                            "model": model,
                            "reasoning_effort": reasoning_effort,
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
    return (job["model"], job["reasoning_effort"], job["word1"], job["word2"], job["price_pair"], job["prompt_style"])


def run_job(job):
    if (job["model"], job["reasoning_effort"]) in broken_models:
        return None
    result = query_with_retry(job["model"], job["prompt_text"], job["reasoning_effort"])
    if result is None:
        return None
    resp, n_requested = result
    scores = extract_scores(resp, job["color1"], job["color2"])
    if scores is None:
        return None
    usage = extract_usage(resp)

    score1, score2 = scores["score1"], scores["score2"]
    both_in_top_n = score1 is not None and score2 is not None
    vocab_prob1 = round(math.exp(score1), 6) if score1 is not None else None
    vocab_prob2 = round(math.exp(score2), 6) if score2 is not None else None
    target_coverage = round((vocab_prob1 or 0) + (vocab_prob2 or 0), 6) if (score1 is not None or score2 is not None) else 0.0

    (rel1, rel2), = rescale_over_temperatures(score1, score2, [BASELINE_TEMPERATURE])

    return [{
        "model": job["model"],
        # The API often resolves a model alias to a specific dated snapshot
        # (e.g. requesting "gpt-5.4" might actually be served by
        # "gpt-5.4-2026-xx-xx") and reports which one back here. Worth
        # keeping distinct from "model" (what was requested) since a
        # snapshot can change under an alias between runs made weeks apart —
        # exactly the kind of drift that matters once you're running the
        # bigger price grid across multiple sessions.
        "response_model": getattr(resp, "model", None),
        "reasoning_effort": job["reasoning_effort"],
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
        "temperature": BASELINE_TEMPERATURE,
        "score_word1": round(score1, 4) if score1 is not None else None,
        "score_word2": round(score2, 4) if score2 is not None else None,
        "rel_prob_word1": rel1,
        "rel_prob_word2": rel2,
        "vocab_prob_word1": vocab_prob1,
        "vocab_prob_word2": vocab_prob2,
        "target_coverage": target_coverage,
        "both_in_top_n": both_in_top_n,
        "top_n_requested": n_requested,
        "sampled_token": scores["sampled_token"],
        "full_answer": scores["full_answer"],
        "top5_tokens": scores["top5_tokens"],
        "top5_probs": scores["top5_probs"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "total_tokens": usage["total_tokens"],
        "prompt_text": job["prompt_text"],
    }]


# ── Main execution ───────────────────────────────────────────────────────
def main():
    jobs = build_jobs()
    print(f"Total raw API calls queued: {len(jobs)}  (1 row each, T={BASELINE_TEMPERATURE} baseline only "
          f"— see temperature_sweep.py for other temperatures)")
    print(f"Active prompt sets: {sorted(ACTIVE_PROMPT_SETS)}  |  Active styles: {sorted(ACTIVE_STYLES)}  "
          f"|  Active reasoning efforts: {ACTIVE_REASONING_EFFORTS}")
    if PILOT_MODE:
        print("[PILOT_MODE] Running a small subset. Flip PILOT_MODE = False once this looks right.")

    item_tag = ITEM_NOUN.replace(" ", "_")
    checkpoint_path = os.path.join(OUTPUT_DIR, f"logprobs_ALL_PROMPTS_OpenAI_{item_tag}_Ladder_checkpoint.csv")

    done_keys = set()
    all_rows = []
    if os.path.exists(checkpoint_path):
        # keep_default_na=False + na_values=[""]: pandas' default NA list
        # includes the literal text "None", which is exactly the sentinel
        # string build_jobs() writes for the no-price condition
        # (price_pair = "None"). Without this, every reload silently turns
        # that string into NaN, job_key() below never matches it against a
        # freshly-built job's "None" string, and the no-price condition gets
        # treated as never-done and re-queried/re-appended on every run.
        # Genuinely empty cells (e.g. a failed score extraction) still parse
        # as NaN via the explicit na_values=[""].
        prev = pd.read_csv(checkpoint_path, keep_default_na=False, na_values=[""])
        if "reasoning_effort" not in prev.columns:
            # Checkpoint predates this column — every row in it was run with
            # the old hardcoded reasoning_effort="none", so backfill that
            # rather than losing the ability to resume off it.
            prev["reasoning_effort"] = "none"
        all_rows = prev.to_dict("records")
        done_keys = set(
            zip(prev["model"], prev["reasoning_effort"], prev["word1"], prev["word2"], prev["price_pair"], prev["prompt_style"])
        )
        print(f"Resuming: {len(done_keys)} conditions already in checkpoint, skipping those.")

    pending = [j for j in jobs if job_key(j) not in done_keys]
    print(f"Conditions left to run: {len(pending)}")

    lock = threading.Lock()
    completed_since_flush = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_job, job): job for job in pending}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Querying"):
            result_rows = fut.result()
            if result_rows:
                with lock:
                    all_rows.extend(result_rows)
                    completed_since_flush += 1
                    if completed_since_flush >= 50:
                        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)
                        completed_since_flush = 0

    # Always flush once more at the end, even if this run had under 50 new
    # rows (e.g. a small pilot batch) — otherwise a run that never crosses
    # the 50-row threshold leaves no checkpoint behind, and the *next* run
    # (done_keys check above) has nothing to resume from and silently
    # redoes this run's conditions instead of skipping them.
    if all_rows:
        pd.DataFrame(all_rows).to_csv(checkpoint_path, index=False)

    df = pd.DataFrame(all_rows)

    if df.empty:
        print("\n[WARN] No rows were collected, nothing to save or summarize.")
        if broken_models:
            print(f"[NOTE] These (model, reasoning_effort) combos were skipped after a fatal param error: {sorted(broken_models)}")
        print("Check the errors printed above before rerunning.")
        return

    master_path = os.path.join(OUTPUT_DIR, f"logprobs_ALL_PROMPTS_OpenAI_{item_tag}_Ladder.csv")
    df.to_csv(master_path, index=False)
    print(f"\nDone! Master output saved to: {master_path}")

    for style, group in df.groupby("prompt_style"):
        style_path = os.path.join(OUTPUT_DIR, f"logprobs_{style}_OpenAI_{item_tag}_Ladder.csv")
        group.to_csv(style_path, index=False)
        print(f"Saved: {style_path}")

    print("\n--- Top-N coverage diagnostic (T=1.0 sampling) ---")
    single_temp = df[df["temperature"] == 1.0]
    for color in sorted(set(single_temp["color1"]) | set(single_temp["color2"])):
        as_c1 = single_temp[single_temp["color1"] == color]
        as_c2 = single_temp[single_temp["color2"] == color]
        total = len(as_c1) + len(as_c2)
        found = as_c1["score_word1"].notna().sum() + as_c2["score_word2"].notna().sum()
        if total:
            n_used = sorted(single_temp["top_n_requested"].unique())
            print(f"  {color:<12s} found in top-N (N={n_used}): {found}/{total} ({found/total:.0%})")

    if broken_models:
        print(f"\n[NOTE] These (model, reasoning_effort) combos were skipped after a fatal param error: {sorted(broken_models)}")


if __name__ == "__main__":
    main()