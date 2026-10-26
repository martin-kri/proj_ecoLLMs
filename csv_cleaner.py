"""
csv_cleaner.py
=======================================================================
Rebuilds a "clean" checkpoint/master CSV out of an older, bulkier results
file (e.g. the nano/mini run from before the 25.8 refactor), so it matches
what OpenAI-25.8.py itself produces today. Concretely:

  - temperature == 1.0 only. Drops the old 13-value-per-condition sweep;
    other temperatures are reconstructed separately by temperature_sweep.py
    if you ever need them again, not stored redundantly here.
  - price_base != 2.0. Drops the broken 10%-pct_gap-at-$2 rows (see the
    comment above PCT_GAP_BASES in OpenAI-25.8.py for why: round(2*1.10)
    collapses to (2, 2), a duplicate of same_price, not a real gap).
  - reasoning_effort backfilled to "none" — the only value that existed
    before this column did.
  - max_completion_tokens backfilled to 5 — the REQUEST-side cap, which was
    constant across every run this project has made so far. This is a known
    fact about how the call was made, not a guess.
  - prompt_tokens / completion_tokens / reasoning_tokens / total_tokens are
    left BLANK for these old rows, not invented. Those are actual usage
    numbers the API returned; we don't have them for calls made before
    usage tracking existed (the data itself shows single-word answers like
    "Green", i.e. ~1 completion token, not 5 — so backfilling 5 there would
    be a specific wrong number, not just an approximation).
  - price_pair normalized so the no-price condition round-trips as the
    literal string "None", matching what OpenAI-25.8.py's build_jobs()
    constructs. Plain pd.read_csv() treats "None" as a null marker by
    default and silently turns it into NaN, which then never matches the
    "None" string a fresh job builds — see the keep_default_na fix in
    OpenAI-25.8.py's main() for the full explanation. Get this wrong here
    and the no-price condition looks "never done" forever, defeating the
    entire point of cleaning this file for checkpoint resume.
  - EXCLUDE_MODELS dropped, defensively, even though none were present in
    the nano/mini file this was built for — the point is to guarantee no
    accidental gpt-5.4 data sneaks in from this file instead of coming
    from the (separately checkpointed) full gpt-5.4 run.

Non-destructive: reads INPUT_CSV, writes OUTPUT_*_CSV as NEW files. The
input is untouched. Once you've eyeballed the output, rename/move the two
_clean files over the active checkpoint/master filenames yourself (or point
OpenAI-25.8.py's checkpoint_path/master_path at them) — this script doesn't
overwrite the live checkpoint automatically, on purpose.

By default this also restricts to ACTIVE_PROMPT_SETS below (Set C only,
matching OpenAI-25.8.py's current scope) — set1/set2 rows in the input are
simply not carried into the output, not deleted from the input file. Set
ACTIVE_PROMPT_SETS = None to keep every prompt_set instead.
"""

import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
INPUT_CSV = "./results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv"

OUTPUT_CHECKPOINT_CSV = "./results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder_checkpoint_clean.csv"
OUTPUT_MASTER_CSV = "./results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder_clean.csv"

# Set to None to keep every prompt_set present in the input instead.
ACTIVE_PROMPT_SETS = {3}

# Rows for these models are dropped outright, regardless of any other
# filter — a defensive guard so this cleaner never becomes an accidental
# second source of truth for a model it wasn't built to hold data for.
EXCLUDE_MODELS = {"gpt-5.4"}

# The request-side cap every run in this project has used so far. NOT the
# actual number of tokens generated (that's completion_tokens, left blank
# below since we don't know it for pre-usage-tracking rows).
KNOWN_MAX_COMPLETION_TOKENS = 5


def main():
    # keep_default_na=False + na_values=[""]: see the module docstring and
    # OpenAI-25.8.py's main() for why — otherwise the no-price condition's
    # price_pair="None" gets silently read back as NaN.
    df = pd.read_csv(INPUT_CSV, keep_default_na=False, na_values=[""])
    before = len(df)

    df = df[df["temperature"].astype(float) == 1.0].copy()
    df = df[df["price_base"].astype(float) != 2.0].copy()

    if ACTIVE_PROMPT_SETS is not None:
        df = df[df["prompt_set"].isin(ACTIVE_PROMPT_SETS)].copy()

    if EXCLUDE_MODELS:
        dropped = df[df["model"].isin(EXCLUDE_MODELS)]
        if len(dropped):
            print(f"Dropping {len(dropped)} row(s) for excluded model(s) {sorted(EXCLUDE_MODELS)}.")
        df = df[~df["model"].isin(EXCLUDE_MODELS)].copy()

    if "reasoning_effort" not in df.columns:
        df["reasoning_effort"] = "none"

    df["max_completion_tokens"] = KNOWN_MAX_COMPLETION_TOKENS
    for col in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
        if col not in df.columns:
            df[col] = pd.NA

    df.to_csv(OUTPUT_CHECKPOINT_CSV, index=False)
    df.to_csv(OUTPUT_MASTER_CSV, index=False)

    print(f"Input rows:  {before}")
    print(f"Output rows: {len(df)}  (temperature==1.0, price_base!=2.0"
          + (f", prompt_set in {sorted(ACTIVE_PROMPT_SETS)}" if ACTIVE_PROMPT_SETS else "")
          + ")")
    print(f"Wrote: {OUTPUT_CHECKPOINT_CSV}")
    print(f"Wrote: {OUTPUT_MASTER_CSV}")
    print("\nReview these before promoting them — rename/move both _clean files "
          "over the active checkpoint/master filenames once you're happy with them.")


if __name__ == "__main__":
    main()
