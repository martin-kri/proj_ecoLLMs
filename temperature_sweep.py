"""
Temperature-sweep post-processor.
=======================================================================
The ladder-experiment scripts (OpenAI-25.8.py, LLAMA-25.8.py) save only
the T=1.0 baseline row per condition — one real API call / forward pass,
no repeated sampling. This script reconstructs relative choice probabilities
at any OTHER temperature(s) you want, purely as post-processing math on the
score_word1/score_word2 columns already sitting in those baseline CSVs.

No API calls, no model loading. Same rescale formula the ladder scripts
used to apply inline (softmax of score/temperature over the 2 candidate
words), just moved here and run once per extra temperature you ask for.

26.8: sweeps BOTH the Llama and OpenAI baseline CSVs together (not just
one), and for each new temperature also regenerates the two combined
order-bias overview figures (Slide1_Overview_AllGaps_Set3 /
Slide_Overview_GroupedBar_AllGaps_Set3) tagged with that temperature, by
calling the plotting functions straight out of
one-by-one_order-bias-26.8.py (dynamically imported — its filename has
hyphens, so a normal `import` won't work) rather than duplicating that
plotting logic here. Both scripts share the exact same, already-tested
merge/clean/filter code via load_and_clean_data()'s new df_hf/df_oa
parameters, so there's one source of truth for "what counts as a valid
row", not two copies that can quietly drift apart.

WHY THIS IS WORTH LOOKING AT (T=1 is not necessarily representative):
order bias compares two DIFFERENT API calls (cheap-item-first vs
cheap-item-second), each rescaled as sigmoid((score_cheap - score_other) /
T). Sigmoid is nonlinear, and those two calls generally have different raw
score gaps, so the same T compresses each one by a different amount — the
GAP between them (the order-bias number itself) generally reshapes as T
changes, not just scales uniformly. As T -> infinity every rel_prob
converges to 0.5, so order bias converges to 0 regardless of any real
effect underneath. As T -> 0, rel_probs saturate toward hard 0/1 (this
script's T<=0 branch does that directly via argmax), which typically makes
order bias MORE extreme, not less. T=1 is one point on that curve.

Usage: edit TEMPERATURES_TO_ADD below, then run. GENERATE_OVERVIEW_PLOTS
controls whether the plots happen automatically or you just get the CSVs
to look at / feed into the full plotting scripts yourself later.
"""

import importlib.util
import math
import os
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
LLAMA_INPUT_CSV = "./results/logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct.csv"
LLAMA_OUTPUT_CSV = "./results/logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct_temps.csv"

OPENAI_INPUT_CSV = "./results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv"
OPENAI_OUTPUT_CSV = "./results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder_temps.csv"

# List any additional temperatures you want reconstructed.
TEMPERATURES_TO_ADD = [4.0, 5.0]

# If True, for every temperature in TEMPERATURES_TO_ADD (besides 1.0) this
# also writes Slide1_Overview_AllGaps_Set3_T<temp>.png and
# Slide_Overview_GroupedBar_AllGaps_Set3_T<temp>.png — same two figures
# one-by-one_order-bias-26.8.py produces at the T=1 baseline, just at this
# temperature instead. Set False to only get the CSVs (you can always run
# the full plotting scripts against *_temps.csv by hand afterward, at any
# temperature, by pointing HF_FILE/OPENAI_FILE at them and adjusting
# load_and_clean_data(temperature=...)).
GENERATE_OVERVIEW_PLOTS = True

RESULTS_DIR = "./results"

ORDER_BIAS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "one-by-one_order-bias-26.8.py"
)


def rescale_over_temperature(score1, score2, temp):
    """Same formula the ladder scripts used inline before the sweep was split
    out: softmax of the two raw logprobs at the given temperature. temp<=0 is
    treated as argmax (hard choice), matching the original inline behavior."""
    if score1 is None or score2 is None or pd.isna(score1) or pd.isna(score2):
        return None, None
    if temp > 0:
        e1 = math.exp(score1 / temp)
        e2 = math.exp(score2 / temp)
        rel1 = e1 / (e1 + e2)
    elif score1 > score2:
        rel1 = 1.0
    elif score2 > score1:
        rel1 = 0.0
    else:
        rel1 = 0.5
    return round(rel1, 4), round(1 - rel1, 4)


def sweep_csv(input_csv, output_csv, extra_temps):
    """Reads a baseline (T=1.0-only) CSV, writes baseline + reconstructed
    rows for every temperature in extra_temps to output_csv, and returns the
    combined dataframe so the caller can reuse it in memory (e.g. for
    plotting) instead of re-reading it from disk."""
    # keep_default_na=False + na_values=[""]: the no-price condition's
    # price_pair="None" is a literal sentinel string, not a null — see
    # OpenAI-25.8.py's main() for the full explanation of why plain
    # pd.read_csv() silently breaks this.
    df_base = pd.read_csv(input_csv, keep_default_na=False, na_values=[""])

    other_temps = sorted(set(df_base["temperature"].astype(float).unique()) - {1.0})
    if other_temps:
        print(f"[WARN] {input_csv} already has rows at temperatures other than "
              f"1.0 ({other_temps}); carried through untouched, but this script "
              f"assumes a single-baseline file as produced by the current "
              f"ladder scripts — double check this is the file you meant.")

    frames = [df_base]
    for temp in extra_temps:
        df_t = df_base.copy()
        rescaled = df_t.apply(
            lambda r: rescale_over_temperature(r["score_word1"], r["score_word2"], temp),
            axis=1,
        )
        df_t["temperature"] = temp
        df_t["rel_prob_word1"] = [r[0] for r in rescaled]
        df_t["rel_prob_word2"] = [r[1] for r in rescaled]
        frames.append(df_t)

    df_out = pd.concat(frames, ignore_index=True)
    df_out.to_csv(output_csv, index=False)
    print(f"Wrote {len(df_out)} rows ({len(df_base)} baseline + "
          f"{len(df_base) * len(extra_temps)} reconstructed) to {output_csv}")
    return df_out


def _load_order_bias_module():
    spec = importlib.util.spec_from_file_location("order_bias_plots", ORDER_BIAS_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    extra_temps = sorted(t for t in set(TEMPERATURES_TO_ADD) if t != 1.0)

    llama_df = sweep_csv(LLAMA_INPUT_CSV, LLAMA_OUTPUT_CSV, extra_temps)
    openai_df = sweep_csv(OPENAI_INPUT_CSV, OPENAI_OUTPUT_CSV, extra_temps)

    if not GENERATE_OVERVIEW_PLOTS or not extra_temps:
        return

    print("\nGenerating overview plots per temperature (reusing "
          "one-by-one_order-bias-26.8.py's plotting functions)...")
    ob = _load_order_bias_module()

    for temp in extra_temps:
        df_all = ob.load_and_clean_data(temperature=temp, df_hf=llama_df, df_oa=openai_df)
        tag = f"_T{temp}".replace(".", "p")
        title_suffix = f" — T={temp}"
        ob.generate_combined_overview_line(df_all, output_dir=RESULTS_DIR, filename_suffix=tag, title_suffix=title_suffix)
        ob.generate_combined_grouped_bar(df_all, output_dir=RESULTS_DIR, filename_suffix=tag, title_suffix=title_suffix)


if __name__ == "__main__":
    main()
