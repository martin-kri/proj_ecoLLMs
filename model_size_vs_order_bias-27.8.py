"""
Model-size vs. order-bias plot.
=======================================================================
Set TEMPERATURE below to run this whole analysis at any temperature, not
just the T=1.0 baseline — reuses temperature_sweep.py's rescale math (via
load_data_at_temperature()) so non-1.0 values cost no extra API calls, same
as that script. Output filenames get a _T<value> tag for anything other
than 1.0, so different temperatures' charts/CSVs don't overwrite each other.

27.8 rework, after external review of the first version raised two real
issues and one good idea:

  1. POOLING FAIRNESS — is every model's average computed over the same
     set of price/gap conditions? VERIFIED before writing this rework
     (not assumed): as of this run, Llama, gpt-5.4-nano, gpt-5.4-mini,
     gpt-5.4, and gpt-5.6-luna all carry the identical 12 price bases and
     the identical 3 fixed + 3 relative gap sizes each (confirmed via a
     direct groupby check on both CSVs, matching the n=72 every model
     already printed in the previous version). This WAS a fair concern to
     raise — it just happens to already be true here. If you ever add a
     model or price condition to only one script, re-check this before
     trusting the chart again; nothing here re-verifies it automatically.

  2. "95% CI" WAS THE WRONG LABEL — correctly called out. Each (price,
     gap) condition is one deterministic score, not a repeated random draw
     from a stable distribution, so mean +/- 1.96*SEM implied sampling
     uncertainty about a "true" bias value that doesn't exist here. What
     the error bar actually showed was real: how much order bias varies
     across different price/gap conditions for that model. Now labeled and
     computed as exactly that — the point is the MEAN across pooled
     conditions, the whisker spans the observed MIN to MAX (not a CI), and
     the summary CSV also reports the median and IQR (25th/75th
     percentile) for anyone who wants a spread measure less sensitive to
     one extreme condition than a bare min/max is.

  3. SPLIT BY GAP TYPE — fixed_gap ("+$10" style) and pct_gap ("+10%"
     style) are two different mechanics; pooling them into one number was
     its own smaller version of the pooling-fairness problem above (mixing
     two different question types under one label). Now two separate
     figures, each its own log/linear pair:
       ModelSize_vs_OrderBias_FixedGap.png
       ModelSize_vs_OrderBias_PctGap.png

WHAT "ORDER BIAS" MEANS HERE: for each model x gap type, every (gap_value,
price_base) combination's order-bias value — already averaged across all
4 rungs and both color orderings by _compute_order_bias() in
one-by-one_order-bias-26.8.py — is pooled (up to 12 bases x 3 gap sizes =
36 values per model per gap type). The "no price" and "same price"
conditions don't produce order-bias values at all (no "cheaper item" to be
mentioned first-or-second there), so they're simply absent from the pool.

MODEL SIZES ARE YOUR NUMBERS, MARKED AS ESTIMATES, NOT VERIFIED FACTS.
Only Llama-3.1-8B-Instruct's 8B is a real, published figure — see
MODEL_SIZES_B below.

NO regression trend line across models — 5 points spanning different labs'
architectures entirely isn't a real scaling relationship to fit a line
through. Points + range whiskers only. See the printed console summary for
an honest read of whatever shape the points make — that's the same one
that would need to hold up before calling it a finding.
"""

import importlib.util
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

ORDER_BIAS_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "one-by-one_order-bias-26.8.py"
)
TEMPERATURE_SWEEP_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "temperature_sweep.py"
)

# 1.0 = the real baseline as originally collected (no rescaling). Any other
# value is reconstructed the same way temperature_sweep.py does it — pure
# softmax-of-logprobs math on the existing score_word1/score_word2 columns,
# no re-querying either API. Change this and rerun to get the same pair of
# charts at a different temperature.
TEMPERATURE = 5.0

# ── Model sizes in billions of parameters — SEE THE MODULE DOCSTRING.
# Only Llama-3.1-8B-Instruct's 8B is a confirmed, published figure.
MODEL_SIZES_B = {
    "Llama-3.1-8B-Instruct": 8,
    "gpt-5.4-nano": 10,
    "gpt-5.6-luna": 10,
    "gpt-5.4-mini": 100,
    "gpt-5.4": 1800,
}

# Cosmetic-only x nudge (multiplicative, so it works sensibly on a log
# axis) for models that share an identical size estimate above — purely so
# their points don't render exactly on top of each other. Does not touch
# size_b in the saved summary CSV.
PLOT_JITTER = 0.05

GAP_TYPE_TITLES = {"fixed_gap": "Fixed $ Gap", "pct_gap": "Relative % Gap"}
GAP_TYPE_FILE_TAGS = {"fixed_gap": "FixedGap", "pct_gap": "PctGap"}


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_order_bias_module():
    return _load_module(ORDER_BIAS_SCRIPT, "order_bias_plots")


def load_data_at_temperature(ob, temperature):
    """T=1.0: read the real baseline directly, exactly like the T=1 version
    did. Any other value: reuse temperature_sweep.py's own sweep_csv() to
    reconstruct that temperature (writes/refreshes the *_temps.csv files as
    a side effect, same as running that script directly would), then hand
    the resulting in-memory dataframes to load_and_clean_data() instead of
    re-reading from disk — same df_hf/df_oa reuse pattern
    temperature_sweep.py itself uses."""
    if temperature == 1.0:
        return ob.load_and_clean_data(temperature=1.0)
    ts = _load_module(TEMPERATURE_SWEEP_SCRIPT, "temperature_sweep_mod")
    llama_df = ts.sweep_csv(ts.LLAMA_INPUT_CSV, ts.LLAMA_OUTPUT_CSV, [temperature])
    openai_df = ts.sweep_csv(ts.OPENAI_INPUT_CSV, ts.OPENAI_OUTPUT_CSV, [temperature])
    return ob.load_and_clean_data(temperature=temperature, df_hf=llama_df, df_oa=openai_df)


def verify_pooling_is_fair(ob, df_all, models):
    """Actually checks the concern raised in review, instead of asserting
    it away: does every model carry the same set of (price_base, gap_type,
    gap_value) conditions for the active prompt set(s)? Prints a clear
    [WARN] and returns False if not — the chart will still generate, but
    you'd be looking at a real pooling mismatch, not a stats-labeling one."""
    condition_sets = {}
    for model in models:
        df_model = df_all[df_all["model"] == model]
        conds = set()
        for pset in ob.ACTIVE_PROMPT_SETS:
            for ptype, gaps in ob.GAP_SIZES.items():
                for gap_value in gaps:
                    sub = df_model[
                        (df_model["price_type"] == ptype)
                        & (df_model["price_gap_or_pct"].astype(float) == float(gap_value))
                        & (df_model["prompt_set"] == pset)
                    ]
                    conds.update(sub["price_base"].dropna().astype(float).unique().tolist())
        condition_sets[model] = conds

    reference_model, reference_set = next(iter(condition_sets.items()))
    all_match = True
    for model, conds in condition_sets.items():
        if conds != reference_set:
            all_match = False
            missing = reference_set - conds
            extra = conds - reference_set
            print(f"[WARN] {model!r} does NOT have the same price-base coverage as "
                  f"{reference_model!r}. Missing: {sorted(missing)}  Extra: {sorted(extra)}")
    if all_match:
        print(f"Pooling check OK: all {len(models)} models share the identical "
              f"{len(reference_set)} price bases across every active gap condition.")
    else:
        print("[WARN] Pooling is NOT apples-to-apples across models right now — "
              "the chart below would be comparing different underlying grids, not "
              "just different models. Fix the data before trusting it.")
    return all_match


def collect_bias_values(ob, df_all, model, ptype):
    """Pools order-bias values for ONE gap type only (fixed_gap or
    pct_gap), across all active prompt sets and every gap size of that
    type — up to 12 bases x 3 gap sizes = 36 values per model."""
    df_model = df_all[df_all["model"] == model]
    values = []
    for pset in ob.ACTIVE_PROMPT_SETS:
        for gap_value in ob.GAP_SIZES[ptype]:
            bias_series = ob._compute_order_bias(df_model, ptype, gap_value, pset)
            values.extend(bias_series.dropna().tolist())
    return values


def range_stats(values):
    """Replaces the old mean+/-1.96*SEM 'CI' with what the data actually
    supports: the observed spread across pooled conditions. mean/median for
    central tendency, min/max for the full range (what the chart plots),
    q25/q75 for a spread measure less sensitive to one extreme condition."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return dict(mean=np.nan, median=np.nan, min=np.nan, max=np.nan, q25=np.nan, q75=np.nan, n=0)
    return dict(
        mean=arr.mean(), median=float(np.median(arr)),
        min=arr.min(), max=arr.max(),
        q25=float(np.percentile(arr, 25)), q75=float(np.percentile(arr, 75)),
        n=n,
    )


def build_summary(ob, df_all, ptype):
    models = ob.ordered_models(df_all)

    if ptype == list(ob.GAP_SIZES.keys())[0]:
        # Only need to run this once, not once per gap type.
        verify_pooling_is_fair(ob, df_all, [m for m in models if m in MODEL_SIZES_B])

    rows = []
    for model in models:
        if model not in MODEL_SIZES_B:
            print(f"[WARN] No size entry for {model!r} in MODEL_SIZES_B — skipping "
                  f"it entirely rather than guessing a size.")
            continue
        values = collect_bias_values(ob, df_all, model, ptype)
        stats = range_stats(values)
        rows.append({
            "model": model,
            "size_b": MODEL_SIZES_B[model],
            "gap_type": ptype,
            "mean_order_bias_pct": round(stats["mean"] * 100, 3),
            "median_order_bias_pct": round(stats["median"] * 100, 3),
            "min_pct": round(stats["min"] * 100, 3),
            "max_pct": round(stats["max"] * 100, 3),
            "q25_pct": round(stats["q25"] * 100, 3),
            "q75_pct": round(stats["q75"] * 100, 3),
            "n_conditions_pooled": stats["n"],
        })
        print(f"{model:24s} size={MODEL_SIZES_B[model]:>6,}B  "
              f"mean={stats['mean']*100:6.2f}pp  range=[{stats['min']*100:5.2f}, {stats['max']*100:5.2f}]pp  "
              f"(n={stats['n']})")

    summary = pd.DataFrame(rows)

    # Cosmetic x-jitter for models tied at the same size_b (nano/luna both
    # at 10B by default) — spreads them symmetrically around the true size
    # so both points are visible instead of stacking exactly.
    summary["plot_x"] = summary["size_b"].astype(float)
    for size_val, group in summary.groupby("size_b"):
        if len(group) > 1:
            offsets = np.linspace(-PLOT_JITTER, PLOT_JITTER, len(group))
            for (idx, _), off in zip(group.iterrows(), offsets):
                summary.loc[idx, "plot_x"] = size_val * (1 + off)

    return summary


def describe_shape(summary):
    """Prints an honest read of what the points actually show, ranked by
    size, instead of defaulting to a "bigger = more biased" narrative —
    the previous version's chart didn't support that story once you looked
    at the actual overlap between models."""
    ordered = summary.sort_values("size_b")
    print("\nShape check (by size, smallest to largest):")
    for _, row in ordered.iterrows():
        print(f"  {row['size_b']:>6,.0f}B  {row['model']:24s}  "
              f"mean={row['mean_order_bias_pct']:6.2f}pp  range=[{row['min_pct']:.2f}, {row['max_pct']:.2f}]")
    means = ordered["mean_order_bias_pct"].tolist()
    is_monotonic_increasing = all(a <= b for a, b in zip(means, means[1:]))
    is_monotonic_decreasing = all(a >= b for a, b in zip(means, means[1:]))
    if is_monotonic_increasing or is_monotonic_decreasing:
        print("  -> Means are monotonic in size for this gap type.")
    else:
        print("  -> Means are NOT monotonic in size for this gap type — do not describe "
              "this as a clean 'bigger = more biased' trend without saying so.")


def generate_figure(summary, ob, ptype, temperature, output_dir=RESULTS_DIR):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=200)

    for ax, scale, title, panel_label in zip(axes, ["log", "linear"], ["Log Scale", "Linear Scale"], ["A", "B"]):
        for _, row in summary.iterrows():
            color = ob.MODEL_COLORS.get(row["model"], "#333333")
            lower = max(row["mean_order_bias_pct"] - row["min_pct"], 0)
            upper = max(row["max_pct"] - row["mean_order_bias_pct"], 0)
            ax.errorbar(
                row["plot_x"], row["mean_order_bias_pct"], yerr=[[lower], [upper]],
                fmt="o", markersize=12, color=color, ecolor=color,
                elinewidth=2, capsize=6, markeredgecolor="white", markeredgewidth=1,
                label=row["model"],
            )

        ax.set_xscale(scale)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Model Parameters (Billions)", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.text(0.02, 0.97, panel_label, transform=ax.transAxes, fontsize=20,
                 fontweight="bold", va="top", ha="left")
        ax.grid(True, which="both" if scale == "log" else "major", alpha=0.4)

    # NOT "95% CI" — see module docstring point 2. The whisker is the
    # observed min-to-max spread across pooled price/gap conditions for
    # that model, not a sampling-uncertainty interval.
    axes[0].set_ylabel("Order Bias (Percentage Points)\nMean, whisker = range across conditions",
                        fontsize=11.5, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(labels), frameon=True, fontsize=10)

    temp_note = "baseline, T=1.0" if temperature == 1.0 else f"reconstructed at T={temperature}"
    fig.suptitle(
        f"Model Size vs. Order Bias — {GAP_TYPE_TITLES[ptype]} only (Set C, {temp_note})\n"
        "Model sizes for nano/luna/mini/gpt-5.4 are estimates, not confirmed by OpenAI. "
        "Whiskers show the range across price/gap conditions, not a confidence interval.",
        fontsize=12.5, y=1.17,
    )
    fig.tight_layout()
    temp_tag = "" if temperature == 1.0 else f"_T{temperature}".replace(".", "p")
    out_path = os.path.join(output_dir, f"ModelSize_vs_OrderBias_{GAP_TYPE_FILE_TAGS[ptype]}{temp_tag}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    ob = _load_order_bias_module()
    print(f"Temperature: {TEMPERATURE}"
          + ("" if TEMPERATURE == 1.0 else " (reconstructed via temperature_sweep.py's rescale math, no re-querying)"))
    df_all = load_data_at_temperature(ob, TEMPERATURE)

    all_summaries = []
    for ptype in ob.GAP_SIZES.keys():
        print(f"\n=== {GAP_TYPE_TITLES[ptype]} ===")
        summary = build_summary(ob, df_all, ptype)
        describe_shape(summary)
        all_summaries.append(summary)
        generate_figure(summary, ob, ptype, TEMPERATURE)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined["temperature"] = TEMPERATURE
    temp_tag = "" if TEMPERATURE == 1.0 else f"_T{TEMPERATURE}".replace(".", "p")
    summary_path = os.path.join(RESULTS_DIR, f"model_size_vs_order_bias_summary{temp_tag}.csv")
    combined.drop(columns=["plot_x"]).to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
