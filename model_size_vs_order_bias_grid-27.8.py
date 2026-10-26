"""
Model-size vs. order-bias — one consolidated grid for a slide.
=======================================================================
model_size_vs_order_bias-27.8.py makes one 2-panel (log + linear) image
PER temperature PER gap type — four images total to show T=1 and T=5,
fixed and relative gaps. That doesn't fit on a presentation page once
each image's own axes/title/legend are accounted for.

This script makes ONE image instead: TEMPERATURES (rows) x GAP_TYPES
(columns), log-scale x-axis only. Dropping the linear panel is a
deliberate space trade, not a downgrade — for parameters spanning 8B to
1800B, log is the scale that actually differentiates the small end;
linear's one real piece of information (gpt-5.4 sits way out to the
right) is still legible from the log panel's own axis labels.

Reuses model_size_vs_order_bias-27.8.py's pooling and statistics
functions directly (collect_bias_values / range_stats / MODEL_SIZES_B /
PLOT_JITTER / load_data_at_temperature) rather than reimplementing them —
one source of truth for "what counts as a data point" and "what the
whiskers mean" (see that script's docstring: they're the observed
min-to-max range across pooled price/gap conditions, NOT a confidence
interval). That script itself reuses one-by-one_order-bias-26.8.py and
temperature_sweep.py, so this is a third-level reuse, not a fork.

All four panels share the same y-axis range so heights are directly
comparable across the grid, not just within a row or column.

Edit TEMPERATURES / GAP_TYPES below to change the grid shape — e.g. add
a third temperature for a 3x2 grid (still within the project's 3x3 cap).
"""

import importlib.util
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPERATURES = [1.0, 5.0]
GAP_TYPES = ["fixed_gap", "pct_gap"]


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_cell_summary(ob, msb, df_all, ptype):
    """One grid cell's worth of points: every model's mean + observed
    min/max order bias for this (temperature, gap_type) combination."""
    models = ob.ordered_models(df_all)
    rows = []
    for model in models:
        if model not in msb.MODEL_SIZES_B:
            continue
        values = msb.collect_bias_values(ob, df_all, model, ptype)
        stats = msb.range_stats(values)
        rows.append({
            "model": model,
            "size_b": msb.MODEL_SIZES_B[model],
            "mean_pct": stats["mean"] * 100,
            "min_pct": stats["min"] * 100,
            "max_pct": stats["max"] * 100,
            "n": stats["n"],
        })
    summary = pd.DataFrame(rows)

    # Same cosmetic jitter as model_size_vs_order_bias-27.8.py, for models
    # tied at the same size estimate (nano/luna both 10B by default).
    summary["plot_x"] = summary["size_b"].astype(float)
    for size_val, group in summary.groupby("size_b"):
        if len(group) > 1:
            n = len(group)
            offsets = [-msb.PLOT_JITTER + 2 * msb.PLOT_JITTER * i / (n - 1) for i in range(n)]
            for (idx, _), off in zip(group.iterrows(), offsets):
                summary.loc[idx, "plot_x"] = size_val * (1 + off)
    return summary


def main():
    ob = _load_module(os.path.join(HERE, "one-by-one_order-bias-26.8.py"), "ob_mod")
    msb = _load_module(os.path.join(HERE, "model_size_vs_order_bias-27.8.py"), "msb_mod")

    # Each temperature's cleaned data loaded once, reused across both gap
    # types in that row (T!=1.0 reconstructed via temperature_sweep.py's
    # rescale math — no extra API calls, same as the single-temperature script).
    temp_dataframes = {}
    for temp in TEMPERATURES:
        print(f"Loading T={temp}" + ("" if temp == 1.0 else " (reconstructed)..."))
        temp_dataframes[temp] = msb.load_data_at_temperature(ob, temp)

    msb.verify_pooling_is_fair(
        ob, temp_dataframes[TEMPERATURES[0]],
        [m for m in ob.ordered_models(temp_dataframes[TEMPERATURES[0]]) if m in msb.MODEL_SIZES_B],
    )

    cell_data = {}
    y_max = 0.0
    all_summaries = []
    for temp in TEMPERATURES:
        for ptype in GAP_TYPES:
            summary = build_cell_summary(ob, msb, temp_dataframes[temp], ptype)
            summary["temperature"] = temp
            summary["gap_type"] = ptype
            cell_data[(temp, ptype)] = summary
            all_summaries.append(summary)
            if len(summary):
                y_max = max(y_max, summary["max_pct"].max())
    y_max *= 1.08

    sns.set_theme(style="whitegrid")
    n_rows, n_cols = len(TEMPERATURES), len(GAP_TYPES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.3 * n_rows), dpi=200, squeeze=False)

    for row_idx, temp in enumerate(TEMPERATURES):
        for col_idx, ptype in enumerate(GAP_TYPES):
            ax = axes[row_idx][col_idx]
            summary = cell_data[(temp, ptype)]
            for _, r in summary.iterrows():
                color = ob.MODEL_COLORS.get(r["model"], "#333333")
                lower = max(r["mean_pct"] - r["min_pct"], 0)
                upper = max(r["max_pct"] - r["mean_pct"], 0)
                ax.errorbar(
                    r["plot_x"], r["mean_pct"], yerr=[[lower], [upper]],
                    fmt="o", markersize=9, color=color, ecolor=color,
                    elinewidth=1.8, capsize=4, markeredgecolor="white", markeredgewidth=1,
                    label=r["model"] if (row_idx == 0 and col_idx == 0) else "",
                )
            ax.set_xscale("log")
            ax.set_ylim(0, y_max)
            temp_label = "T=1.0 (baseline)" if temp == 1.0 else f"T={temp:g} (reconstructed)"
            ax.set_title(f"{temp_label} — {msb.GAP_TYPE_TITLES[ptype]}", fontsize=11.5, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel("Order Bias (pp)", fontsize=10.5, fontweight="bold")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Model Parameters (Billions)", fontsize=10.5, fontweight="bold")
            ax.grid(True, which="both", alpha=0.35)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=len(labels), frameon=True, fontsize=9.5)

    fig.suptitle(
        "Model Size vs. Order Bias — Temperature x Gap Type (Set C, log scale)\n"
        "Whiskers = observed min-to-max range across price/gap conditions, not a confidence interval. "
        "nano/luna/mini/gpt-5.4 sizes are estimates.",
        fontsize=11, y=1.115,
    )
    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "ModelSize_vs_OrderBias_Grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    combined = pd.concat(all_summaries, ignore_index=True)
    csv_path = os.path.join(RESULTS_DIR, "model_size_vs_order_bias_grid_summary.csv")
    combined.drop(columns=["plot_x"]).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
