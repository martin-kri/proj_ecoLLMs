"""
Order-bias plotting script.
=======================================================================
26.8 rework: the price grid it plots grew from 1 gap size per type (fixed
$10, pct 10%) to 3 each (fixed $10/$20/$50, pct 10%/20%/50%), across 7
price bases instead of 5. That turns "gap size" from a title string into a
real axis, so this is a structural rework, not just updated constants.

Two output styles per the "both, please" answer:
  - COMBINED: one figure per active prompt set showing ALL 3 gap sizes AND
    both types (fixed $ and relative %) together, laid out as a grid —
    rows = gap-size tier (Small/Medium/Large), cols = type (Fixed, Pct).
    That's a 3x2 grid per set, capped well under the 3x3 max. If you ever
    widen ACTIVE_PROMPT_SETS beyond one set, this still produces one
    separate combined image per set rather than trying to cram sets into
    the same grid too (that would push past 3x3).
  - SEPARATE: the original one-condition-at-a-time chart, unchanged in
    spirit, just now looped over all 6 (type, gap-size) combinations
    instead of the old single implied gap per type.

Also removed a stale compatibility patch (hf_100_fixed) that used to
synthetically relabel Llama's pct_gap rows at base=100 as fixed_gap rows,
hardcoded to gap=10.0 — correct back when pct_gap only ever had one value
(0.10, which happens to equal +$10 at base 100), but wrong now that
pct_gap has 3 values: it would've mislabeled the 0.20/0.50 rows at base 100
as "+$10 fixed gap" too. A real fixed_gap base=100/gap=10 condition already
exists natively in the current data (FIXED_GAPS includes 10), so the patch
is both wrong and redundant now — removed rather than fixed.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── File Paths & Configuration ───────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

HF_FILE = os.path.join(
    RESULTS_DIR, "logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct.csv"
)
OPENAI_FILE = os.path.join(
    RESULTS_DIR, "logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv"
)

# 26.8: extended again to match the current FIXED_GAP_BASES / PCT_GAP_BASES
# in LLAMA-25.8.py / OpenAI-25.8.py (both now carry the same 12 bases, added
# 700/750/850/900/1200 to the earlier 7-base grid).
ALL_PRICE_BASES = [10, 40, 100, 180, 500, 700, 750, 800, 850, 900, 1000, 1200]

# Matches FIXED_GAPS / PCT_GAPS in the experiment scripts. Redeclared here
# rather than imported since this is a standalone plotting script reading
# CSVs, not importing the experiment scripts directly.
GAP_SIZES = {
    "fixed_gap": [10, 20, 50],
    "pct_gap": [0.10, 0.20, 0.50],
}

# Pairs each fixed gap with the pct gap at the same list position, for the
# combined "both types together" view: (tier label, fixed gap, pct gap).
GAP_TIERS = list(zip(["Small", "Medium", "Large"], GAP_SIZES["fixed_gap"], GAP_SIZES["pct_gap"]))

# Which complexity ladder set(s) to include. 1=Set A, 2=Set B, 3=Set C.
# Currently just Set C, matching OpenAI-25.8.py/LLAMA-25.8.py's current
# scope, so the 4-model comparison stays apples-to-apples. Every combined
# figure is generated once PER active set (not with sets as a grid axis),
# so this can grow to {1,2,3} later without breaking the 3x3 cap on any
# single image — see the module docstring.
ACTIVE_PROMPT_SETS = [3]

# Updated to Set A, Set B, Set C
SET_LABELS = {
    1: "Set A: Constraints",
    2: "Set B: Context",
    3: "Set C: Pure Length + 1 Constraint",
}

MODEL_COLORS = {
    "Llama-3.1-8B-Instruct": "#2ca02c",
    "gpt-5.4-mini": "#1f77b4",
    "gpt-5.4-nano": "#d62728",
    # Explicitly assigned rather than left to DEFAULT_COLOR_CYCLE fallback:
    # with 4 models sorted alphabetically (Llama, gpt-5.4, gpt-5.4-mini,
    # gpt-5.4-nano), gpt-5.4 would land on DEFAULT_COLOR_CYCLE[1], which is
    # "#1f77b4" — the same blue already hardcoded above for gpt-5.4-mini.
    "gpt-5.4": "#9467bd",
    # Same reasoning applied preemptively here: with 5 models sorted
    # alphabetically this one happens to land on DEFAULT_COLOR_CYCLE[4]
    # ("#8c564b") anyway with no collision today, but pinning it explicitly
    # means a future 6th model (or any model rename) can't silently break
    # that coincidence the way gpt-5.4 did above.
    "gpt-5.6-luna": "#8c564b",
}
DEFAULT_COLOR_CYCLE = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]

# Fixed legend/bar-group order, per request — NOT alphabetical (that's what
# `sorted()` was doing everywhere before this). Any model not in this list
# (e.g. a future 6th model) is appended alphabetically at the end rather
# than silently vanishing from the chart.
MODEL_ORDER = ["Llama-3.1-8B-Instruct", "gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4"]


def ordered_models(df):
  present = set(df["model"].unique())
  known = [m for m in MODEL_ORDER if m in present]
  unknown = sorted(present - set(MODEL_ORDER))
  return known + unknown


# ── Gap-condition label helpers (replace the old static GAP_CONFIG dicts,
# which assumed exactly one gap size per type) ──────────────────────────────
def gap_title_label(ptype, gap_value):
    if ptype == "fixed_gap":
        return f"Fixed \\${gap_value} Gap"
    return f"{int(round(gap_value * 100))}% Relative Gap"


def gap_file_suffix(ptype, gap_value):
    if ptype == "fixed_gap":
        return f"FixedGap{gap_value}"
    return f"PctGap{int(round(gap_value * 100))}"


def price_label(ptype, base, gap_value, short=False, compact_fixed=False):
    """e.g. '$100 (+$10)' for fixed_gap, '$100 (+$10)' for pct_gap too (the
    dollar diff is what's shown either way — the pct itself is in the title).
    compact_fixed=True drops the "(+$X)" suffix for fixed_gap specifically
    (it's the same constant on every tick, already stated in the title —
    repeating it 7 times just eats horizontal space); pct_gap keeps it since
    the dollar diff genuinely varies per base."""
    if ptype == "fixed_gap":
        if compact_fixed:
            return f"\\${base}"
        diff = gap_value
    else:
        diff = round(base * (1 + gap_value)) - base
    sep = "\n" if short else " "
    return f"\\${base}{sep}(+\\${diff})"


# ── Data Loading & Preprocessing ──────────────────────────────────────────────
def get_cheaper_order(row):
  pp = str(row["price_pair"])
  if pp in ["nan", "None"]:
    return "no_price"
  try:
    p1, p2 = [float(x) for x in pp.strip("() ").split(",")]
    if p1 < p2:
      return "cheaper_first"
    elif p1 > p2:
      return "cheaper_second"
    return "equal_price"
  except Exception:
    return "unknown"


def get_p_cheaper(row):
  pp = str(row["price_pair"])
  if pp in ["nan", "None"]:
    return np.nan
  try:
    p1, p2 = [float(x) for x in pp.strip("() ").split(",")]
    if p1 < p2:
      return row["rel_prob_word1"]
    elif p2 < p1:
      return row["rel_prob_word2"]
    return np.nan
  except Exception:
    return np.nan


def _parse_prices(pp):
  pp = str(pp)
  if pp in ("nan", "None"):
    return (None, None)
  try:
    p1, p2 = [float(x) for x in pp.strip("() ").split(",")]
    return (p1, p2)
  except Exception:
    return (None, None)


def load_and_clean_data(temperature=1.0, df_hf=None, df_oa=None):
  """df_hf / df_oa let a caller (temperature_sweep.py) hand in already-
  loaded, already-temperature-swept dataframes instead of this function
  re-reading the T=1.0-only files from disk — same cleaning logic either
  way, just skipping the redundant read when the data's already in memory."""
  dfs = []
  if df_hf is not None:
    dfs.append(df_hf.copy())
  elif os.path.exists(HF_FILE):
    df_hf = pd.read_csv(HF_FILE, keep_default_na=False, na_values=[""])
    if "model" not in df_hf.columns:
      df_hf["model"] = "Llama-3.1-8B-Instruct"
    dfs.append(df_hf)

  if df_oa is not None:
    dfs.append(df_oa.copy())
  elif os.path.exists(OPENAI_FILE):
    df_oa = pd.read_csv(OPENAI_FILE, keep_default_na=False, na_values=[""])
    dfs.append(df_oa)

  df_all = pd.concat(dfs, ignore_index=True)
  df_all = df_all[df_all["temperature"].astype(float) == temperature].copy()

  # Defensive: no $2 base exists in the current price grid (see the
  # comment above PCT_GAP_BASES in the experiment scripts for why it was
  # dropped), but keep this filter as a no-op safety net in case an older
  # CSV still has it mixed in.
  df_all = df_all[df_all["price_base"].astype(float) != 2.0].copy()

  parsed = df_all["price_pair"].apply(_parse_prices)
  df_all["_p1"] = parsed.apply(lambda t: t[0])
  df_all["_p2"] = parsed.apply(lambda t: t[1])
  degenerate = df_all["price_type"].isin(["fixed_gap", "pct_gap"]) & (
      df_all["_p1"] == df_all["_p2"]
  )
  if degenerate.any():
    df_all = df_all[~degenerate]
  df_all = df_all.drop(columns=["_p1", "_p2"])

  df_all["cheaper_pos"] = df_all.apply(get_cheaper_order, axis=1)
  df_all["p_cheaper"] = df_all.apply(get_p_cheaper, axis=1)
  return df_all


def _compute_order_bias(df_model, ptype, gap_value, pset):
  sub = df_model[
      (df_model["price_type"] == ptype)
      & (df_model["price_gap_or_pct"].astype(float) == float(gap_value))
      & (df_model["prompt_set"] == pset)
  ]
  first = (
      sub[sub["cheaper_pos"] == "cheaper_first"]
      .groupby("price_base")["p_cheaper"]
      .mean()
  )
  second = (
      sub[sub["cheaper_pos"] == "cheaper_second"]
      .groupby("price_base")["p_cheaper"]
      .mean()
  )
  return (first - second).abs().dropna()


def _plot_bias_lines(ax, df_all, ptype, gap_value, pset, models, show_ylabel, title):
  # Evenly-spaced categorical x positions, NOT a log-scale numeric axis.
  # With 12 bases now including a tight cluster (700/750/800/850/900, a
  # 1.29x span), a real log axis crams 5 labels into ~4.5% of its width —
  # confirmed by rendering it: even single-character vertical labels
  # collided into an unreadable mess there. The bar charts already solved
  # this the same way (arange positions, ignoring actual dollar spacing),
  # confirmed clean at 12 bases — same fix applied here for consistency.
  x_indices = np.arange(len(ALL_PRICE_BASES))
  for i, model in enumerate(models):
    bias = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
    vals = [bias.get(b, np.nan) for b in ALL_PRICE_BASES]
    color = MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)])
    ax.plot(x_indices, vals, marker="o", linewidth=2.5, markersize=8, color=color, label=model)

  ax.set_xlim(-0.5, len(ALL_PRICE_BASES) - 0.5)
  ax.set_xticks(x_indices)
  # compact_fixed drops the redundant "(+$X)" for fixed_gap (constant across
  # every tick, already in the title) to shorten the text further.
  # rotation=30 (not -30): matplotlib's sign convention is the opposite of
  # what "tilt your head right to read" suggests -- +30 with ha="right" is
  # what actually produces down-to-the-right sloping text.
  ax.set_xticklabels(
      [price_label(ptype, b, gap_value, compact_fixed=True) for b in ALL_PRICE_BASES],
      fontsize=9, fontweight="medium", rotation=30, ha="right",
  )
  ax.set_ylim(0, 1.0)
  ax.set_yticks(np.arange(0.0, 1.1, 0.2))
  ax.tick_params(axis="y", labelsize=10)
  if show_ylabel:
    ax.set_ylabel("Order Bias\n(|P(1st)-P(2nd)|)", fontsize=10, fontweight="bold")
  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)


# ── 1a. COMBINED overview: all 3 gap sizes x both types, one grid per set ────
def generate_combined_overview_line(df_all, output_dir=RESULTS_DIR, filename_suffix="", title_suffix=""):
  sns.set_theme(style="whitegrid")
  models = ordered_models(df_all)

  for pset in ACTIVE_PROMPT_SETS:
    n_rows, n_cols = len(GAP_TIERS), 2  # rows=tier (<=3), cols=type (2) -> within 3x3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.6 * n_rows), dpi=200, sharey=True, squeeze=False)

    for row_idx, (tier_name, fixed_gap, pct_gap) in enumerate(GAP_TIERS):
      for col_idx, (ptype, gap_value) in enumerate([("fixed_gap", fixed_gap), ("pct_gap", pct_gap)]):
        ax = axes[row_idx][col_idx]
        title = f"{tier_name} — {gap_title_label(ptype, gap_value)}"
        _plot_bias_lines(ax, df_all, ptype, gap_value, pset, models, show_ylabel=(col_idx == 0), title=title)
        if row_idx == 0 and col_idx == 0:
          ax.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=0.95)

    fig.suptitle(
        f"Order Bias — All Gap Sizes ({SET_LABELS.get(pset, f'Set {pset}')}){title_suffix}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"Slide1_Overview_AllGaps_Set{pset}{filename_suffix}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 1b. SEPARATE overview: one figure per (type, gap size), sets as columns ──
def generate_separate_overview_line(df_all, ptype, gap_value, output_dir=RESULTS_DIR):
  sns.set_theme(style="whitegrid")
  n_sets = len(ACTIVE_PROMPT_SETS)
  fig, axes = plt.subplots(1, n_sets, figsize=(6.3 * n_sets, 6.2), dpi=200, sharey=True, squeeze=False)
  axes = axes[0]
  models = ordered_models(df_all)

  for ax_idx, pset in enumerate(ACTIVE_PROMPT_SETS):
    ax = axes[ax_idx]
    _plot_bias_lines(
        ax, df_all, ptype, gap_value, pset, models,
        show_ylabel=(ax_idx == 0), title=SET_LABELS.get(pset, f"Set {pset}"),
    )
    ax.legend(fontsize=11, loc="upper left", frameon=True, framealpha=0.95)

  fig.suptitle(
      f"Overview: Order Bias across Base Prices ({gap_title_label(ptype, gap_value)})",
      fontsize=16, fontweight="bold", y=1.03,
  )
  fig.tight_layout()
  out_path = os.path.join(output_dir, f"Slide1_Models_Overview_{gap_file_suffix(ptype, gap_value)}.png")
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out_path}")


# ── 2a. COMBINED grouped bar: all 3 gap sizes x both types, one grid per set ─
def generate_combined_grouped_bar(df_all, output_dir=RESULTS_DIR, filename_suffix="", title_suffix=""):
  sns.set_theme(style="whitegrid")
  models = ordered_models(df_all)
  n_models = len(models)
  bar_width = 0.8 / n_models
  x_indices = np.arange(len(ALL_PRICE_BASES))

  for pset in ACTIVE_PROMPT_SETS:
    n_rows, n_cols = len(GAP_TIERS), 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.8 * n_rows), dpi=200, sharey=True, squeeze=False)

    for row_idx, (tier_name, fixed_gap, pct_gap) in enumerate(GAP_TIERS):
      for col_idx, (ptype, gap_value) in enumerate([("fixed_gap", fixed_gap), ("pct_gap", pct_gap)]):
        ax = axes[row_idx][col_idx]
        for i, model in enumerate(models):
          bias_series = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
          vals = [bias_series.get(b, np.nan) for b in ALL_PRICE_BASES]
          offsets = (i - (n_models - 1) / 2) * bar_width
          pos = x_indices + offsets
          color = MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)])
          ax.bar(pos, vals, width=bar_width, color=color, label=model if (row_idx == 0 and col_idx == 0) else "")

        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xticks(x_indices)
        # Single-line + rotated instead of the old 2-line-stacked short=True:
        # with pct_gap's dollar-diff parenthetical getting wider (up to
        # "(+$600)"), stacked 2-line labels crowded into each other
        # vertically at this font size. rotation=30 (not -30) + ha="right":
        # matplotlib's sign convention is the opposite of "tilt your head
        # right to read" — +30 is what actually slopes down-to-the-right.
        ax.set_xticklabels(
            [price_label(ptype, b, gap_value, short=False) for b in ALL_PRICE_BASES],
            fontsize=8, fontweight="medium", rotation=30, ha="right",
        )
        if col_idx == 0:
          ax.set_ylabel("Order Bias", fontsize=9.5, fontweight="bold")
        ax.set_title(f"{tier_name} — {gap_title_label(ptype, gap_value)}", fontsize=10.5, fontweight="bold", pad=8)
        if row_idx == 0 and col_idx == 0:
          ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.95)

    fig.suptitle(
        f"Order Bias Across All Base Prices — All Gap Sizes ({SET_LABELS.get(pset, f'Set {pset}')}){title_suffix}",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"Slide_Overview_GroupedBar_AllGaps_Set{pset}{filename_suffix}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 2b. SEPARATE grouped bar: one figure per (type, gap size) ────────────────
def generate_separate_grouped_bar(df_all, ptype, gap_value, output_dir=RESULTS_DIR):
  sns.set_theme(style="whitegrid")
  n_sets = len(ACTIVE_PROMPT_SETS)
  fig, axes = plt.subplots(1, n_sets, figsize=(6.3 * n_sets, 6.2), dpi=200, sharey=True, squeeze=False)
  axes = axes[0]
  models = ordered_models(df_all)
  n_models = len(models)
  bar_width = 0.8 / n_models
  x_indices = np.arange(len(ALL_PRICE_BASES))

  for ax_idx, pset in enumerate(ACTIVE_PROMPT_SETS):
    ax = axes[ax_idx]
    for i, model in enumerate(models):
      bias_series = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
      vals = [bias_series.get(b, np.nan) for b in ALL_PRICE_BASES]
      offsets = (i - (n_models - 1) / 2) * bar_width
      pos = x_indices + offsets
      color = MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)])
      bars = ax.bar(pos, vals, width=bar_width, color=color, label=model if ax_idx == 0 else "")
      for bar, v in zip(bars, vals):
        if not np.isnan(v):
          ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(axis="y", labelsize=11)
    ax.set_xticks(x_indices)
    # Same fix as the combined grouped-bar chart above: single-line +
    # rotated instead of 2-line-stacked, since pct_gap's widening dollar
    # diff (up to "(+$600)") crowded stacked labels into each other. See
    # that comment for the rotation-sign note (+30, not -30).
    ax.set_xticklabels(
        [price_label(ptype, b, gap_value, short=False) for b in ALL_PRICE_BASES],
        fontsize=9.5, fontweight="medium", rotation=30, ha="right",
    )
    if ax_idx == 0:
      ax.set_ylabel("Order Bias", fontsize=12, fontweight="bold")
      ax.legend(fontsize=10, loc="upper left", frameon=True, framealpha=0.95)
    ax.set_title(SET_LABELS.get(pset, f"Set {pset}"), fontsize=13, fontweight="bold", pad=10)

  fig.suptitle(
      f"Overview: Order Bias Across All Base Prices ({gap_title_label(ptype, gap_value)})",
      fontsize=16, fontweight="bold", y=1.03,
  )
  fig.tight_layout()
  out_path = os.path.join(output_dir, f"Slide_Overview_GroupedBar_{gap_file_suffix(ptype, gap_value)}.png")
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out_path}")


# ── 3a. COMBINED average bar: all 3 gap sizes x both types, one grid per set ─
def generate_combined_average_bar(df_all, output_dir=RESULTS_DIR):
  sns.set_theme(style="whitegrid")
  models = ordered_models(df_all)

  for pset in ACTIVE_PROMPT_SETS:
    n_rows, n_cols = len(GAP_TIERS), 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 3.4 * n_rows), dpi=200, sharey=True, squeeze=False)

    for row_idx, (tier_name, fixed_gap, pct_gap) in enumerate(GAP_TIERS):
      for col_idx, (ptype, gap_value) in enumerate([("fixed_gap", fixed_gap), ("pct_gap", pct_gap)]):
        ax = axes[row_idx][col_idx]
        values, colors, labels = [], [], []
        for i, model in enumerate(models):
          bias = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
          subset = bias[bias.index.isin(ALL_PRICE_BASES)]
          values.append(subset.mean() if len(subset) else np.nan)
          colors.append(MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)]))
          labels.append(model)

        bars = ax.bar(labels, values, color=colors, width=0.6)
        for bar, v in zip(bars, values):
          if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")

        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.1, 0.2))
        ax.tick_params(axis="y", labelsize=9)
        if col_idx == 0:
          ax.set_ylabel("Avg Order Bias", fontsize=9.5, fontweight="bold")
        ax.set_title(f"{tier_name} — {gap_title_label(ptype, gap_value)}", fontsize=10.5, fontweight="bold", pad=8)
        ax.tick_params(axis="x", rotation=20, labelsize=8.5)

    fig.suptitle(
        f"Overall Average Order Bias — All Gap Sizes ({SET_LABELS.get(pset, f'Set {pset}')})",
        fontsize=15, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out_path = os.path.join(output_dir, f"Slide_Overview_AverageBar_AllGaps_Set{pset}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── 3b. SEPARATE average bar: one figure per (type, gap size) ────────────────
def generate_separate_average_bar(df_all, ptype, gap_value, output_dir=RESULTS_DIR):
  sns.set_theme(style="whitegrid")
  n_sets = len(ACTIVE_PROMPT_SETS)
  fig, axes = plt.subplots(1, n_sets, figsize=(5.3 * n_sets, 6.0), dpi=200, sharey=True, squeeze=False)
  axes = axes[0]
  models = ordered_models(df_all)

  for ax_idx, pset in enumerate(ACTIVE_PROMPT_SETS):
    ax = axes[ax_idx]
    values, colors, labels = [], [], []
    for i, model in enumerate(models):
      bias = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
      subset = bias[bias.index.isin(ALL_PRICE_BASES)]
      values.append(subset.mean() if len(subset) else np.nan)
      colors.append(MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)]))
      labels.append(model)

    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
      if not np.isnan(v):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=14, fontweight="bold")

    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.1, 0.2))
    ax.tick_params(axis="y", labelsize=13)
    if ax_idx == 0:
      ax.set_ylabel(f"Order Bias (Average \\${ALL_PRICE_BASES[0]}\u2013\\${ALL_PRICE_BASES[-1]})", fontsize=12.5, fontweight="bold")
    ax.set_title(SET_LABELS.get(pset, f"Set {pset}"), fontsize=15, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=15, labelsize=12)

  fig.suptitle(
      f"Overview: Overall Average Order Bias by Model ({gap_title_label(ptype, gap_value)})",
      fontsize=17, fontweight="bold", y=1.03,
  )
  fig.tight_layout()
  out_path = os.path.join(output_dir, f"Slide_Overview_AverageBar_{gap_file_suffix(ptype, gap_value)}.png")
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out_path}")


# ── 4. Detailed grids: one condition at a time, prices paginated in groups
# of <=3 to respect the 3x3 cap (7 bases no longer fit in one row) ──────────
def generate_price_grid_slide(df_all, ptype, gap_value, prices, grid_title_prefix, out_suffix, output_dir=RESULTS_DIR):
  sns.set_theme(style="whitegrid")
  n_rows = len(prices)
  n_cols = len(ACTIVE_PROMPT_SETS)

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.3 * n_cols, 4.3 * n_rows), dpi=200, sharey=True, squeeze=False)
  models = ordered_models(df_all)

  for row_idx, base_price in enumerate(prices):
    p_label = price_label(ptype, base_price, gap_value)

    for col_idx, pset in enumerate(ACTIVE_PROMPT_SETS):
      ax = axes[row_idx][col_idx]
      values, colors, labels = [], [], []

      for i, model in enumerate(models):
        bias = _compute_order_bias(df_all[df_all["model"] == model], ptype, gap_value, pset)
        values.append(bias.get(base_price, np.nan))
        colors.append(MODEL_COLORS.get(model, DEFAULT_COLOR_CYCLE[i % len(DEFAULT_COLOR_CYCLE)]))
        labels.append(model)

      bars = ax.bar(labels, values, color=colors, width=0.55)
      for bar, v in zip(bars, values):
        if not np.isnan(v):
          ax.text(bar.get_x() + bar.get_width() / 2, v + 0.025, f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")

      ax.set_ylim(0, 1.0)
      ax.set_yticks(np.arange(0.0, 1.1, 0.2))
      ax.tick_params(axis="y", labelsize=11.5)

      if col_idx == 0:
        ax.set_ylabel(f"Order Bias\n[{p_label}]", fontsize=12.5, fontweight="bold", labelpad=8)

      ax.set_title(f"{SET_LABELS.get(pset, f'Set {pset}')} (Base: {p_label})", fontsize=13, fontweight="bold", pad=10)
      ax.tick_params(axis="x", rotation=15, labelsize=10.5)

  price_str = ", ".join([f"\\${p}" for p in prices])
  fig.suptitle(
      f"{grid_title_prefix} ({price_str}): Order Bias by Model ({gap_title_label(ptype, gap_value)})",
      fontsize=16, fontweight="bold", y=1.01,
  )
  fig.tight_layout()
  out_name = f"Slide_Grid_{out_suffix}_{gap_file_suffix(ptype, gap_value)}.png"
  out_path = os.path.join(output_dir, out_name)
  fig.savefig(out_path, dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved: {out_path}")


def _price_base_pages(bases, page_size=3):
  """Splits ALL_PRICE_BASES into <=page_size chunks so no single detail
  grid image exceeds the 3x3 cap (7 bases -> pages of 3, 3, 1)."""
  return [bases[i:i + page_size] for i in range(0, len(bases), page_size)]


# ── Execution ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
  df_clean = load_and_clean_data(temperature=1.0)

  # Combined: all gap sizes + both types together, one image per active set.
  generate_combined_overview_line(df_clean)
  generate_combined_grouped_bar(df_clean)
  generate_combined_average_bar(df_clean)

  # Separate: one set of images per individual (type, gap size) condition.
  all_conditions = (
      [("fixed_gap", g) for g in GAP_SIZES["fixed_gap"]]
      + [("pct_gap", g) for g in GAP_SIZES["pct_gap"]]
  )
  for ptype, gap_value in all_conditions:
    generate_separate_overview_line(df_clean, ptype, gap_value)
    generate_separate_grouped_bar(df_clean, ptype, gap_value)
    generate_separate_average_bar(df_clean, ptype, gap_value)

    for page_idx, price_page in enumerate(_price_base_pages(ALL_PRICE_BASES), 1):
      generate_price_grid_slide(
          df_clean, ptype, gap_value, price_page,
          "Detail", f"Page{page_idx}_Prices_{'_'.join(str(p) for p in price_page)}",
      )

  print("\nAll figures regenerated: combined all-gap-sizes views + separate per-condition views.")
