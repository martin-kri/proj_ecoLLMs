"""
plot_pasta_probability.py

Reads all LLM-response CSV files in a folder, parses metadata from filenames,
and plots P(choice_A) as a function of temperature — plus a companion panel
showing P vs U/T (the natural sigmoid axis).

Grouping variables extracted from filenames:
  - model        (gpt-4.1-nano, Llama-3.1-8B, ...)
  - question     (chicken_pasta, flight, indian, console)
  - order        (chicken_first, pasta_first, PS_first, XB_first)
  - price        (e.g., "$5 vs $20", "$5", or None if no price in filename)

Plot layout:
  - One figure per (model, question)
  - If price variants exist: grid layout with max 3 columns per row
  - Each price column has two stacked panels: P vs T (top) and P vs U/T (bottom)
  - Within each panel: one curve/set of points per order

Usage:
    python plot_pasta_probability.py                    # current directory
    python plot_pasta_probability.py --folder ./data
    python plot_pasta_probability.py --out figures/
    python plot_pasta_probability.py --invert           # plot second choice
"""

import argparse
import re
import os
import glob
import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit


# ── Lookup tables ─────────────────────────────────────────────────────────────

VALID_TEMPS = {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.1, 2.2, 2.5}

ORDER_MAP = {
    "chicken or pasta":    "chicken_first",
    "chicken_or_pasta":    "chicken_first",
    "pasta or chicken":    "pasta_first",
    "pasta_or_chicken":    "pasta_first",
    "chop":                "chicken_first",
    "poch":                "pasta_first",
    "playstation or xbox": "PS_first",
    "xbox or playstation": "XB_first",
    "ps5xbx":  "PS_first",
    "xbxps5":  "XB_first",
    "psxxb":   "PS_first",
    "xbxps":   "XB_first",
    "psxb":    "PS_first",
    "xbps":    "XB_first",
}

MODEL_PATTERNS = [
    (re.compile(r"gpt[\-_]?5[\.\-_]?4[\.\-_]?nano", re.I), "gpt-4.1-nano"),
    (re.compile(r"gpt[\-_]?5[\.\-_]?4(?![\d\.])", re.I),   "gpt-4.1"),
    (re.compile(r"gpt5nano", re.I),                          "gpt-4o-mini"),
    (re.compile(r"openai", re.I),                            "gpt-unknown"),
    (re.compile(r"llama[\-_\.]*3[\.\-_]*1[\.\-_]*8b", re.I), "Llama-3.1-8B"),
    (re.compile(r"meta[\-_]llama", re.I),                     "Llama-3.1-8B"),
    (re.compile(r"llama", re.I),                              "Llama"),
]

QUESTION_KEYWORDS = {
    "indian":      "indian",
    "flight":      "flight",
    "chicken":     "chicken_pasta",
    "pasta":       "chicken_pasta",
    "chop":        "chicken_pasta",
    "poch":        "chicken_pasta",
    "playstation": "console",
    "xbox":        "console",
    "psxb":        "console",
    "xbps":        "console",
    "psxxb":       "console",
    "xbxps":       "console",
    "ps5":         "console",
}


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_temperature(name: str) -> float | None:
    m = re.search(r"temp[=_](\d+\.?\d*)", name, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"instruct[_\-\s]+(\d+\.?\d*)", name, re.I)
    if m:
        v = float(m.group(1))
        if v in VALID_TEMPS or 0 <= v <= 2:
            return v
    m = re.search(r"[_\s](\d+\.?\d*)$", name)
    if m:
        v = float(m.group(1))
        if v in VALID_TEMPS:
            return v
    return None


def extract_price(name: str, temperature: float) -> str | None:
    m_pair = re.search(r"_(\d+)_(\d+)$", name)
    if m_pair:
        return f"${m_pair.group(1)} vs ${m_pair.group(2)}"
    m_single = re.search(r"_(\d+)$", name)
    if not m_single:
        return None
    v = int(m_single.group(1))
    if float(v) == temperature or v in (0, 1, 2):
        return None
    return f"${v}"


def extract_order(name: str) -> str:
    lower = name.lower()
    for phrase in sorted(ORDER_MAP, key=len, reverse=True):
        if phrase in lower:
            return ORDER_MAP[phrase]
    return "unknown_order"


def extract_model(name: str) -> str:
    for pattern, label in MODEL_PATTERNS:
        if pattern.search(name):
            return label
    return "unknown_model"


def extract_question(name: str) -> str:
    lower = name.lower()
    for kw, label in QUESTION_KEYWORDS.items():
        if kw in lower:
            return label
    return "unknown_question"


def parse_filename(path: str) -> dict | None:
    name = os.path.splitext(os.path.basename(path))[0]
    temperature = extract_temperature(name)
    if temperature is None:
        print(f"  [SKIP – no temperature] {name}")
        return None
    test_match = re.search(r"TEST(\d+)", name, re.I)
    test_id = int(test_match.group(1)) if test_match else -1
    return dict(
        test_id=test_id,
        order=extract_order(name),
        model=extract_model(name),
        temperature=temperature,
        question=extract_question(name),
        price=extract_price(name, temperature),
        path=path,
    )


# ── Data loader ───────────────────────────────────────────────────────────────

def load_files(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    records = []
    for path in files:
        meta = parse_filename(path)
        if meta is None:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"  [SKIP – read error] {os.path.basename(path)}: {e}")
            continue

        df["response"] = df["response"].astype(str).str.strip().str.lower()

        ORDER_TO_CHOICES = {
            "chicken_first": ["chicken", "pasta"],
            "pasta_first":   ["chicken", "pasta"],
            "PS_first":      ["playstation", "xbox"],
            "XB_first":      ["playstation", "xbox"],
        }
        known_choices = ORDER_TO_CHOICES.get(meta["order"], [])

        def extract_choice(text, candidates):
            from difflib import SequenceMatcher
            text = str(text).lower()
            if text in candidates:
                return text
            # Correction added after the internship: an answer naming both
            # options used to count as the first one in the list (chicken),
            # even for "pasta, it's cheaper than the chicken". Now it isn't
            # counted for either option.
            named = [c for c in candidates if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', text)]
            if len(named) > 1:
                return text
            for c in candidates:
                if re.search(r'(?<![a-z])' + re.escape(c) + r'(?![a-z])', text):
                    return c
            for c in candidates:
                if c in text:
                    return c
            best_match, best_score = None, 0.0
            for word in re.split(r'[^a-zA-Z؀-ۿ]+', text):
                if not word:
                    continue
                for c in candidates:
                    score = SequenceMatcher(None, word.lower(), c).ratio()
                    if score > best_score:
                        best_score, best_match = score, c
            if best_score >= 0.75:
                return best_match
            return text

        if known_choices:
            df["response"] = df["response"].apply(
                lambda t: extract_choice(t, known_choices)
            )

        n_total = len(df)
        n_invalid = int((~df["response"].isin(known_choices + ["nan", ""])).sum()) if known_choices else 0
        # Correction added after the internship: P(choice) is the share among
        # classified answers; unclassified ones no longer count against choice_a.
        if known_choices:
            df = df[df["response"].isin(known_choices)]
            n_total = len(df)
        choices = sorted(c for c in df["response"].unique() if c not in ("nan", ""))

        if len(choices) < 2:
            all_candidates = ORDER_TO_CHOICES.get(meta["order"], [])
            if len(choices) == 1 and all_candidates:
                choice_a = all_candidates[0]
                n_a = (df["response"] == choice_a).sum()
                p_a = n_a / n_total if n_total else float("nan")
                se = (np.sqrt(p_a * (1 - p_a) / n_total)) if n_total else float("nan")
                moe_96 = se * 2.054 if not np.isnan(se) else float("nan")
                print(
                    f"  {os.path.basename(path)[:62]:<64}"
                    f"T={meta['temperature']:.1f}  price={meta['price']}  "
                    f"P({choice_a})={p_a:.3f} ({n_a}/{n_total})  "
                    f"[only one choice seen]"
                )
            else:
                print(f"  [SKIP – no recognisable responses] {os.path.basename(path)}: {choices}")
                continue
        else:
            choice_a = choices[0]
            n_a = (df["response"] == choice_a).sum()
            p_a = n_a / n_total if n_total else float("nan")

        se = (np.sqrt(p_a * (1 - p_a) / n_total)) if n_total else float("nan")
        moe_96 = se * 2.054 if not np.isnan(se) else float("nan")

        p_clipped = np.clip(p_a, 1e-6, 1 - 1e-6)
        U_empirical = meta["temperature"] * np.log(p_clipped / (1 - p_clipped))

        meta.update(n_total=n_total, choice_a=choice_a,
                    n_choice_a=int(n_a), p_choice_a=p_a, moe=moe_96,
                    n_invalid=n_invalid, U_empirical=U_empirical)

        price_str = f"  price={meta['price']}" if meta['price'] else ""
        invalid_str = f"  [{n_invalid} invalid]" if n_invalid else ""
        print(
            f"  {os.path.basename(path)[:62]:<64}"
            f"T={meta['temperature']:.1f}{price_str}  "
            f"P({choice_a})={p_a:.3f} ({n_a}/{n_total}){invalid_str}"
        )
        records.append(meta)

    if not records:
        raise ValueError("No valid files could be parsed.")

    return pd.DataFrame(records).sort_values(
        ["temperature", "price"], na_position="first"
    )


# ── Shared U-fit helper ───────────────────────────────────────────────────────

def _fit_U(temps, probs):
    """Return fitted U for this (temps, probs) series, or None on failure."""
    def helbing_model(T, U):
        return 1.0 / (1.0 + np.exp(-U / T))

    mask = (probs > 0) & (probs < 1)
    if mask.sum() < 2:
        return None
    try:
        popt, _ = curve_fit(
            helbing_model, temps[mask], probs[mask],
            p0=[0.5], maxfev=4000,
        )
        return popt[0]
    except Exception:
        return None


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_curves(ax, subdata, colors, invert=False):
    """Top panel: P vs Temperature (original view)."""
    orders = sorted(subdata["order"].unique())
    for idx, order in enumerate(orders):
        sub = subdata[subdata["order"] == order].sort_values("temperature")
        temps = sub["temperature"].values
        probs = sub["p_choice_a"].values
        color = colors[idx % len(colors)]
        label = order.replace("_", " ")

        probs_for_fit = probs.copy()
        probs_plot = 1.0 - probs if invert else probs
        errors = sub["moe"].values

        ax.errorbar(temps, probs_plot, yerr=errors,
                    marker="o", linewidth=1.8, markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    color=color, label=label)

        mask = (probs_for_fit > 0) & (probs_for_fit < 1)
        if mask.sum() >= 2:
            U_fit = _fit_U(temps, probs_for_fit)
            if U_fit is not None:
                def helbing_model(T, U):
                    return 1.0 / (1.0 + np.exp(-U / T))

                t_fine = np.linspace(max(temps.min(), 0.01), temps.max(), 300)
                p_fit = helbing_model(t_fine, U_fit)
                if invert:
                    p_fit = 1 - p_fit
                    U_display = -U_fit
                else:
                    U_display = U_fit

                ax.plot(t_fine, p_fit, "--", color=color, linewidth=1.4,
                        alpha=0.8, label=f"{label} (U={U_display:.3f})")

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Temperature", fontsize=11)

    n_min, n_max = int(subdata["n_total"].min()), int(subdata["n_total"].max())
    n = f"{n_max}" if n_min == n_max else f"{n_min}–{n_max}"
    n_inv = int(subdata["n_invalid"].sum()) if "n_invalid" in subdata.columns else 0
    base_text = f"n = {n} per run" if n_inv == 0 else f"n = {n} per run  ({n_inv} invalid total)"
    label_text = f"{base_text}\nError bars: 96% Confidence Interval"
    ax.text(0.55, 0.40, label_text, transform=ax.transAxes,
            fontsize=9, ha="right", va="bottom", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.4))


def _draw_curves_ut(ax, subdata, colors, invert=False):
    """Bottom panel: P vs U/T — the natural sigmoid axis.

    The universal sigmoid S-curve is drawn once in gray.
    Each order's data points are plotted at x = U_fit / T,
    so all orders collapse onto the same S regardless of their U value.
    Horizontal scatter reflects how far each temperature sits along the curve.
    """
    # Universal sigmoid background
    x_sig = np.linspace(-9, 9, 600)
    y_sig = 1.0 / (1.0 + np.exp(-x_sig))
    if invert:
        y_sig = 1.0 - y_sig
    ax.plot(x_sig, y_sig, "-", color="gray", linewidth=1.5,
            alpha=0.35, zorder=1, label="sigmoid (universal)")

    orders = sorted(subdata["order"].unique())
    for idx, order in enumerate(orders):
        sub = subdata[subdata["order"] == order].sort_values("temperature")
        temps  = sub["temperature"].values
        probs  = sub["p_choice_a"].values
        errors = sub["moe"].values
        color  = colors[idx % len(colors)]
        label  = order.replace("_", " ")

        U_fit = _fit_U(temps, probs)
        if U_fit is None:
            continue

        # x-coordinate for each data point: U_fit / T
        x_ut = U_fit / temps
        probs_plot = 1.0 - probs if invert else probs

        # if inverted the logistic argument flips sign
        if invert:
            x_ut = -x_ut
            U_display = -U_fit
        else:
            U_display = U_fit

        ax.errorbar(x_ut, probs_plot, yerr=errors,
                    marker="o", linewidth=0, markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    color=color, zorder=3,
                    label=f"{label} (U={U_display:.3f})")

        # Annotate each point with its temperature value
        for xi, yi, ti in zip(x_ut, probs_plot, temps):
            ax.annotate(f"{ti:.1f}", (xi, yi),
                        textcoords="offset points", xytext=(0, 7),
                        fontsize=7, ha="center", color=color, alpha=0.8)

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axvline(0.0, color="gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_xlim(-9, 9)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("U / T  (logistic argument)", fontsize=11)

    ax.text(0.02, 0.04, "← low T, strong pref.\n→ high T, no pref.",
            transform=ax.transAxes, fontsize=7.5, color="gray",
            va="bottom", linespacing=1.5)


# ── Main plotting function ────────────────────────────────────────────────────

def plot_p_choice(data: pd.DataFrame, out_folder: str, invert: bool = False):
    os.makedirs(out_folder, exist_ok=True)
    colors = plt.cm.tab10.colors

    for (model, question), grp in data.groupby(["model", "question"]):

        choice_a_label = grp["choice_a"].mode().iloc[0]

        _ORDER_TO_CHOICES = {
            "chicken_first": ["chicken", "pasta"],
            "pasta_first":   ["chicken", "pasta"],
            "PS_first":      ["playstation", "xbox"],
            "XB_first":      ["playstation", "xbox"],
        }
        _candidates = _ORDER_TO_CHOICES.get(grp["order"].iloc[0], [choice_a_label, "other"])
        choice_b_label = next((c for c in _candidates if c != choice_a_label), "other")
        ylabel = f"P({choice_b_label})" if invert else f"P({choice_a_label})"

        prices = sorted(p for p in grp["price"].unique() if pd.notna(p))
        priced_data   = grp[grp["price"].notna()]
        unpriced_data = grp[grp["price"].isna()]
        has_price = len(prices) > 0

        if has_price:
            n_prices  = len(prices)
            max_cols  = 3
            cols      = min(n_prices, max_cols)
            price_rows = math.ceil(n_prices / max_cols)

            # Each price row is duplicated: top = P vs T, bottom = P vs U/T
            # Layout: [T_row_0, T_row_1, ..., UT_row_0, UT_row_1, ...]
            # But we interleave them so each price's two panels sit together:
            # Row 0: P vs T for prices 0..cols-1
            # Row 1: P vs U/T for prices 0..cols-1
            # Row 2: P vs T for prices cols..2*cols-1
            # Row 3: P vs U/T for prices cols..2*cols-1  etc.
            total_rows = 2 * price_rows

            fig, axes = plt.subplots(
                total_rows, cols,
                figsize=(5.5 * cols, 4.2 * total_rows),
                squeeze=False,
            )

            fig.suptitle(
                f"Model: {model}  |  Question: {question}\n"
                f"Top rows: P vs Temperature  ·  Bottom rows: P vs U/T",
                fontsize=12, y=1.01,
            )

            for idx, price in enumerate(prices):
                price_row = idx // max_cols   # which group of rows (0, 1, ...)
                col       = idx % max_cols

                ax_t  = axes[price_row * 2    ][col]   # P vs T panel
                ax_ut = axes[price_row * 2 + 1][col]   # P vs U/T panel

                sub = priced_data[priced_data["price"] == price]

                # P vs T (top)
                _draw_curves(ax_t, sub, colors, invert=invert)
                ax_t.set_title(f"Prices: {price}", fontsize=11)
                ax_t.legend(fontsize=8, loc="best")
                if col == 0:
                    ax_t.set_ylabel(ylabel, fontsize=11)

                # P vs U/T (bottom)
                _draw_curves_ut(ax_ut, sub, colors, invert=invert)
                ax_ut.legend(fontsize=8, loc="upper left")
                if col == 0:
                    ax_ut.set_ylabel(ylabel, fontsize=11)

                # Thin separator line between the two panel types
                ax_t.spines["bottom"].set_linewidth(1.5)
                ax_ut.spines["top"].set_linewidth(1.5)

            # Hide leftover empty subplots
            for j in range(n_prices, price_rows * cols):
                pr = j // max_cols
                c  = j % max_cols
                fig.delaxes(axes[pr * 2    ][c])
                fig.delaxes(axes[pr * 2 + 1][c])

            fig.tight_layout()

        else:
            # No price variants: single column, two rows
            fig, (ax_t, ax_ut) = plt.subplots(
                2, 1,
                figsize=(7, 8.5),
                gridspec_kw={"hspace": 0.45},
            )

            _draw_curves(ax_t, unpriced_data, colors, invert=invert)
            ax_t.set_title(
                f"Model: {model}  |  Question: {question}\nP vs Temperature",
                fontsize=12,
            )
            ax_t.set_ylabel(ylabel, fontsize=11)
            ax_t.legend(fontsize=9, loc="best")

            _draw_curves_ut(ax_ut, unpriced_data, colors, invert=invert)
            ax_ut.set_title("P vs U/T  (natural sigmoid axis)", fontsize=11)
            ax_ut.set_ylabel(ylabel, fontsize=11)
            ax_ut.legend(fontsize=9, loc="upper left")

            fig.tight_layout()

        fname = f"p_{choice_a_label}_{model}_{question}.png".replace(" ", "_")
        out_path = os.path.join(out_folder, fname)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot P(choice) vs Temperature + U/T")
    parser.add_argument("--folder", default=".", help="Folder with CSV files")
    parser.add_argument("--out", default="figures", help="Output folder for plots")
    parser.add_argument("--invert", action="store_true",
                        help="Plot P(second choice) instead of P(first choice)")
    args = parser.parse_args()

    print(f"\nLoading files from: {args.folder}")
    data = load_files(args.folder)

    print(f"\nSummary table:")
    print(
        data[["test_id", "model", "question", "order", "temperature",
              "price", "choice_a", "p_choice_a", "U_empirical", "n_total", "n_invalid"]]
        .sort_values(["model", "question", "price", "order", "temperature"])
        .to_string(index=False)
    )

    print(f"\nGenerating plots -> {args.out}/")
    plot_p_choice(data, out_folder=args.out, invert=args.invert)
    print("\nDone.")


if __name__ == "__main__":
    main()