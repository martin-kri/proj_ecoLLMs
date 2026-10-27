"""
Summarizes every PlayStation-vs-Xbox run in one figure: the share of answers
naming PlayStation, per model and temperature, for both prompt orders.

Reads files named  <Order>_<model>_temp<T>.csv  (as written by the two
generate_console_*.py scripts), e.g. Xbox_or_PlayStation_gpt-5.4_temp1.0.csv

Usage:
    python plot_console.py --folder data --out figures
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDERS = ["PlayStation_or_Xbox", "Xbox_or_PlayStation"]
ORDER_LABELS = {"PlayStation_or_Xbox": "PlayStation listed first",
                "Xbox_or_PlayStation": "Xbox listed first"}
ORDER_COLORS = {"PlayStation_or_Xbox": "#2a78d6", "Xbox_or_PlayStation": "#eb6834"}
MODEL_ORDER = ["Llama-3.1-8B-Instruct", "gpt-5.4-nano", "gpt-5.4"]

FILE_RE = re.compile(r"^(PlayStation_or_Xbox|Xbox_or_PlayStation)_(.+)_temp(\d+(?:\.\d+)?)\.csv$")


def classify(text):
    """'playstation' / 'xbox' if exactly one is named, otherwise 'other'."""
    t = str(text).lower()
    ps, xb = "playstation" in t, "xbox" in t
    if ps and not xb:
        return "playstation"
    if xb and not ps:
        return "xbox"
    return "other"


def load(folder):
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.csv"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            print(f"  [SKIP – unrecognised name] {os.path.basename(path)}")
            continue
        order, model, temp = m.group(1), m.group(2), float(m.group(3))
        choices = pd.read_csv(path)["response"].apply(classify)
        n = len(choices)
        rows.append({
            "model": model, "temperature": temp, "order": order, "n": n,
            "n_playstation": int((choices == "playstation").sum()),
            "n_xbox": int((choices == "xbox").sum()),
            "n_other": int((choices == "other").sum()),
            "p_playstation": (choices == "playstation").sum() / n if n else np.nan,
        })
    return pd.DataFrame(rows)


def plot(df, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    models = [m for m in MODEL_ORDER if m in set(df["model"])] + \
             sorted(set(df["model"]) - set(MODEL_ORDER))
    groups = [(m, t) for m in models for t in sorted(df.loc[df["model"] == m, "temperature"].unique())]

    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(1.35 * len(groups) + 2, 4.6))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for i, order in enumerate(ORDERS):
        offset = (i - 0.5) * width
        for gx, (model, temp) in zip(x, groups):
            row = df[(df["model"] == model) & (df["temperature"] == temp) & (df["order"] == order)]
            if row.empty:
                continue
            p = row["p_playstation"].iloc[0]
            ax.bar(gx + offset, p, width - 0.03, color=ORDER_COLORS[order],
                   label=ORDER_LABELS[order] if gx == 0 else None)
            ax.text(gx + offset, p + 0.015, f"{p:.2f}", ha="center", va="bottom",
                    fontsize=8, color="#3d3d3a")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\nT={t:g}" for m, t in groups], fontsize=8.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Share of answers naming PlayStation")
    ax.axhline(0.5, color="#9a9a96", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="#e6e6e3", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=9)
    ax.set_title("PlayStation vs. Xbox — 100 answers per bar", fontsize=11, pad=26)

    fig.tight_layout()
    out_path = os.path.join(out_folder, "playstation_vs_xbox.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize PlayStation vs. Xbox runs")
    parser.add_argument("--folder", default=".", help="Folder with CSV files")
    parser.add_argument("--out", default="figures", help="Output folder for plots")
    args = parser.parse_args()

    df = load(args.folder)
    if df.empty:
        print(f"No matching CSV files in {args.folder}")
        return
    print(df.sort_values(["model", "temperature", "order"]).to_string(index=False))
    plot(df, args.out)


if __name__ == "__main__":
    main()
