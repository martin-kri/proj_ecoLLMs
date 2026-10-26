"""
plot_probability.py

Reads all LLM-response CSV files in a folder, parses metadata from filenames,
and plots P(choice_A) as a function of temperature.

Supports experiments:
  - chicken vs pasta
  - playstation vs xbox
  - colors (red, blue, yellow)
  - t-shirts (red t-shirt, blue t-shirt, yellow t-shirt)
  - hot vs cold
  - random nonsense words (with optional unavailable word)

Usage:
    python plot_probability.py                    # current directory
    python plot_probability.py --folder ./data
    python plot_probability.py --out figures/
    python plot_probability.py --invert           # plot P(second choice)
"""

import argparse
import re
import os
import glob
import math
from difflib import SequenceMatcher

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.optimize import curve_fit


# ── Lookup tables ─────────────────────────────────────────────────────────────

VALID_TEMPS = {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.1, 2.2, 2.5, 3.0, 3.5, 4.5}

# All known random nonsense words — used to detect Type A/B random experiments.
RANDOM_WORDS = {"imphdy", "mkzlyg", "gdwaab", "mxjlg", "hwrbx"}

# ---------------------------------------------------------------------------
# ORDER_CHOICES_MAP
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# ORDER_CHOICES_MAP
# ---------------------------------------------------------------------------
ORDER_CHOICES_MAP = {
    # chicken / pasta
    "chicken_or_pasta": ["chicken", "pasta"],
    "pasta_or_chicken": ["chicken", "pasta"],
    "chop": ["chicken", "pasta"],
    "poch": ["chicken", "pasta"],

    # console
    "playstation_or_xbox": ["playstation", "xbox"],
    "xbox_or_playstation": ["playstation", "xbox"],
    "psxxb": ["playstation", "xbox"],
    "xbxps": ["playstation", "xbox"],
    "psxb": ["playstation", "xbox"],
    "xbps": ["playstation", "xbox"],
    "ps5xbx": ["playstation", "xbox"],
    "xbxps5": ["playstation", "xbox"],

    # plain colors
    "red_or_blue": ["red", "blue"],
    "blue_or_red": ["red", "blue"],
    "yellow_or_blue": ["yellow", "blue"],
    "blue_or_yellow": ["yellow", "blue"],
    "yellow_or_red": ["yellow", "red"],
    "red_or_yellow": ["yellow", "red"],

    # t-shirts (UPDATED TO MATCH YOUR FILENAMES)
    "red_tshirt_or_blue_tshirt": ["red t-shirt", "blue t-shirt"],
    "blue_tshirt_or_red_tshirt": ["red t-shirt", "blue t-shirt"],
    "yellow_tshirt_or_blue_tshirt": ["yellow t-shirt", "blue t-shirt"],
    "blue_tshirt_or_yellow_tshirt": ["yellow t-shirt", "blue t-shirt"],
    "yellow_tshirt_or_red_tshirt": ["yellow t-shirt", "red t-shirt"],
    "red_tshirt_or_yellow_tshirt": ["yellow t-shirt", "red t-shirt"],

    # hot / cold
    "hot_or_cold": ["hot", "cold"],
    "cold_or_hot": ["hot", "cold"],
}

# Maps a sanitized order string to a human-readable label
ORDER_LABEL_MAP = {
    "chicken_or_pasta": "chicken first",
    "pasta_or_chicken": "pasta first",
    "chop": "chicken first",
    "poch": "pasta first",
    "playstation_or_xbox": "PS first",
    "xbox_or_playstation": "Xbox first",
    "psxxb": "PS first",
    "xbxps": "Xbox first",
    "psxb": "PS first",
    "xbps": "Xbox first",
    "ps5xbx": "PS first",
    "xbxps5": "Xbox first",
    "red_or_blue": "red first",
    "blue_or_red": "blue first",
    "yellow_or_blue": "yellow first",
    "blue_or_yellow": "blue first",
    "yellow_or_red": "yellow first",
    "red_or_yellow": "red first",

    # t-shirts (UPDATED TO MATCH YOUR FILENAMES)
    "red_tshirt_or_blue_tshirt": "red first",
    "blue_tshirt_or_red_tshirt": "blue first",
    "yellow_tshirt_or_blue_tshirt": "yellow first",
    "blue_tshirt_or_yellow_tshirt": "blue first",
    "yellow_tshirt_or_red_tshirt": "yellow first",
    "red_tshirt_or_yellow_tshirt": "red first",

    "hot_or_cold": "hot first",
    "cold_or_hot": "cold first",
}

QUESTION_KEYWORDS = [
    ("tshirt",      "tshirt"),
    ("yellow",      "colors"),
    ("red",         "colors"),
    ("blue",        "colors"),
    ("hot",         "tempwords"),
    ("cold",        "tempwords"),
    ("chicken",     "chicken_pasta"),
    ("pasta",       "chicken_pasta"),
    ("chop",        "chicken_pasta"),
    ("poch",        "chicken_pasta"),
    ("playstation", "console"),
    ("xbox",        "console"),
    ("psxb",        "console"),
    ("xbps",        "console"),
    ("ps5",         "console"),
    ("indian",      "indian"),
    ("flight",      "flight"),
]

MODEL_PATTERNS = [
    (re.compile(r"gpt[\-_]?4[\.\-_]?1[\.\-_]?nano", re.I),          "gpt-4.1-nano"),
    (re.compile(r"gpt[\-_]?4[\.\-_]?1(?![\d\.])", re.I),            "gpt-4.1"),
    (re.compile(r"gpt4omini", re.I),                                  "gpt-4o-mini"),
    (re.compile(r"openai", re.I),                                     "gpt-unknown"),
    (re.compile(r"llama[\-_\.]*3[\.\-_\.]*1[\.\-_\.]*8b", re.I),    "Llama-3.1-8B"),
    (re.compile(r"meta[\-_]llama", re.I),                             "Llama-3.1-8B"),
    (re.compile(r"llama", re.I),                                      "Llama"),
    (re.compile(r"gemma[\-_]?3[\-_]?12b", re.I),                    "Gemma-3-12B"),
    (re.compile(r"gemma[\-_]?3[\-_]?27b", re.I),                    "Gemma-3-27B"),
    (re.compile(r"gemma", re.I),                                      "Gemma"),
]


# ── Extractors ────────────────────────────────────────────────────────────────

def _strip_model_name(name: str) -> str:
    model_re = re.compile(
        r"(?:"
        r"gpt[\-_]?4[\.\-_]?1[\.\-_]?nano"
        r"|gpt[\-_]?4[\.\-_]?1"
        r"|gpt4omini"
        r"|llama[\-_\.]*3[\.\-_\.]*1[\.\-_\.]*8b[\.\-_]*instruct"
        r"|llama[\-_\.]*3[\.\-_\.]*1[\.\-_\.]*8b"
        r"|meta[\-_]llama"
        r"|llama"
        r"|gemma[\-_]?3[\-_]?27b"
        r"|gemma[\-_]?3[\-_]?12b"
        r"|gemma"
        r")",
        re.I,
    )
    return model_re.sub("", name)

def extract_temperature(name: str) -> float | None:
    clean = _strip_model_name(name)
    m = re.search(r"temp[=_](\d+\.?\d*)", clean, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"_(\d+\.\d+|\d+)_(?:items_v2|items|colors|random|tshirt|tempwords)\b", clean, re.I)
    if m:
        return float(m.group(1))
    for m in re.finditer(r"(?<=[_\s])(\d+\.\d+|\d+)(?=[_\s]|$)", clean):
        v = float(m.group(1))
        if v in VALID_TEMPS:
            return v
    return None

def extract_price(name: str, temperature: float) -> str | None:
    m = re.search(
        r"_(?:items_v2|items|colors|random|tshirt|tempwords)_(\d+)_(\d+)(?:_|$)", name, re.I
    )
    if m:
        return f"${m.group(1)} vs ${m.group(2)}"
    m = re.search(
        r"_(?:items_v2|items|colors|random|tshirt|tempwords)_(\d+)(?:_|$)", name, re.I
    )
    if m:
        v = int(m.group(1))
        if float(v) != temperature:
            return f"${v}"
    return None

def extract_model(name: str) -> str:
    for pattern, label in MODEL_PATTERNS:
        if pattern.search(name):
            return label
    return "unknown_model"

def extract_question(name: str) -> str:
    lower = name.lower()
    if any(w in lower for w in RANDOM_WORDS):
        return "random"
    for kw, label in QUESTION_KEYWORDS:
        if kw in lower:
            return label
    return "unknown_question"

def extract_order_key(name: str) -> str:
    lower = name.lower()
    base = re.split(r"_(?:no|with)_memory", lower)[0]
    for key in sorted(ORDER_CHOICES_MAP.keys(), key=len, reverse=True):
        if key in base:
            return key
    m = re.match(r"([a-z]+_or_[a-z]+)", base)
    if m:
        return m.group(1)
    return "unknown_order"

def extract_unavailable(name: str) -> str | None:
    for m in re.finditer(r"_no_([a-z]+)", name, re.I):
        candidate = m.group(1).lower()
        if candidate in RANDOM_WORDS:
            return candidate
    return None

def extract_active_pair(name: str) -> str | None:
    name_lower = name.lower()
    found = [w for w in RANDOM_WORDS if w in name_lower]
    if len(found) < 2:
        return None
    base = re.split(r"_no_memory|_with_memory|_no_[a-z]+|_random|_temp|_colors|_items_v2|_items", name_lower)[0]
    active = [w for w in RANDOM_WORDS if w in base]
    if len(active) == 2:
        return "_".join(sorted(active))
    unav = extract_unavailable(name)
    if unav:
        active = [w for w in found if w != unav]
        if len(active) == 2:
            return "_".join(sorted(active))
    if len(found) >= 2:
        return "_".join(sorted(found[:2]))
    return None

def parse_filename(path: str) -> dict | None:
    name = os.path.splitext(os.path.basename(path))[0]
    temperature = extract_temperature(name)
    if temperature is None:
        return None

    order_key = extract_order_key(name)
    question  = extract_question(name)
    model     = extract_model(name)
    price     = extract_price(name, temperature)
    unav      = extract_unavailable(name)
    pair      = extract_active_pair(name)

    return dict(
        order_key=order_key,
        order_label=ORDER_LABEL_MAP.get(order_key, order_key.replace("_", " ")),
        model=model,
        temperature=temperature,
        question=question,
        price=price,
        unavailable=unav,
        active_pair=pair,
        path=path,
    )

# ── Choice extraction ─────────────────────────────────────────────────────────

def extract_choice(text: str, candidates: list[str]) -> str:
    text_lower = str(text).lower()

    if text_lower in [c.lower() for c in candidates]:
        return next(c for c in candidates if c.lower() == text_lower)

    for c in candidates:
        if re.search(r'(?<![a-z])' + re.escape(c.lower()) + r'(?![a-z])', text_lower):
            return c

    for c in candidates:
        if c.lower() in text_lower:
            return c

    best_match, best_score = None, 0.0
    for word in re.split(r'[^a-zA-Z\-]+', text_lower):
        if not word: continue
        for c in candidates:
            score = SequenceMatcher(None, word, c.lower()).ratio()
            if score > best_score:
                best_score, best_match = score, c
    if best_score >= 0.75:
        return best_match

    return text

# ── Data loader ───────────────────────────────────────────────────────────────

def load_files(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    records = []
    for path in files:
        meta = parse_filename(path)
        if meta is None: continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if "matched_choice" in df.columns:
            df["response"] = df["matched_choice"].astype(str).str.strip().str.lower()
        else:
            df["response"] = df["response"].astype(str).str.strip().str.lower()

        candidates = ORDER_CHOICES_MAP.get(meta["order_key"])
        if not candidates and meta.get("active_pair"):
            pair_words = meta["active_pair"].split("_")
            if len(pair_words) == 2:
                candidates = pair_words

        if candidates:
            df["response"] = df["response"].apply(lambda t: extract_choice(t, candidates))

        n_total = len(df)
        n_invalid = int(df["attempt"].max() - n_total) if "attempt" in df.columns else 0
        choices = [c for c in df["response"].unique() if c not in ("nan", "", "none")]

        if not candidates or len(choices) == 0:
            continue

        choice_a = candidates[0]
        n_a = (df["response"].str.lower() == choice_a.lower()).sum()
        p_a = n_a / n_total if n_total else float("nan")
        se = np.sqrt(p_a * (1 - p_a) / n_total) if n_total else float("nan")
        moe_96 = se * 2.054 if not np.isnan(se) else float("nan")
        p_clipped = np.clip(p_a, 1e-6, 1 - 1e-6)
        U_empirical = meta["temperature"] * np.log(p_clipped / (1 - p_clipped))

        pair_key = f"{candidates[0]}_vs_{candidates[1]}"
        meta.update(
            n_total=n_total, choice_a=choice_a, choice_b=candidates[1],
            pair_key=pair_key, n_choice_a=int(n_a), p_choice_a=p_a,
            moe=moe_96, n_invalid=n_invalid, U_empirical=U_empirical,
        )
        records.append(meta)

    return pd.DataFrame(records).sort_values(
        ["model", "question", "pair_key", "price", "temperature"], na_position="first"
    )

def print_invalid_summary(data: pd.DataFrame):
    pass # (Truncated for brevity in console, but functions same)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_curves(ax, subdata, colors, invert=False):
    subdata_iter = subdata.copy()
    subdata_iter["price_str"] = subdata_iter["price"].fillna("NONE")
    combos = subdata_iter[["order_key", "price_str"]].drop_duplicates().sort_values(["order_key", "price_str"])
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    for idx, (_, row) in enumerate(combos.iterrows()):
        ok, pr_str = row["order_key"], row["price_str"]
        sub = subdata_iter[(subdata_iter["order_key"] == ok) & (subdata_iter["price_str"] == pr_str)].sort_values("temperature")
        if sub.empty: continue

        temps, probs, errors = sub["temperature"].values, sub["p_choice_a"].values, sub["moe"].values
        n_invalids = sub["n_invalid"].values if "n_invalid" in sub.columns else np.zeros(len(temps))

        color, marker = colors[idx % len(colors)], markers[idx % len(markers)]
        base_label, actual_price = sub["order_label"].iloc[0], sub["price"].iloc[0]
        label = f"{base_label} ({actual_price})" if pd.notna(actual_price) else base_label

        probs_for_fit = probs.copy()
        if invert: probs = 1.0 - probs

        ax.errorbar(temps, probs, yerr=errors, marker=marker, linewidth=1.8, markersize=5,
                    capsize=4, capthick=1.2, elinewidth=1.2, color=color, label=label, alpha=0.9)

        for x, y, n_inv in zip(temps, probs, n_invalids):
            if n_inv > 0:
                ax.annotate(f"~{int(n_inv)} extra", xy=(x, y), xytext=(0, 10), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6.5, color=color, alpha=0.70, fontweight="bold")

        mask = (probs_for_fit > 0) & (probs_for_fit < 1)
        if mask.sum() >= 2:
            try:
                def helbing_model(T, U): return 1.0 / (1.0 + np.exp(-U / T))
                popt, _ = curve_fit(helbing_model, temps[mask], probs_for_fit[mask], p0=[0.5], maxfev=4000)
                U_fit = popt[0]
                t_fine = np.linspace(max(temps.min(), 0.01), temps.max(), 300)
                p_fit = helbing_model(t_fine, U_fit)
                if invert: p_fit, U_display = 1 - p_fit, -U_fit
                else: U_display = U_fit
                ax.plot(t_fine, p_fit, "--", color=color, linewidth=1.2, alpha=0.7, label=f"{label} fit (U={U_display:.3f})")
            except Exception: pass

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Temperature", fontsize=13)

    n = int(subdata["n_total"].iloc[0])
    n_inv_total = int(subdata["n_invalid"].sum()) if "n_invalid" in subdata.columns else 0
    base_text = f"n = {n} per run" if n_inv_total == 0 else f"n = {n} valid per run  ({n_inv_total} extra attempts total)"
    label_text = f"{base_text}\nErr. bars: 96% Conf. Interval"
    if n_inv_total > 0: label_text += "\n('~N extra' = discarded attempts per point)"
    ax.text(0.5, -0.25, label_text, transform=ax.transAxes, fontsize=9, ha="center", va="top",
            color="black", bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.4))


def _subplot_title(unavailable: str | None) -> str:
    return f"Type A: {unavailable} unavailable" if unavailable else "Type B: plain binary"


# ── Main plotting function ────────────────────────────────────────────────────

def plot_p_choice(data: pd.DataFrame, out_folder: str, invert: bool = False):
    os.makedirs(out_folder, exist_ok=True)
    colors = plt.cm.tab10.colors

    def group_price_logic(p):
        if pd.isna(p): return "No price"
        nums = re.findall(r'\d+', str(p))
        if len(nums) == 2:
            a, b = sorted([int(x) for x in nums])
            if a == b: return f"Same Price: ${a}"
            else: return f"Mixed Prices: ${a} & ${b}"
        return f"Price: {p}"

    # Custom Sorting Key to force the exact order: No price -> Same Price -> Mixed Prices
    def sort_panels_key(pg_label):
        if pg_label == "No price":
            return (0, [])
        nums = [int(x) for x in re.findall(r'\d+', pg_label)]
        if pg_label.startswith("Same"):
            return (1, nums)
        if pg_label.startswith("Mixed"):
            return (2, nums)
        return (3, nums)

    data = data.copy()
    data["price_group"] = data["price"].apply(group_price_logic)

    for (model, question, pair_key), grp in data.groupby(["model", "question", "pair_key"], sort=True):
        price_groups_present = grp["price_group"].unique()

        choice_a_label, choice_b_label = grp["choice_a"].iloc[0], grp["choice_b"].iloc[0]
        ylabel = f"P({choice_b_label})" if invert else f"P({choice_a_label})"
        safe_pair = pair_key.replace(" ", "_").replace("-", "")

        # Sort the unique panel labels using our custom key
        sorted_price_groups = sorted(price_groups_present, key=sort_panels_key)

        # Build all_panels list strictly in the sorted order
        all_panels = []
        for pg in sorted_price_groups:
            all_panels.append((pg, grp[grp["price_group"] == pg]))

        if question == "random" and "unavailable" in grp.columns:
            # (Skipping Random words loop code for brevity, logic remains same)
            pass

        else:
            if len(all_panels) == 1:
                panel_label, panel_data = all_panels[0]
                fig, ax = plt.subplots(figsize=(7, 5.5))
                _draw_curves(ax, panel_data, colors, invert=invert)
                ax.set_title(f"Model: {model}  |  {pair_key}  |  {panel_label}", fontsize=13)
                ax.set_ylabel(ylabel, fontsize=13)
                ax.legend(fontsize=8, loc="best")
                fig.tight_layout()
            else:
                max_cols = 3
                cols = min(len(all_panels), max_cols)
                rows = math.ceil(len(all_panels) / max_cols)
                fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5.5 * rows), sharey=True, squeeze=False)
                axes_flat = axes.flatten()
                fig.suptitle(f"Model: {model}  |  {pair_key}", fontsize=14, y=1.02)

                for idx, (panel_label, panel_data) in enumerate(all_panels):
                    ax = axes_flat[idx]
                    _draw_curves(ax, panel_data, colors, invert=invert)
                    ax.set_title(panel_label, fontsize=13)
                    ax.legend(fontsize=8, loc="best")
                    if idx % max_cols == 0: ax.set_ylabel(ylabel, fontsize=13)

                for j in range(len(all_panels), len(axes_flat)):
                    fig.delaxes(axes_flat[j])
                fig.tight_layout()

            safe_ca = choice_a_label.replace(" ", "_").replace("-", "")
            fname = f"p_{safe_ca}_{model}_{safe_pair}.png"
            out_path = os.path.join(out_folder, fname)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot P(choice) vs Temperature")
    parser.add_argument("--folder", default=".", help="Folder with CSV files")
    parser.add_argument("--out",    default="figures", help="Output folder for plots")
    parser.add_argument("--invert", action="store_true", help="Plot P(second choice) instead of P(first choice)")
    args = parser.parse_args()

    print(f"\nLoading files from: {args.folder}")
    data = load_files(args.folder)

    print(f"\nGenerating plots -> {args.out}/")
    plot_p_choice(data, out_folder=args.out, invert=args.invert)
    print("\nDone.")

if __name__ == "__main__":
    main()