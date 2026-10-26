"""
plot_pasta_probability.py

Reads all LLM-response CSV files in a folder, parses metadata from filenames,
and plots P(choice_A) as a function of temperature.

Grouping variables extracted from filenames:
  - model        (gpt-4.1-nano, Llama-3.1-8B, gemma-3-12b, ...)
  - question     (chicken_pasta, flight, indian, console, random)
  - order        (chicken_first, pasta_first, PS_first, XB_first, mxjlg_first, hwrbx_first, ...)
  - price        (e.g., "$5 vs $20", "$5", or None if no price in filename)
  - unavailable  (for random-word experiments: the word mentioned as unavailable,
                  or None for plain binary runs — parsed from "_no_<word>" in filename)

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

VALID_TEMPS = {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.7, 2.0, 2.1, 2.2, 2.5, 3.5, 4.5}

ORDER_MAP = {
    # chicken / pasta
    "chicken or pasta":    "chicken_first",
    "chicken_or_pasta":    "chicken_first",
    "pasta or chicken":    "pasta_first",
    "pasta_or_chicken":    "pasta_first",
    "chop":                "chicken_first",
    "poch":                "pasta_first",
    # console
    "playstation or xbox": "PS_first",
    "xbox or playstation": "XB_first",
    "ps5xbx":  "PS_first",
    "xbxps5":  "XB_first",
    "psxxb":   "PS_first",
    "xbxps":   "XB_first",
    "psxb":    "PS_first",
    "xbps":    "XB_first",
    # random words (original pair)
    "mxjlg or hwrbx":  "mxjlg_first",
    "mxjlg_or_hwrbx":  "mxjlg_first",
    "hwrbx or mxjlg":  "hwrbx_first",
    "hwrbx_or_mxjlg":  "hwrbx_first",
    # random words (new pair)
    "imphdy or mkzlyg": "imphdy_first",
    "imphdy_or_mkzlyg": "imphdy_first",
    "mkzlyg or imphdy": "mkzlyg_first",
    "mkzlyg_or_imphdy": "mkzlyg_first",
    # gdwaab variants (if used as first word in a pair with another)
    "gdwaab or imphdy": "gdwaab_first",
    "gdwaab_or_imphdy": "gdwaab_first",
    "imphdy or gdwaab": "imphdy_first",
    "imphdy_or_gdwaab": "imphdy_first",
    "gdwaab or mkzlyg": "gdwaab_first",
    "gdwaab_or_mkzlyg": "gdwaab_first",
    "mkzlyg or gdwaab": "mkzlyg_first",
    "mkzlyg_or_gdwaab": "mkzlyg_first",
}

MODEL_PATTERNS = [
    (re.compile(r"gpt[\-_]?5[\.\-_]?4[\.\-_]?nano", re.I), "gpt-4.1-nano"),
    (re.compile(r"gpt[\-_]?5[\.\-_]?4(?![\d\.])", re.I),   "gpt-4.1"),
    (re.compile(r"gpt5nano", re.I),                          "gpt-4o-mini"),
    (re.compile(r"openai", re.I),                            "gpt-unknown"),
    (re.compile(r"llama[\-_\.]*3[\.\-_]*1[\.\-_]*8b", re.I), "Llama-3.1-8B"),
    (re.compile(r"meta[\-_]llama", re.I),                    "Llama-3.1-8B"),
    (re.compile(r"llama", re.I),                             "Llama"),
    (re.compile(r"gemma[\-_]?3[\-_]?12b", re.I),            "Gemma-3-12B"),
    (re.compile(r"gemma[\-_]?3[\-_]?27b", re.I),            "Gemma-3-27B"),
    (re.compile(r"gemma", re.I),                             "Gemma"),
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
    # random words — check before generic substrings
    "mxjlg":       "random",
    "hwrbx":       "random",
    # new random words
    "imphdy":      "random",
    "mkzlyg":      "random",
    "gdwaab":      "random",
}

# Maps order key -> [choice_a, choice_b]
ORDER_TO_CHOICES = {
    "chicken_first": ["chicken", "pasta"],
    "pasta_first":   ["chicken", "pasta"],
    "PS_first":      ["playstation", "xbox"],
    "XB_first":      ["playstation", "xbox"],
    "mxjlg_first":   ["mxjlg", "hwrbx"],
    "hwrbx_first":   ["mxjlg", "hwrbx"],
    # new random word pairs — only unambiguous entries
    # gdwaab pairs are resolved dynamically from the _no_<word> token in filename
    "imphdy_first":  ["imphdy", "mkzlyg"],
    "mkzlyg_first":  ["imphdy", "mkzlyg"],
}


# ── Extractors ────────────────────────────────────────────────────────────────

def extract_temperature(name: str) -> float | None:
    m = re.search(r"temp[=_](\d+\.?\d*)", name, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"instruct[_\-\s]+(\d+\.?\d*)", name, re.I)
    if m:
        v = float(m.group(1))
        if v in VALID_TEMPS or 0 <= v <= 5:
            return v
    for m in re.finditer(r"(?<=[_\s])(\d+\.\d+|\d+)(?=[_\s]|$)", name):
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
    for kw, label in sorted(QUESTION_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in lower:
            return label
    return "unknown_question"


# All known random words — only these are valid unavailable candidates.
# This prevents tokens like "no_memory" in filenames from being matched.
RANDOM_WORDS = {"imphdy", "mkzlyg", "gdwaab", "mxjlg", "hwrbx"}

def extract_unavailable(name: str) -> str | None:
    """
    Returns the word declared unavailable in the prompt, parsed from the
    '_no_<word>' token in the filename (e.g. '..._no_gdwaab...').
    Only matches known random words, so 'no_memory' etc. are never picked up.
    Returns None for plain binary runs.
    """
    for m in re.finditer(r"_no_([a-z]+)", name, re.I):
        candidate = m.group(1).lower()
        if candidate in RANDOM_WORDS:
            return candidate
    return None


def extract_active_pair(name: str) -> str | None:
    """
    For random-word filenames, extract the canonical sorted pair of the two
    ACTIVE words directly from the filename (e.g. 'gdwaab_or_mkzlyg' -> 'gdwaab_mkzlyg').
    This is more reliable than inferring the partner from the order key alone.
    Returns None for non-random files.
    """
    name_lower = name.lower()
    rw = sorted(RANDOM_WORDS)  # consistent ordering for search
    found = [w for w in rw if w in name_lower]
    if len(found) < 2:
        return None
    # Among all found random words, the active pair is the two that appear in
    # the 'word_or_word' part of the filename (before any tags like _no_, _random, etc.)
    # Strip everything from the first tag suffix onward, then re-search.
    # Suffixes that mark the end of the word-pair section:
    base = re.split(r'_no_memory|_with_memory|_no_[a-z]+|_random|_temp', name_lower)[0]
    active = [w for w in rw if w in base]
    if len(active) == 2:
        return "_".join(sorted(active))
    # Fallback: if unavailable is known, exclude it from found words
    unav = extract_unavailable(name)
    if unav:
        active = [w for w in found if w != unav]
        if len(active) == 2:
            return "_".join(sorted(active))
    # Last resort: just return all found words sorted
    if len(found) >= 2:
        return "_".join(sorted(found[:2]))
    return None


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
        unavailable=extract_unavailable(name),
        active_pair=extract_active_pair(name),
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

        # Derive known_choices from the static lookup first.
        # For random-word pairs not in the table (any gdwaab combination, or new pairs),
        # use the active_pair field extracted directly from the filename.
        known_choices = ORDER_TO_CHOICES.get(meta["order"], [])
        if not known_choices and meta.get("active_pair"):
            pair_words = meta["active_pair"].split("_")   # e.g. ["gdwaab", "mkzlyg"]
            if len(pair_words) == 2:
                order_key = meta["order"]
                pair_words_ordered = sorted(
                    pair_words,
                    key=lambda w: order_key.find(w) if w in order_key else 99
                )
                known_choices = pair_words_ordered

        # n_invalid: how many extra attempts the generation script needed beyond
        # the 100 valid responses it collected. The CSV only contains valid rows,
        # so we read this from the "attempt" column (max attempt - num valid rows).
        # Falls back to 0 if the column is absent (older CSV format).
        n_total = len(df)
        if "attempt" in df.columns:
            n_invalid = int(df["attempt"].max() - n_total)
        else:
            n_invalid = 0

        def extract_choice(text, candidates):
            from difflib import SequenceMatcher
            text = str(text).lower()
            if text in candidates:
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

        choices = sorted(c for c in df["response"].unique() if c not in ("nan", ""))

        if len(choices) < 2:
            all_candidates = known_choices or ORDER_TO_CHOICES.get(meta["order"], [])
            if len(choices) == 1 and all_candidates:
                choice_a = all_candidates[0]
                n_a = (df["response"] == choice_a).sum()
                p_a = n_a / n_total if n_total else float("nan")
                se = np.sqrt(p_a * (1 - p_a) / n_total) if n_total else float("nan")
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

        se = np.sqrt(p_a * (1 - p_a) / n_total) if n_total else float("nan")
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


# ── Invalid-count summary (terminal) ─────────────────────────────────────────

def print_invalid_summary(data: pd.DataFrame):
    """
    Print a table of invalid counts grouped by every unique data point:
    model, question, order, price, temperature.
    Only rows with at least one invalid are shown (if all are zero, says so).
    """
    cols = ["model", "question", "order", "unavailable", "price", "temperature", "n_invalid", "n_total"]
    available = [c for c in cols if c in data.columns]
    summary = (
        data[available]
        .sort_values(["model", "question", "price", "unavailable", "order", "temperature"],
                     na_position="first")
        .reset_index(drop=True)
    )

    has_any_invalid = summary["n_invalid"].gt(0).any()

    print("\n" + "=" * 70)
    print("INVALID RESPONSE COUNTS  (per graph point = temp + order + price)")
    print("=" * 70)

    if not has_any_invalid:
        print("  No invalid responses detected across all files.")
    else:
        # Compute pct for display
        summary["pct_invalid"] = (
            summary["n_invalid"] / summary["n_total"] * 100
        ).round(1)

        # Pretty-print with grouping headers
        current_group = None
        for _, row in summary.iterrows():
            group_key = (row["model"], row["question"], row.get("price"), row.get("unavailable"))
            if group_key != current_group:
                current_group = group_key
                price_str = f"  price={row['price']}" if pd.notna(row.get("price")) else ""
                unav = row.get("unavailable")
                unav_str = f"  unavailable={unav}" if pd.notna(unav) and unav else "  (plain binary)"
                print(f"\n  [{row['model']}  |  question: {row['question']}{price_str}{unav_str}]")
                print(f"  {'order':<20} {'temp':>6}  {'invalid':>8}  {'of total':>9}  {'%':>6}")
                print(f"  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*9}  {'-'*6}")

            flag = "  <--" if row["n_invalid"] > 0 else ""
            print(
                f"  {row['order']:<20} {row['temperature']:>6.2f}"
                f"  {int(row['n_invalid']):>8}  {int(row['n_total']):>9}  "
                f"{row['pct_invalid']:>5.1f}%{flag}"
            )

    print("=" * 70 + "\n")


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _draw_curves(ax, subdata, colors, invert=False):
    orders = sorted(subdata["order"].unique())
    for idx, order in enumerate(orders):
        sub = subdata[subdata["order"] == order].sort_values("temperature")
        temps = sub["temperature"].values
        probs = sub["p_choice_a"].values
        n_invalids = sub["n_invalid"].values if "n_invalid" in sub.columns else np.zeros(len(temps))
        color = colors[idx % len(colors)]
        label = order.replace("_", " ")

        probs_for_fit = probs.copy()

        if invert:
            probs = 1.0 - probs

        errors = sub["moe"].values

        ax.errorbar(temps, probs, yerr=errors,
                    marker="o", linewidth=1.8, markersize=6,
                    capsize=4, capthick=1.2, elinewidth=1.2,
                    color=color, label=label)

        # Annotate extra attempts needed above each point (only if > 0).
        # n_inv = total attempts - 100 valid responses collected.
        for x, y, n_inv in zip(temps, probs, n_invalids):
            if n_inv > 0:
                ax.annotate(
                    f"~{int(n_inv)} extra",
                    xy=(x, y),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=6.5,
                    color=color,
                    alpha=0.70,
                    fontweight="bold",
                )

        mask = (probs_for_fit > 0) & (probs_for_fit < 1)

        if mask.sum() >= 2:
            try:
                def helbing_model(T, U):
                    return 1.0 / (1.0 + np.exp(-U / T))

                popt, _ = curve_fit(
                    helbing_model, temps[mask], probs_for_fit[mask],
                    p0=[0.5], maxfev=4000,
                )
                U_fit = popt[0]
                t_fine = np.linspace(max(temps.min(), 0.01), temps.max(), 300)
                p_fit = helbing_model(t_fine, U_fit)
                if invert:
                    p_fit = 1 - p_fit
                    U_display = -U_fit
                else:
                    U_display = U_fit

                ax.plot(t_fine, p_fit, "--", color=color, linewidth=1.4,
                        alpha=0.8, label=f"{label} (U={U_display:.3f})")
            except Exception:
                pass

    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Temperature", fontsize=13)

    n = int(subdata["n_total"].iloc[0])
    n_inv_total = int(subdata["n_invalid"].sum()) if "n_invalid" in subdata.columns else 0
    base_text = f"n = {n} per run" if n_inv_total == 0 else f"n = {n} valid per run  ({n_inv_total} extra attempts total)"
    label_text = f"{base_text}\nErr. bars: 96% Conf. Interval"
    if n_inv_total > 0:
        label_text += "\n('~N extra' = discarded attempts per point)"
    ax.text(0.5, -0.25, label_text, transform=ax.transAxes,
            fontsize=9, ha="center", va="top", color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="gray", alpha=0.4))


# ── Main plotting function ────────────────────────────────────────────────────

def _subplot_title(unavailable: str | None) -> str:
    if unavailable:
        return f"Type A: {unavailable} unavailable"
    return "Type B: plain binary"


def plot_p_choice(data: pd.DataFrame, out_folder: str, invert: bool = False):
    os.makedirs(out_folder, exist_ok=True)
    colors = plt.cm.tab10.colors

    for (model, question), grp in data.groupby(["model", "question"]):

        prices = sorted(p for p in grp["price"].unique() if pd.notna(p))
        has_price = len(prices) > 0

        # For random-word questions, produce one figure per active word pair,
        # with panels inside each figure for: Type A (each unavailable word)
        # and Type B (plain binary) — so the comparison is clean and nothing
        # from different pairs bleeds into the same plot.
        if question == "random" and "unavailable" in grp.columns:

            # active_pair was extracted directly from the filename in parse_filename,
            # so we just use the column — no inference needed here.
            # Fill any missing values (non-random files) with "unknown_pair".
            grp = grp.copy()
            if "active_pair" not in grp.columns:
                grp["active_pair"] = "unknown_pair"
            grp["active_pair"] = grp["active_pair"].fillna("unknown_pair")

            # Always include both priced groups and the no-price group separately.
            # If we only used "has_price" to decide, unpriced files would be
            # dropped whenever any priced file exists in the same folder.
            price_groups = []
            unpriced = grp[grp["price"].isna()]
            if not unpriced.empty:
                price_groups.append((None, unpriced))
            for p in prices:
                price_groups.append((p, grp[grp["price"] == p]))

            for price, price_data in price_groups:
                for pair_key, pair_grp in price_data.groupby("active_pair"):
                    if pair_grp.empty:
                        continue

                    # choice_a must be one of the known random words, not a
                    # garbage LLM response that slipped through. Pick the known
                    # word from the pair that appears most as choice_a.
                    pair_words = [w for w in pair_key.split("_") if w in RANDOM_WORDS]
                    valid_choice_a = pair_grp[pair_grp["choice_a"].isin(RANDOM_WORDS)]
                    if not valid_choice_a.empty:
                        choice_a_label = valid_choice_a["choice_a"].mode().iloc[0]
                    elif pair_words:
                        choice_a_label = pair_words[0]
                    else:
                        choice_a_label = pair_grp["choice_a"].mode().iloc[0]

                    choice_b_label = next(
                        (w for w in pair_words if w != choice_a_label),
                        "other"
                    )
                    ylabel = f"P({choice_b_label})" if invert else f"P({choice_a_label})"

                    # Panels: Type A entries (one per unavailable word) + Type B last
                    unav_values = (
                        sorted(u for u in pair_grp["unavailable"].unique()
                               if pd.notna(u) and u)
                        + [None]  # Type B: plain binary
                    )
                    # Drop None panel if no plain-binary data exists for this pair
                    has_plain = pair_grp["unavailable"].isna().any() or                                 (pair_grp["unavailable"] == "").any()
                    if not has_plain:
                        unav_values = [u for u in unav_values if u is not None]

                    n_panels = len(unav_values)
                    if n_panels == 0:
                        continue

                    max_cols = min(n_panels, 3)
                    rows = math.ceil(n_panels / max_cols)

                    fig, axes = plt.subplots(
                        rows, max_cols,
                        figsize=(5 * max_cols, 5.5 * rows),
                        sharey=True,
                        squeeze=False,
                    )
                    axes_flat = axes.flatten()

                    price_str = f"  |  Price: {price}" if price else ""
                    fig.suptitle(
                        f"Model: {model}  |  Pair: {pair_key}{price_str}",
                        fontsize=14, y=1.02,
                    )

                    for idx, unav in enumerate(unav_values):
                        ax = axes_flat[idx]
                        if unav is None:
                            sub = pair_grp[
                                pair_grp["unavailable"].isna() |
                                (pair_grp["unavailable"] == "")
                            ]
                        else:
                            sub = pair_grp[pair_grp["unavailable"] == unav]

                        if sub.empty:
                            ax.set_visible(False)
                            continue

                        _draw_curves(ax, sub, colors, invert=invert)
                        ax.set_title(_subplot_title(unav), fontsize=12)
                        ax.legend(fontsize=9, loc="best")

                        if idx % max_cols == 0:
                            ax.set_ylabel(ylabel, fontsize=13)

                    for j in range(n_panels, len(axes_flat)):
                        fig.delaxes(axes_flat[j])

                    fig.tight_layout()

                    price_tag = f"_price_{price}".replace(" ", "_") if price else ""
                    fname = (
                        f"p_{choice_a_label}_{model}_{pair_key}{price_tag}.png"
                        .replace(" ", "_")
                    )
                    out_path = os.path.join(out_folder, fname)
                    fig.savefig(out_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  Saved: {out_path}")

        else:
            # Original layout for non-random questions
            priced_data   = grp[grp["price"].notna()]
            unpriced_data = grp[grp["price"].isna()]

            if has_price:
                n_prices = len(prices)
                max_cols = 3
                cols = min(n_prices, max_cols)
                rows = math.ceil(n_prices / max_cols)

                fig, axes = plt.subplots(
                    rows, cols,
                    figsize=(5 * cols, 5.5 * rows),
                    sharey=True,
                    squeeze=False,
                )
                axes_flat = axes.flatten()

                fig.suptitle(
                    f"Model: {model}  |  Question: {question}",
                    fontsize=14, y=1.02,
                )

                for idx, price in enumerate(prices):
                    ax = axes_flat[idx]
                    sub = priced_data[priced_data["price"] == price]
                    _draw_curves(ax, sub, colors, invert=invert)
                    ax.set_title(f"Prices: {price}", fontsize=13)
                    ax.legend(fontsize=10, loc="best")

                    if idx % max_cols == 0:
                        ax.set_ylabel(ylabel, fontsize=13)

                for j in range(n_prices, len(axes_flat)):
                    fig.delaxes(axes_flat[j])

                fig.tight_layout()

            else:
                fig, ax = plt.subplots(figsize=(7, 5.5))
                _draw_curves(ax, unpriced_data, colors, invert=invert)
                ax.set_title(f"Model: {model}  |  Question: {question}", fontsize=14)
                ax.set_ylabel(ylabel, fontsize=13)
                ax.legend(fontsize=11, loc="best")
                fig.tight_layout()

            fname = f"p_{choice_a_label}_{model}_{question}.png".replace(" ", "_")
            out_path = os.path.join(out_folder, fname)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot P(choice) vs Temperature")
    parser.add_argument("--folder", default=".", help="Folder with CSV files")
    parser.add_argument("--out", default="figures", help="Output folder for plots")
    parser.add_argument("--invert", action="store_true",
                        help="Plot P(second choice) instead of P(first choice)")
    args = parser.parse_args()

    print(f"\nLoading files from: {args.folder}")
    data = load_files(args.folder)

    print(f"\nSummary table:")
    print(
        data[["test_id", "model", "question", "order", "unavailable", "temperature",
              "price", "choice_a", "p_choice_a", "U_empirical", "n_total", "n_invalid"]]
        .sort_values(["model", "question", "price", "order", "unavailable", "temperature"])
        .to_string(index=False)
    )

    print_invalid_summary(data)

    print(f"\nGenerating plots -> {args.out}/")
    plot_p_choice(data, out_folder=args.out, invert=args.invert)
    print("\nDone.")


if __name__ == "__main__":
    main()