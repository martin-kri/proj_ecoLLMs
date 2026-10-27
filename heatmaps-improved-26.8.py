import ast
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── File Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
OPENAI_FILE = os.path.join(RESULTS_DIR, "logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv")
HF_FILE = os.path.join(RESULTS_DIR, "logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct.csv")

# Outputs: one image per color pair and price condition, e.g.
# Ladder_Prompt_Heatmaps_OrderSplit_NoPrice_red_vs_green.png
# Ladder_Prompt_Heatmaps_OrderSplit_10Dollar_red_vs_green.png
OUTPUT_PREFIX = os.path.join(RESULTS_DIR, "Ladder_Prompt_Heatmaps_OrderSplit")

# Which color pairs to draw, each as (color whose probability is shown, other
# color), e.g. [("red", "green")]. None = every pair found in the CSVs, in the
# order they're listed in COLOR_PAIRS in OpenAI-25.8.py (pairs not listed
# there are added in alphabetical order).
COLOR_PAIRS = None
EXPERIMENT_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenAI-25.8.py")

# Which complexity ladder set(s) to include: 1=Set A, 2=Set B, 3=Set C.
# Currently just Set C, matching what the gpt-5.4 family actually has data
# for right now (the Llama CSV still has all three sets from the earlier
# run) — restricting here keeps the 4-model comparison apples-to-apples
# instead of showing 3 models with gaps where set 1/2 data doesn't exist.
# Add 1 and/or 2 back in later once more data exists for them.
ACTIVE_PROMPT_SETS = [3]
_SET_LETTER = {1: "A", 2: "B", 3: "C"}

# Canonical order of the ladder styles within the active set(s) (A/B/C
# naming, matching the set1->setA etc. rename load_data() does below).
STYLE_ORDER = [
    f"set{_SET_LETTER[s]}_r{r}" for s in ACTIVE_PROMPT_SETS for r in (1, 2, 3, 4)
]

# Fixed panel order, per request — NOT alphabetical (matches
# one-by-one_order-bias-26.8.py's MODEL_ORDER/ordered_models() for
# consistency across both plotting scripts). Any model not in this list
# (e.g. a future 6th model) is appended alphabetically at the end rather
# than silently vanishing from the chart.
MODEL_ORDER = ["Llama-3.1-8B-Instruct", "gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.4-mini", "gpt-5.4"]


def ordered_models(df):
    present = set(df["model"].unique())
    known = [m for m in MODEL_ORDER if m in present]
    unknown = sorted(present - set(MODEL_ORDER))
    return known + unknown

def _parse_prices(pp):
    """Safely extracts price values from a string like '(10.0, 10.0)'."""
    pp = str(pp)
    if pp in ("nan", "None"):
        return (None, None)
    try:
        p1, p2 = [float(x) for x in pp.strip("() ").split(",")]
        return (p1, p2)
    except Exception:
        return (None, None)

def load_data():
    dfs = []

    # 1. Load OpenAI Data
    if os.path.exists(OPENAI_FILE):
        df_openai = pd.read_csv(OPENAI_FILE)
        if "both_in_top_n" in df_openai.columns:
            df_openai = df_openai[df_openai["both_in_top_n"] == True]
        dfs.append(df_openai)

    # 2. Load HuggingFace Data
    if os.path.exists(HF_FILE):
        df_hf = pd.read_csv(HF_FILE)
        if "model" not in df_hf.columns:
            df_hf["model"] = "Llama-3.1-8B-Instruct"
        dfs.append(df_hf)

    if not dfs:
        raise FileNotFoundError("No result CSV files found in ./results")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["price_pair"] = df_all["price_pair"].fillna("None").astype(str)
    
    # Safely convert old 'set1/2/3' names to 'setA/B/C' in case the CSV hasn't been updated
    df_all["prompt_style"] = df_all["prompt_style"].str.replace("set1", "setA")
    df_all["prompt_style"] = df_all["prompt_style"].str.replace("set2", "setB")
    df_all["prompt_style"] = df_all["prompt_style"].str.replace("set3", "setC")
    
    # Parse prices for accurate filtering
    parsed = df_all["price_pair"].apply(_parse_prices)
    df_all["p1"] = parsed.apply(lambda t: t[0])
    df_all["p2"] = parsed.apply(lambda t: t[1])

    df_all["color1"] = df_all["color1"].str.lower()
    df_all["color2"] = df_all["color2"].str.lower()

    # Filter for baseline temperature T=1.0
    return df_all[df_all["temperature"] == 1.0]


def configured_color_pairs():
    """COLOR_PAIRS as written in OpenAI-25.8.py, read as text rather than
    imported: importing that script would create an API client. Returns []
    if the script or the setting can't be read."""
    try:
        with open(EXPERIMENT_SCRIPT, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "COLOR_PAIRS" for t in node.targets):
            try:
                return [(c1.lower(), c2.lower()) for c1, c2 in ast.literal_eval(node.value)]
            except (ValueError, TypeError):
                return []
    return []


def color_pairs_in_data(df):
    """Every color pair present in the data, as (target, other)."""
    found = {frozenset(p) for p in zip(df["color1"], df["color2"]) if p[0] != p[1]}
    pairs = []
    for c1, c2 in configured_color_pairs():
        if frozenset((c1, c2)) in found:
            pairs.append((c1, c2))
            found.discard(frozenset((c1, c2)))
    return pairs + sorted(tuple(sorted(p)) for p in found)


def item_label(df, color):
    """The full item text for a color as it appears in the prompts, e.g. 'Red T-Shirt'."""
    words = df.loc[df["color1"] == color, "word1"]
    return words.iloc[0].title() if not words.empty else color.title()


def build_order_comparison_matrix(df, model_name, target_color, other_color, price_filter=None):
    """price_filter: None for the no-price condition, or a number for the
    same-price condition at that price (both items cost that amount)."""
    sub = df[df["model"] == model_name]
    # Only this pair's rows. Otherwise a model that has rows for another pair
    # gets all-empty rows here, which seaborn can't draw.
    pair = {target_color, other_color}
    sub = sub[sub["color1"].isin(pair) & sub["color2"].isin(pair)]

    # Apply Price Filtering
    if price_filter is None:
        sub = sub[sub["p1"].isna() & sub["p2"].isna()]
    else:
        target_price = float(price_filter)
        sub = sub[(sub["p1"] == target_price) & (sub["p2"] == target_price)]

    records = []

    # Column headers using exact item text
    col_a_first = f"{item_label(df, target_color)} First"
    col_avg = "Average"
    col_b_first = f"{item_label(df, other_color)} First"

    for style in STYLE_ORDER:
        style_rows = sub[sub["prompt_style"] == style]
        if style_rows.empty:
            continue

        # Condition 1: Target item presented first. Mean, not the first row,
        # in case a condition has several rows (e.g. several reasoning efforts).
        c1 = style_rows[(style_rows["color1"] == target_color) & (style_rows["color2"] == other_color)]
        p_target_first = c1["rel_prob_word1"].mean() if not c1.empty else None

        # Condition 2: Other item presented first
        c2 = style_rows[(style_rows["color1"] == other_color) & (style_rows["color2"] == target_color)]
        p_target_second = c2["rel_prob_word2"].mean() if not c2.empty else None

        # Condition 3: Average
        if p_target_first is not None and p_target_second is not None:
            p_avg = (p_target_first + p_target_second) / 2.0
        else:
            p_avg = p_target_first if p_target_first is not None else p_target_second

        records.append({
            "prompt_style": style,
            col_a_first: p_target_first,
            col_avg: p_avg,
            col_b_first: p_target_second
        })

    # Explicit columns=... matters when records is empty (e.g. a model with no
    # rows for this price condition):
    # pd.DataFrame([]) with no columns argument produces a frame with NO
    # columns at all, so .set_index("prompt_style") throws KeyError instead
    # of falling through to the "No matching data" handling below that
    # already exists for exactly this case.
    matrix = pd.DataFrame(records, columns=["prompt_style", col_a_first, col_avg, col_b_first])
    # float: a missing order is None, which would otherwise make the column
    # non-numeric and crash sns.heatmap. As NaN it's just drawn as a blank cell.
    return matrix.set_index("prompt_style").astype(float)


def generate_heatmaps(df, output_path, target_color, other_color, price_filter=None, title_suffix=""):
    models = ordered_models(df)
    target_item = item_label(df, target_color)
    other_item = item_label(df, other_color)

    # Create Side-by-Side Subplots: increased height to 12 for taller rows
    fig, axes = plt.subplots(
        1, len(models), 
        figsize=(9.5 * len(models), 12), 
        squeeze=False,
        sharey=True 
    )
    sns.set_theme(style="white")
    y_labeled = False

    for idx, model_name in enumerate(models):
        ax = axes[0, idx]
        matrix = build_order_comparison_matrix(df, model_name, target_color, other_color, price_filter)

        if matrix.empty:
            ax.set_title(f"{model_name}\n(No matching data)", fontsize=22)
            ax.axis("off")
            continue

        # Diverging colormap with a light midpoint at 0.5. Its colors are
        # deliberately unrelated to the item colors, so they never suggest
        # "red means the red item" when another pair is plotted.
        hm = sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="PuOr_r",
            center=0.5,
            vmin=0.0,
            vmax=1.0,
            linewidths=1.5,  # Slightly thicker lines to match the larger scale
            linecolor="white",
            annot_kws={"size": 18, "weight": "bold"},  # Bigger text inside cells
            ax=ax
        )
        
        # Increase text size for the colorbar
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(f"P({target_item})", size=16, weight="bold")

        ax.set_title(
            f"{model_name}",
            fontsize=22,  # Bigger subplot title
            fontweight="bold",
            pad=18
        )
        
        # Y-axis labels only on the first panel that has data (not idx == 0:
        # if the first model has no data for this pair, its panel is hidden
        # and the labels would disappear with it)
        if not y_labeled:
            ax.set_ylabel("Ladder Prompt Style", fontsize=18, fontweight="bold")
            ax.tick_params(axis="y", labelsize=16, rotation=0, labelleft=True)
            y_labeled = True
        else:
            ax.set_ylabel("")

        ax.set_xlabel(f"{target_item} vs {other_item}", fontsize=18, fontweight="bold", labelpad=15)
        ax.tick_params(axis="x", rotation=0, labelsize=16)

    # Main Title
    fig.suptitle(
        f"Order Probability Split: {target_item} vs {other_item} {title_suffix}",
        fontsize=28,  # Bigger main title
        fontweight="bold",
        y=1.04
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    df_all = load_data()

    pairs = COLOR_PAIRS or color_pairs_in_data(df_all)
    # Every equal-price level in the data (SAME_PRICE_LEVELS in the experiment scripts)
    same_price_levels = sorted(df_all.loc[df_all["price_type"] == "same", "price_base"].dropna().unique())
    print(f"Color pairs: {pairs}  |  same-price levels: {same_price_levels}")

    for target_color, other_color in pairs:
        tag = f"{target_color}_vs_{other_color}".replace(" ", "-")

        # 1. Baseline (no prices)
        generate_heatmaps(
            df_all,
            f"{OUTPUT_PREFIX}_NoPrice_{tag}.png",
            target_color,
            other_color,
            price_filter=None,
            title_suffix="(No Prices)"
        )

        # 2. One image per equal-price level (both items cost the same)
        for level in same_price_levels:
            generate_heatmaps(
                df_all,
                f"{OUTPUT_PREFIX}_{level:g}Dollar_{tag}.png",
                target_color,
                other_color,
                price_filter=level,
                title_suffix=f"(Same Price: \\${level:g})"
            )