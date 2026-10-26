import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ── File Paths ────────────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
OPENAI_FILE = os.path.join(RESULTS_DIR, "logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv")
HF_FILE = os.path.join(RESULTS_DIR, "logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct.csv")

# Outputs
OUTPUT_IMG_NOPRICE = os.path.join(RESULTS_DIR, "Ladder_Prompt_Heatmaps_OrderSplit_NoPrice.png")
OUTPUT_IMG_10DOLLAR = os.path.join(RESULTS_DIR, "Ladder_Prompt_Heatmaps_OrderSplit_10Dollar.png")

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

    # Filter for baseline temperature T=1.0
    return df_all[df_all["temperature"] == 1.0]


def build_order_comparison_matrix(df, model_name, target_item="red t-shirt", other_item="green t-shirt", price_filter="None"):
    sub = df[df["model"] == model_name]
    
    # Apply Price Filtering
    if price_filter == "None":
        sub = sub[sub["p1"].isna() & sub["p2"].isna()]
    else:
        # For tests where both items cost the specified amount (e.g., $10)
        target_price = float(price_filter)
        sub = sub[(sub["p1"] == target_price) & (sub["p2"] == target_price)]

    records = []

    # Column headers using exact item text
    col_a_first = f"{target_item.title()} First"
    col_avg = "Average"
    col_b_first = f"{other_item.title()} First"

    for style in STYLE_ORDER:
        style_rows = sub[sub["prompt_style"] == style]
        if style_rows.empty:
            continue

        # Condition 1: Target item presented first
        c1 = style_rows[
            (style_rows["word1"].str.lower() == target_item.lower()) & 
            (style_rows["word2"].str.lower() == other_item.lower())
        ]
        p_target_first = c1["rel_prob_word1"].values[0] if not c1.empty else None

        # Condition 2: Other item presented first
        c2 = style_rows[
            (style_rows["word1"].str.lower() == other_item.lower()) & 
            (style_rows["word2"].str.lower() == target_item.lower())
        ]
        p_target_second = c2["rel_prob_word2"].values[0] if not c2.empty else None

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

    # Explicit columns=... matters when records is empty (e.g. Llama has no
    # "no price" condition data — RUN_NO_PRICE=False in LLAMA-25.8.py):
    # pd.DataFrame([]) with no columns argument produces a frame with NO
    # columns at all, so .set_index("prompt_style") throws KeyError instead
    # of falling through to the "No matching data" handling below that
    # already exists for exactly this case.
    matrix = pd.DataFrame(records, columns=["prompt_style", col_a_first, col_avg, col_b_first])
    return matrix.set_index("prompt_style")


def generate_heatmaps(df, output_path, target_item="red t-shirt", other_item="green t-shirt", price_filter="None", title_suffix=""):
    models = ordered_models(df)

    # Create Side-by-Side Subplots: increased height to 12 for taller rows
    fig, axes = plt.subplots(
        1, len(models), 
        figsize=(9.5 * len(models), 12), 
        squeeze=False,
        sharey=True 
    )
    sns.set_theme(style="white")

    for idx, model_name in enumerate(models):
        ax = axes[0, idx]
        matrix = build_order_comparison_matrix(df, model_name, target_item, other_item, price_filter)

        if matrix.empty:
            ax.set_title(f"{model_name}\n(No matching data)", fontsize=22)
            ax.axis("off")
            continue

        # Heatmap: RdYlGn_r -> 1.0 Red, 0.5 Neutral/Yellow, 0.0 Green
        hm = sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
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
        cbar.set_label(f"P({target_item.title()})", size=16, weight="bold")

        ax.set_title(
            f"{model_name}",
            fontsize=22,  # Bigger subplot title
            fontweight="bold",
            pad=18
        )
        
        # Ensure Y-axis is big and only on the left
        if idx == 0:
            ax.set_ylabel("Ladder Prompt Style", fontsize=18, fontweight="bold")
            ax.tick_params(axis="y", labelsize=16, rotation=0)
        else:
            ax.set_ylabel("")

        ax.set_xlabel(f"{target_item.title()} vs {other_item.title()}", fontsize=18, fontweight="bold", labelpad=15)
        ax.tick_params(axis="x", rotation=0, labelsize=16)

    # Main Title
    fig.suptitle(
        f"Order Probability Split: {target_item.title()} vs {other_item.title()} {title_suffix}", 
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
    
    # 1. Generate Heatmaps for Baseline (No Prices)
    generate_heatmaps(
        df_all, 
        OUTPUT_IMG_NOPRICE, 
        target_item="red t-shirt", 
        other_item="green t-shirt",
        price_filter="None",
        title_suffix="(No Prices)"
    )
    
    # 2. Generate Heatmaps for Equal $10 Prices
    generate_heatmaps(
        df_all, 
        OUTPUT_IMG_10DOLLAR, 
        target_item="red t-shirt", 
        other_item="green t-shirt",
        price_filter=10.0,
        title_suffix="(Fixed \\$10 Prices)"
    )