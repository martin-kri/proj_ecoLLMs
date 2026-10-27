# proj_ecoLLMs: The Decision-Making Behavior of Large Language Models

Code and data from my internship at the CHuB Lab, Fondazione Bruno Kessler (FBK), 2026. This repo contains the final experiment: order bias and price sensitivity in a simple two-item choice.

The experiment asks an LLM to choose between two otherwise identical items that differ only in color, for example:

> *You are being asked to choose between two t-shirts, one is red (for $800) and the other one is green (for $810). Please reply with only one word.*

The script doesn't sample a text answer. It reads the model's **first-token log-probabilities** for the two candidate words and turns them into a choice probability. Each prompt is asked in both orders (red first, then green first). That shows whether the model prefers the cheaper item or just the item listed first (**order bias**).

This README explains how to reproduce the results in [`results/`](results/), step by step.

---

## What's in the repo

| File | What it does | Step |
|---|---|---|
| `OpenAI-25.8.py` | Queries the OpenAI models (`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.6-luna`) and saves the T=1 logprobs | 2a |
| `LLAMA-25.8.py` | Runs `meta-llama/Llama-3.1-8B-Instruct` locally on a GPU and saves the T=1 logits | 2b |
| `one-by-one_order-bias-26.8.py` | Main order-bias figures (line, bar and per-price grids) | 3 |
| `heatmaps-improved-26.8.py` | Order-split heatmaps for the "no price" and "same $10 price" conditions | 4 |
| `temperature_sweep.py` | Rebuilds other temperatures (T=4, T=5) from the T=1 data, **with no new API calls**, and plots them | 5 |
| `model_size_vs_order_bias-27.8.py` | Order bias vs. model size, at one temperature | 6 |
| `model_size_vs_order_bias_grid-27.8.py` | The same as one 2×2 slide image (T=1/T=5 × fixed/relative gap) | 7 |
| `csv_cleaner.py` | *Not needed to reproduce.* A one-time script I used to migrate an older nano/mini results file to the current format | n/a |
| `results/` | My data and figures from the final runs | n/a |
| `earlier-experiments/` | The text-sampling experiments from March–July: chicken vs. pasta, PlayStation vs. Xbox, random strings / IIA, colors, t-shirts, hot vs. cold. They have their own [README](earlier-experiments/README.md). | separate |

The scripts load each other by filename with `importlib`, so **keep them all in the same folder and don't rename them**.

---

## Order of operations

```
.env ──► 2a OpenAI-25.8.py ─┐
         2b LLAMA-25.8.py ──┤  (both write results/*.csv)
                            ▼
         3  one-by-one_order-bias-26.8.py      (T=1 figures)
         4  heatmaps-improved-26.8.py          (heatmaps)
         5  temperature_sweep.py               (T=4, T=5 data + figures)
         6  model_size_vs_order_bias-27.8.py   (run with T=1, then T=5)
         7  model_size_vs_order_bias_grid-27.8.py
```

Steps 3–7 only read the CSVs from step 2. **If you just want the figures, skip step 2**: my CSVs are already in `results/`.

Run every script **from the repo root**. They all read and write `./results`.

---

## Step 0: Install

**Python 3.10 specifically.** The Llama data was generated on a GPU server. The pinned CUDA 12.4 build of `torch` and the rest of the stack in `requirements.txt` were set up against Python 3.10; later Python versions haven't been tested for the Llama script. This is the same setup as my [CSS project](https://github.com/martin-kri/CSS-project-Robustness-Audit), which ran the same model.

```bash
python3.10 -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -r requirements.txt
```

- `torch` / `transformers` are only needed for `LLAMA-25.8.py`. The OpenAI script and all the plotting scripts (steps 2a and 3–7) don't need a GPU.

## Step 1: Create the `.env` file

Copy the template and fill in your own keys:

```bash
cp .env.example .env
```

```
HF_TOKEN=...          # Hugging Face token, needed for LLAMA-25.8.py
OPENAI_API_KEY=...    # needed for OpenAI-25.8.py
OPENAI_ORG_ID=...     # optional, leave empty if you don't use one
```

- **Hugging Face:** Llama 3.1 is a gated model. Accept the license on the [model page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) with the same account the token belongs to.
- `.env` is in `.gitignore`. Never commit it.

## Step 2: Generate the data

> **Checkpointing:** while running, both scripts save their progress to `results/*_checkpoint.csv` (every 50 results for OpenAI, every 20 for Llama). On start, they read that file and skip every prompt that's already in it, so a crashed run continues where it stopped. To start over, delete the checkpoint file.
>
> My checkpoint files are deliberately **not** in this repo: they contain every prompt already, so a run would skip everything and just return my data.

### 2a: OpenAI models

```bash
python OpenAI-25.8.py
```

- 4 models × 1,168 prompts = **4,672 API calls**. My full run cost about **$0.27** in total.
- Settings used for my results (the defaults in the script): `ACTIVE_PROMPT_SETS = {3}` (Set C), `REASONING_EFFORTS = ["none"]`, `MAX_COMPLETION_TOKENS = 16`, `temperature=1.0`, `top_p=1.0`, `seed=42`, `PILOT_MODE = False`.
- To try it cheaply first, set `PILOT_MODE = True`. It runs a handful of calls per model so you can check that the API accepts `logprobs` / `reasoning_effort`.
- Output:
  - `results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder.csv` (master file)
  - `results/logprobs_set3_r{1..4}_OpenAI_t-shirt_Ladder.csv` (one file per prompt rung)

### 2b: Llama-3.1-8B-Instruct (local GPU)

```bash
python LLAMA-25.8.py
```

- 1,168 forward passes, one at a time, on a GPU server. In float16 the model needs roughly **16 GB of GPU memory**.
- **Weights revision.** Every model on Hugging Face is stored as a git repository, and each update to its files (weights, tokenizer, chat template) gets a new commit ID. `from_pretrained` downloads the newest version on the day you run it. The version that produced my results was `0e9e39f249a16976918f6564b8830bc894c89659`.
  - The script records this in the `model_revision` column. It's filled in for 1,160 of the 1,168 rows. The 8 "no price" rows are blank because they were copied in from an earlier run.
  - To load exactly that version, add `revision="0e9e39f249a16976918f6564b8830bc894c89659"` to both `from_pretrained(...)` calls in `LLAMA-25.8.py`.
- Output:
  - `results/logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct.csv`
  - `results/logprobs_set3_r{1..4}_Llama-3.1-8B-Instruct.csv`

**What the grid contains (same in both scripts).** Each script runs these conditions × 2 color orders × 4 prompt rungs (Set C):

| Condition | Values |
|---|---|
| No price | — |
| Same price | $10 vs $10 |
| Fixed gap | bases $10, 40, 100, 180, 500, 700, 750, 800, 850, 900, 1000, 1200 × gaps +$10 / +$20 / +$50, each pair in both orders |
| Relative gap | same 12 bases × +10% / +20% / +50%, each pair in both orders |

That gives 1,168 rows per model.

## Step 3: Order-bias figures (T=1)

```bash
python one-by-one_order-bias-26.8.py
```

Reads both master CSVs and writes to `results/`:

- `Slide1_Overview_AllGaps_Set3.png`, `Slide_Overview_GroupedBar_AllGaps_Set3.png`, `Slide_Overview_AverageBar_AllGaps_Set3.png`: all gap sizes in one image
- `Slide1_Models_Overview_{FixedGap,PctGap}{10,20,50}.png`, `Slide_Overview_GroupedBar_*.png`, `Slide_Overview_AverageBar_*.png`: one image per gap condition
- `Slide_Grid_Page{1..4}_Prices_*_{gap}.png`: per-price detail grids, 3 price bases per page

## Step 4: Heatmaps

```bash
python heatmaps-improved-26.8.py
```

Writes `Ladder_Prompt_Heatmaps_OrderSplit_NoPrice.png` and `Ladder_Prompt_Heatmaps_OrderSplit_10Dollar.png`.

A few cells come out blank. That's expected: in those rows the model answered with something like "Either", and one of the two colors fell outside the top-5 logprobs the OpenAI API returns. Those rows are dropped (`both_in_top_n == False`).

## Step 5: Other temperatures (no API calls)

```bash
python temperature_sweep.py
```

At T=1 the logprob difference equals the raw logit difference, so the choice probability at any temperature $T$ can be computed from the saved scores:

$$
P(\text{option}_1) = \frac{e^{s_1/T}}{e^{s_1/T} + e^{s_2/T}}, \qquad P(\text{option}_2) = 1 - P(\text{option}_1)
$$

where $s_1$ and $s_2$ are the scores of the two words (`score_word1`, `score_word2` in the CSV). As $T \to 0$ the model always picks the higher-scoring word; as $T \to \infty$ both probabilities approach 0.5.

The script does this for every temperature in `TEMPERATURES_TO_ADD` (set to `[4.0, 5.0]`) and writes:

- `results/logprobs_ALL_PROMPTS_Llama-3.1-8B-Instruct_temps.csv`
- `results/logprobs_ALL_PROMPTS_OpenAI_t-shirt_Ladder_temps.csv`
- `Slide1_Overview_AllGaps_Set3_T4p0.png`, `..._T5p0.png`, and the matching `Slide_Overview_GroupedBar_AllGaps_Set3_T*.png`

## Step 6: Model size vs. order bias

This script runs at one temperature, set by `TEMPERATURE` near the top. Run it twice to get both sets of outputs:

1. Set `TEMPERATURE = 1.0`, then run:

   ```bash
   python model_size_vs_order_bias-27.8.py
   ```

   It writes `ModelSize_vs_OrderBias_FixedGap.png`, `ModelSize_vs_OrderBias_PctGap.png` and `model_size_vs_order_bias_summary.csv`.

2. Set `TEMPERATURE = 5.0` and run it again. It writes the same three files with a `_T5p0` suffix.

Notes:
- **Model sizes are estimates.** Only Llama's 8B is officially published; OpenAI doesn't disclose parameter counts. The values in `MODEL_SIZES_B` are single points picked from informal community ranges ([arXiv 2604.24827](https://arxiv.org/abs/2604.24827v2), [cbowdon's blog](https://cbowdon.github.io/posts/gpt-params/), Reddit discussions):

  | Model | Range | Value used |
  |---|---|---|
  | gpt-5.4-nano, gpt-5.6-luna | ~8–18B | 10 |
  | gpt-5.4-mini | ~85–150B | 100 |
  | gpt-5.4 | ~300B–3T | 1800 |
- **Whiskers show the min–max range** across price/gap conditions. They are not confidence intervals.
- Running at T≠1 also rewrites the `*_temps.csv` files with just that temperature. This is harmless, and step 5 regenerates them anyway.

## Step 7: Slide grid (2×2)

```bash
python model_size_vs_order_bias_grid-27.8.py
```

Writes `ModelSize_vs_OrderBias_Grid.png` and `model_size_vs_order_bias_grid_summary.csv`: T=1 and T=5 × fixed and relative gap, on a log scale.

---

## What's in the CSVs

Each row is one prompt sent once, at T=1. Most columns describe the condition and are self-explanatory (`word1`/`word2` in the order they appear in the prompt, `price_pair`, `price_type`, `price_base`, `price_gap_or_pct`, `prompt_style` = set + rung, `prompt_text`). The rest need a bit of LLM background:

| Column | Meaning |
|---|---|
| `score_word1`, `score_word2` | How strongly the model "wants" to answer with each word as its **first token**. For Llama these are raw **logits** (unnormalized scores over the whole vocabulary). For OpenAI they are **logprobs** (the log of the token's probability) because the API doesn't expose logits. The two aren't comparable in absolute terms, but their *difference* means the same thing, and that's all the choice probability uses. Several spellings count as the same word ("red", " Red", "RED", …) and the highest-scoring one is kept. |
| `rel_prob_word1`, `rel_prob_word2` | The choice probability between just the two words (the formula in step 5). They always sum to 1. **This is the main outcome.** |
| `vocab_prob_word1`, `vocab_prob_word2` | The probability of each word among *all* possible first tokens, so it isn't renormalized to the two options. |
| `target_coverage` | `vocab_prob_word1 + vocab_prob_word2`: how much of the model's first-token probability falls on the two options at all. Low coverage means the model mostly wanted to start with something else ("Either", "Neither", "I", …), so `rel_prob` describes a small slice of its behavior. |
| `top5_tokens`, `top5_probs` | The five most likely first tokens and their probabilities, useful for seeing what the model wanted to say instead. |
| `both_in_top_n` (OpenAI) | Whether both words appeared among the top tokens the API returned. The API only returns the top 5, so a word outside them has no score. Only 3 of 4,672 rows are `False`, and those are the blank heatmap cells. |
| `top_n_requested` (OpenAI) | How many top tokens were requested (5, the API's cap for these models). |
| `sampled_token`, `full_answer` (OpenAI) | The token the API actually generated first, and the whole short answer. |
| `reasoning_effort` (OpenAI) | How much hidden "thinking" the model was allowed before answering. All rows use `none`. |
| `prompt_tokens`, `completion_tokens`, `reasoning_tokens`, `total_tokens` (OpenAI) | Token usage reported by the API, which is what you pay for. `reasoning_tokens` is 0 everywhere, confirming `none` really means no hidden reasoning. Blank for 351 older rows (176 gpt-5.4-nano, 175 gpt-5.4-mini) that were run before usage was recorded. |
| `max_completion_tokens` (OpenAI) | The answer-length cap, filled in only for those 351 older rows (5). All other rows used the script's current cap of 16. |
| `response_model` (OpenAI) | The exact model snapshot that answered, e.g. `gpt-5.4-2026-03-05`. A name like `gpt-5.4` can point to a newer snapshot later. |
| `model_revision` (Llama) | The exact version of the model files used (see step 2b). |

The `*_temps.csv` files have the same columns, with extra rows where `temperature` and `rel_prob_*` are recomputed for T=4 and T=5.

## Changing the experiment

All the settings are constants at the top of each script. Change them in **both** `OpenAI-25.8.py` and `LLAMA-25.8.py` if you want the models to stay comparable, and rerun from step 2. The plotting scripts read whatever is in the CSVs, except where noted.

**What the models are asked about**

- **`COLOR_PAIRS`**: which two colors the model chooses between. Add pairs such as `("red", "blue")`; both orders are generated automatically. This tests whether the order bias depends on the specific colors. The order-bias plots average over all color pairs in the CSV, so plot one pair at a time if you want them separately. The heatmaps are hardcoded to red vs. green t-shirts.
- **`ITEM_NOUN`**: the item that has the color, currently `"t-shirt"`. Something like `"mug"` or `"car"` works directly. The prompts are written for colored objects ("one is in the color red"), so a non-color choice such as chicken vs. pasta needs new prompt templates in `build_prompts()`.
- **`CURRENCY`**: `"USD"` writes "$10", `"EUR"` writes "10€". It tests whether the currency format changes price sensitivity.
- **`ACTIVE_PROMPT_SETS`**: which wording ladder to use. `{3}` (Set C, longer wording without new information) is what the results use. `{1}` (Set A) adds stricter output instructions and `{2}` (Set B) adds a story ("you are on a long flight…"). Set it in the plotting scripts too. Rung 1 is the same text in all three sets, so running several sets repeats those prompts.

**Which prices**

- **`RUN_NO_PRICE`, `RUN_SAME_PRICE`, `RUN_FIXED_GAP`, `RUN_PCT_GAP`**: turn each kind of price condition on or off. No price and same price show the model's baseline preference for a color, so any order effect there isn't about price. The two gap types show whether the cheaper item wins and how much the order still matters.
- **`SAME_PRICE_LEVELS`**: the equal prices to test, e.g. `[10, 500]` to see whether high prices alone change behavior.
- **`FIXED_GAP_BASES`, `FIXED_GAPS`**: base prices and dollar gaps, e.g. $800 vs. $810. A fixed gap gets relatively smaller as the base grows (+$10 is 100% at $10 but 1.25% at $800). That's what makes order bias come back at high prices.
- **`PCT_GAP_BASES`, `PCT_GAPS`**: the same with percentage gaps (e.g. +10%), which keep the relative difference constant at every base price. Comparing the two gap types separates "the gap is small in dollars" from "the gap is small relative to the price".
- After changing the bases or gaps, update `ALL_PRICE_BASES` and `GAP_SIZES` in `one-by-one_order-bias-26.8.py`, or the new values won't appear in the plots.
- The $2 base was dropped on purpose: `round(2 * 1.10) == 2`, so the 10% gap turned into a same-price condition. A few fixed and relative pairs produce identical prompts (e.g. $100 vs $110). That's intentional; they're tagged separately by `price_type`.

**Which models and settings**

- **`MODELS`** (OpenAI) / **`model_id`** (Llama): the models to query. An OpenAI model has to accept `logprobs`; if it rejects a parameter, the script stops querying it and says so. The Llama script uses the model's own chat template, so other Hugging Face chat models should work, but I've only run it with Llama-3.1-8B-Instruct. Add new names to `MODEL_ORDER` in the plotting scripts to control the panel order, and to `MODEL_SIZES_B` in step 6, or they're left out of the size plots.
- **`REASONING_EFFORTS`** (OpenAI): e.g. `["none", "low"]` lets the model think before answering. Each value is a full extra run, so it multiplies the cost, and the plotting scripts don't separate the values (they'd be averaged together).
- **`TEMPERATURES_TO_ADD`** (step 5), **`TEMPERATURE`** (step 6), **`TEMPERATURES`** (step 7): which temperatures to compute from the T=1 data. These cost nothing, because no new queries are made.
- **`PILOT_MODE`** (OpenAI): runs a tiny subset first. **`MAX_WORKERS`** (OpenAI): parallel requests; lower it if you hit rate limits.

## Results

**Order bias** is the gap between how often the model picks the cheaper t-shirt when it's listed first and when it's listed second, for the same prices. For example, red is chosen with probability 0.93 in "red $800 or green $810" but 0.37 in "green $810 or red $800", so the order bias is |0.93 − 0.37| = 0.56. A rational, price-sensitive buyer would have an order bias of 0. Values below are in percentage points, averaged over all base prices and gap sizes (`results/model_size_vs_order_bias_summary*.csv`).

| Model | Fixed $ gap, T=1 | Relative % gap, T=1 | Fixed $ gap, T=5 | Relative % gap, T=5 |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | **3.4** | **5.1** | 6.0 | 8.5 |
| gpt-5.4-nano | 35.9 | 15.8 | 13.6 | 5.6 |
| gpt-5.6-luna | 12.3 | 10.6 | 7.3 | 6.0 |
| gpt-5.4-mini | 13.7 | 8.8 | 6.3 | 4.1 |
| gpt-5.4 | 24.5 | 30.0 | 22.7 | 24.0 |

What stands out:

- **A bigger model isn't a less biased one.** At T=1 the smallest model, Llama (8B), is the least affected by order on both gap types. gpt-5.4, the largest, has the most order bias on relative gaps and the second most on fixed gaps. The ranking doesn't follow model size.
- **Order matters more when the price difference is small relative to the price.** At base prices of $500 and above, every model shows more order bias than at $180 and below, most clearly with fixed gaps (gpt-5.4-nano: 15.8 → 45.9 pp; gpt-5.4: 8.6 → 32.5 pp). A $10 difference is decisive between $10 and $20, but at $800 vs. $810 it barely registers, and the order of the options takes over. Relative gaps also show more bias at high prices, though less.
- **Temperature affects each model differently.** Going from T=1 to T=5 cuts gpt-5.4-nano's fixed-gap bias by more than half, barely changes gpt-5.4's, and slightly *raises* Llama's. gpt-5.4's two scores are much further apart on average (a gap of 7.4, against 2.5 for gpt-5.4-nano), so dividing them by the temperature still leaves a clear winner. "Bias at temperature T" is therefore not one number that transfers between models.

Why this is interesting: when an LLM chooses or recommends between options (shopping assistants, agents, decision support), it can be swayed by something as irrelevant as which option was mentioned first, especially when the options differ only slightly in price. Larger, newer models aren't automatically more robust to this.

**From the [earlier experiments](earlier-experiments/README.md)** (Llama-3.1-8B, text sampling):

- **Chicken vs. pasta:** Llama has a stable preference for chicken. Pasta only wins when it's both cheaper *and* listed second. As temperature rises, the choice drifts toward 50/50, as the fitted utility model predicts.
- **Wording changes the preference.** At T ≤ 1, choosing between "blue" and "yellow" gives blue about 75% in either order. Between a "blue t-shirt" and a "yellow t-shirt" with yellow listed first, blue drops to about 35%, although the added word carries no new information.
- **PlayStation vs. Xbox:** gpt-5.4, gpt-5.4-nano and Llama almost always answer PlayStation. The only exception is Llama with Xbox listed first, at T=0.7 and T=1.
- **Random strings (IIA):** with three meaningless words, the "preferred" word and how the choice shifts when one becomes unavailable follow a fixed order. That looks more like a positional or lexical artifact than a preference. Many answers were invalid (refusals, both words repeated), so how answers are counted matters.

## Possible extensions

Ideas from the end of the internship that I didn't get to:

- **Separate order bias from color preference.** A gap like 0.40 (1.00 vs. 0.60) is large, but part of it could still be a plain preference for one color.
- **Check at T=0 first.** If a model still shows order bias with greedy decoding, it's not just a preference. Then repeat at T=1.
- **Chain-of-thought prompting.**
- **Reasoning effort:** OpenAI's newer models support `none`, `low`, `medium`, `high`, `xhigh` and `max`. All results here use `none`.
- **Newer models**, since model developers work on known biases.

---

## Appendix: other parts of the internship

**Earlier text-sampling experiments** (chicken vs. pasta and the utility fit, PlayStation vs. Xbox, random strings and IIA, colors vs. colored t-shirts, hot vs. cold): see [`earlier-experiments/`](earlier-experiments/README.md). It has its own README covering the scripts, data, figures and run order.

**Behavioral economics games** (Dictator Game, Ultimatum Game, Prisoner's Dilemma): see **[CSS-project-Robustness-Audit](https://github.com/martin-kri/CSS-project-Robustness-Audit)**. That repo has the cleaned-up, reproducible versions of the game scripts I first wrote during the internship, run on Llama-3.1-8B-Instruct and Gemma-3-4B-it, with its own README and run order. The games replicate:

- Aher et al. (2023), [*Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies*](https://arxiv.org/abs/2208.10264): Ultimatum Game
- Brookins & DeBacker (2024), [*Playing games with GPT: What can we learn about a large language model from canonical strategic games?*](https://ideas.repec.org/a/ebl/ecbull/eb-23-00457.html): Dictator Game and Prisoner's Dilemma
