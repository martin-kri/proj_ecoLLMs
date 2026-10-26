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

- If your machine has a different CUDA version (or no GPU), install `torch` separately for your setup ([pytorch.org](https://pytorch.org/get-started/locally/)) before installing the rest.
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
> My checkpoint files are deliberately **not** in this repo: they contain every prompt already, so a run would skip everything and just return my data. Without them, a fresh clone sends every prompt again. The new run overwrites the master CSVs in `results/`; `git checkout results/` brings mine back.

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

At T=1 the logprob difference equals the raw logit difference, so the choice probability at any temperature T can be computed from the saved scores:

P(option1) = e^(s1/T) / (e^(s1/T) + e^(s2/T))

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

## Changing the experiment

All the settings are constants at the top of each script:

- **Other prompt sets:** set `ACTIVE_PROMPT_SETS = {1, 2, 3}` in both data scripts and in the plotting scripts. Set A adds more output constraints, Set B adds more context; the full prompt text is in `build_prompts()`.
- **Other colors:** add pairs to `COLOR_PAIRS`.
- **Currency:** set `CURRENCY = "EUR"`.
- **Price grid:** edit `FIXED_GAP_BASES`, `FIXED_GAPS`, `PCT_GAP_BASES` and `PCT_GAPS`. Keep them the same in both data scripts so the models stay comparable.
  - The $2 base was dropped on purpose: `round(2 * 1.10) == 2`, so the 10% gap turned into a same-price condition.
  - A few fixed and relative pairs produce identical prompts, e.g. $100 vs $110. That's intentional; they're tagged separately by `price_type`.
- **Reasoning effort (OpenAI):** add values to `REASONING_EFFORTS`. Each value is a full extra run, so it multiplies the cost.

## Possible extensions

Ideas from the end of the internship that I didn't get to:

- **Separate order bias from color preference.** A gap like 0.40 (1.00 vs. 0.60) is large, but part of it could still be a plain preference for one color.
- **Check at T=0 first.** If a model still shows order bias with greedy decoding, it's not just a preference. Then repeat at T=1.
- **Chain-of-thought prompting.**
- **Reasoning effort:** OpenAI's newer models support `none`, `low`, `medium`, `high`, `xhigh` and `max`. All results here use `none`.
- **Newer models**, since model developers work on known biases.

---

## Appendix: other parts of the internship

This repo contains only the final experiment. The rest of the internship (chicken vs. pasta and the utility fit, random strings and IIA, colors vs. colored t-shirts, hot vs. cold) is described in my internship report, not here.

**Behavioral economics games** (Dictator Game, Ultimatum Game, Prisoner's Dilemma): see **[CSS-project-Robustness-Audit](https://github.com/martin-kri/CSS-project-Robustness-Audit)**. That repo has the cleaned-up, reproducible versions of the game scripts I first wrote during the internship, run on Llama-3.1-8B-Instruct and Gemma-3-4B-it, with its own README and run order. The games replicate:

- Aher et al. (2023), [*Using Large Language Models to Simulate Multiple Humans and Replicate Human Subject Studies*](https://arxiv.org/abs/2208.10264): Ultimatum Game
- Brookins & DeBacker (2024), [*Playing games with GPT: What can we learn about a large language model from canonical strategic games?*](https://ideas.repec.org/a/ebl/ecbull/eb-23-00457.html): Dictator Game and Prisoner's Dilemma
