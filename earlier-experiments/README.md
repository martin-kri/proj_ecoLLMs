# Earlier experiments (March – July)

The experiments that came before the final order-bias pipeline in the repo root. All of them use **text sampling**: the model writes an answer, the script checks which option it named, and this repeats until there are 100 valid answers per condition. The final pipeline instead reads the first-token logprobs once. Everything here ran on **Llama-3.1-8B-Instruct** on the GPU server, except the PlayStation vs. Xbox pilot, which also used gpt-5.4 and gpt-5.4-nano.


| Folder | What it tests | Months |
|---|---|---|
| [`food_chicken_vs_pasta/`](food_chicken_vs_pasta/) | Chicken vs. pasta on a flight, with price pairs, across temperatures; fitted with the binary-choice (sigmoid) model | Mar – May |
| [`playstation_vs_xbox/`](playstation_vs_xbox/) | A pilot with a brand choice, on Llama and two OpenAI models | April |
| [`random_strings_iia/`](random_strings_iia/) | Three random 6-letter words, with and without a third "unavailable" option (Independence of Irrelevant Alternatives) | June |
| [`colors_tshirts_food/`](colors_tshirts_food/) | Colors alone vs. colored t-shirts, hot vs. cold, and chicken vs. pasta with a new and the old prompt | June – July |

---

## Setup

Same as the main README: Python 3.10, `pip install -r requirements.txt` from the repo root, and a `.env` with `HF_TOKEN` (and `OPENAI_API_KEY` for the OpenAI PlayStation vs. Xbox runs). The generators find the `.env` in the repo root on their own.

**Generators** write their CSVs into the **current folder** and **skip any condition whose CSV already exists**. Running one inside `data/` does nothing, because my data is already there. To rerun, start in a new empty folder:

```bash
cd earlier-experiments/food_chicken_vs_pasta
mkdir new_run && cd new_run
python ../generate_food_llama.py
```

**Plot scripts** take `--folder` (where the CSVs are) and `--out` (where to save figures). `--invert` plots the probability of the second option instead of the first. My figures are in each experiment's `figures/` folder.

---

## 1. Chicken vs. pasta — `food_chicken_vs_pasta/`

**Prompt:** *"You are on a flight and can choose between chicken or pasta. The chicken costs $5 and the pasta costs $20. Reply with only one of the two, without any punctuation."*

- 2 orders × 6 price pairs ($5/$20, $10/$15, $5/$10, each both ways) × 6 temperatures (0.2 – 2.5), 100 valid answers each, giving 72 files.
- Answers that don't name either option are discarded and re-asked, up to 500 attempts.

```bash
python generate_food_llama.py
python plot_food.py --folder data --out figures
```

`plot_food.py` plots P(chicken) against temperature, with 96% confidence intervals ($\pm 2.054 \cdot \text{SE}$, where $\text{SE} = \sqrt{p(1-p)/n}$). It also fits the binary-choice (logistic) model

$$
P(\text{chicken}) = \frac{1}{1 + e^{-U/T}}
$$

with `scipy.optimize.curve_fit`, where $U$ is the fitted utility difference: positive means chicken is preferred, and the larger $\lvert U \rvert$, the stronger the preference. A second row plots P against $U/T$, where all conditions fall onto the same S-curve. Points at exactly 0 or 1 are left out of the fit, because their logit is undefined.

## 2. PlayStation vs. Xbox — `playstation_vs_xbox/`

A short pilot from April, before the price experiments.

- **Llama:** *"Choose between PlayStation or Xbox, reply with only one of the two, without any punctuation"*, sampled with top_p = 0.9.
- **OpenAI:** *"Choose between PlayStation or Xbox, reply with only one of the two"*, with the system message *"You are a helpful assistant."*.
- Both orders, 100 answers per run. Llama-3.1-8B at T = 0.2, 0.7, 1.0; gpt-5.4-nano at T = 0.7, 1.0; gpt-5.4 at T = 1.0, the API default.

Each run is started by hand: set `ORDER` and `TEMPERATURE` (and `MODEL` for OpenAI) at the top of the script, run it, and repeat for the next condition.

```bash
python generate_console_llama.py
python generate_console_openai.py
python plot_console.py --folder data --out figures
```

`plot_console.py` prints the counts for every run and makes one bar chart of the share of PlayStation answers, per model, temperature and order.

## 3. Random strings and IIA — `random_strings_iia/`

Three random 6-letter words (`imphdy`, `mkzlyg`, `gdwaab`) remove any real preference. `random_word.py` generates such a word.

- **Type B (plain binary):** *"Choose between imphdy and mkzlyg. Reply with only one of the two."*: every pair, both orders.
- **Type A (IIA):** the same, plus *"Gdwaab is unavailable at the moment."*: each word as the unavailable one.
- Temperatures 0.2 – 4.5, 100 valid answers each; a few runs are also at $10 vs. $10.
- The runs at T = 3.5 and 4.5 weren't finished for every condition, so some curves stop early.

```bash
python generate_random_iia_llama.py
python plot_random_strings.py --folder data --out figures
```

One figure per word pair, with a panel per unavailable word plus the plain binary case.

The generator stored some refusals as valid answers, e.g. *"I'm not aware of any information about gdwaab or mkzlyg"*: 7.6% of all answers name two or more of the words. The plot doesn't count those (see [Corrections made after the internship](#corrections-made-after-the-internship)), so some points rest on far fewer than 100 answers. The label under each figure shows the range.

## 4. Colors, t-shirts, hot vs. cold, food — `colors_tshirts_food/`

**Prompt:** *"Choose between a red t-shirt and a blue t-shirt. Do not write full sentences. Do not explain your choice. Reply with only one of them."*, optionally with a price after each item.

| `data/` folder | Pairs | Prices | Temps | Made by |
|---|---|---|---|---|
| `colors_and_tshirts/` | red/blue, yellow/blue, yellow/red, each alone and as t-shirts | none, $10/$10, $20/$20, $10/$20, $20/$10 | 0.5 – 3.5 | earlier version of `generate_items_llama.py` |
| `hot_vs_cold/` | hot / cold | same | 0.5 – 3.5 | same as above |
| `tshirts/` | the three colored t-shirt pairs | none, same ($10, $50), mixed ($10/$15, $10/$20, $20/$50, both ways) | 0.5 – 3.5 (3.0 and 3.5 only partly) | `generate_items_llama.py` |
| `food/` | chicken / pasta | same as `tshirts/` | 0.5 – 3.5 (3.0 only partly) | `generate_items_llama.py` |
| `food_old_prompt/` | chicken / pasta with the March–May prompt wording | same as `tshirts/` | 0.5 – 3.0 | `generate_food_old_prompt_llama.py` |

```bash
python generate_items_llama.py
python generate_food_old_prompt_llama.py
python plot_text_choices.py --folder data/tshirts --out figures/tshirts
```

Run the plot once per `data/` subfolder. It reads all three file formats (`_colors`, `_items`, `_items_v2`).

- **Note on `colors_and_tshirts/` and `hot_vs_cold/`:** these were generated by the June version of the script (`18.6/colors_llama.py`). It used the same prompt but a looser answer matcher. `generate_items_llama.py` replaced it: it only reads the first 60 characters of an answer, and it rejects answers that name both options. To rerun colors alone or hot vs. cold with it, add e.g. `("red", "blue")` or `("hot", "cold")` to `BASE_PAIRS`. Because of the stricter matching, the results can differ slightly from the data here.
- **`generate_food_old_prompt_llama.py`** is the same script with the t-shirt pairs commented out and the old wording: *"You can choose between chicken or pasta. The chicken costs $10 and the pasta costs $15. Reply with only one of the two, without any punctuation."* Comparing `food/` with `food_old_prompt/` shows how much the wording alone changes the food results.

---

## Corrections made after the internship

When I looked back at these scripts while preparing this repo, I found mistakes in how two of the plot scripts count answers. The data and the generators are unchanged, but the figures here come from the corrected plot scripts, so they differ from the ones in my internship presentations. The corrected lines are marked in the code.

- **`random_strings_iia/plot_random_strings.py`**
  - **Refusals were counted as votes.** An answer naming several of the words was counted for whichever word the script checked first. That's where the earlier "the point always goes to imphdy" observation came from. These answers are no longer counted.
  - **Unclassified answers counted against the first word.** P(word) was computed over all stored answers, so every unclassified one lowered it. It's now the share among answers that name one of the two words.
  - **Wrong candidates for some files.** For files like `imphdy_or_gdwaab_…`, the script took its two candidates from a fixed table meant for the imphdy/mkzlyg pair. Now they come from the file's own word pair.
- **`food_chicken_vs_pasta/plot_food.py`**: an answer naming both foods (e.g. *"pasta, it's cheaper than the chicken"*) counted as chicken. Such answers (39 of 7,200) are no longer counted, and P(chicken) is the share among classified answers. No point moved by more than 0.03.

The later scripts in `colors_tshirts_food/` already handled this, because their generator rejects answers that name both options.

## What's in the CSVs

One file per condition, one row per answer. The filename holds the condition: order, model, temperature, and the price pair if there is one, e.g. `chicken or pasta_no_memory_Llama_3.1_8B_Instruct_1.0_flight_5_20.csv`. `no_memory` means every answer comes from a fresh conversation.

| Column | Meaning |
|---|---|
| `iteration` | The number of the valid answer (1–100). |
| `attempt` | How many tries it took to get there. Invalid answers (refusals, both options named, something else) are thrown away and asked again, so `attempt − iteration` is the number of discarded answers so far. The plots show these as "~N extra" (in the random-strings and food plots, together with any stored answers the plot doesn't count). |
| `response` | The model's full answer text. |
| `matched_choice` | (`colors_tshirts_food/` only) Which option the answer was counted as. |
| `experiment_type`, `unavailable` | (random strings, colors) `A_unavailable` with the word that was declared unavailable, or `B_plain` for a normal two-option choice. |

The PlayStation vs. Xbox files only have `iteration` and `response`. Invalid answers weren't re-asked there, so `plot_console.py` counts them as "other" (there were none).

---
