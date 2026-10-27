import os
from openai import OpenAI
import pandas as pd
import time
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


# ── API config ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID") or None,
)

# ── Settings: change these and rerun once per condition ───────────────────────
MODEL = "gpt-5.4"               # my runs: "gpt-5.4" and "gpt-5.4-nano"
ORDER = "Xbox or PlayStation"   # or "PlayStation or Xbox"
TEMPERATURE = None              # None = API default (1.0); my nano runs also used 0.7

prompt = f"Choose between {ORDER}, reply with only one of the two"

# number of iterations
num_iter = 100

# To store results
responses = []

# Only send temperature when it's set, so the API default is used otherwise
extra_args = {} if TEMPERATURE is None else {"temperature": TEMPERATURE}

# --- Loop 100 times ---
for i in range(num_iter):
    print(f"Iteration {i + 1}...")

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=8384,
        **extra_args,
    )

    response = None

    try:
        response = completion.choices[0].message.content
        print(response)
        responses.append({"iteration": i + 1, "response": response})

        # Optional: a short pause to avoid rate limits
        time.sleep(0.5)

    except Exception as e:
        print(f"Error at iteration {i + 1}: {e}")
        responses.append({"iteration": i + 1, "response": None})

# --- Save to CSV ---
df = pd.DataFrame(responses)
temp_tag = 1.0 if TEMPERATURE is None else float(TEMPERATURE)
file_path = f"{ORDER.replace(' ', '_')}_{MODEL}_temp{temp_tag}.csv"
df.to_csv(file_path, index=False)

print(f"✅ Done! Responses saved to {file_path}")
