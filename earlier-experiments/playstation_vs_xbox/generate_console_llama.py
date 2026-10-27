import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from huggingface_hub import login
import transformers
import torch
import time
import pandas as pd

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Missing HF_TOKEN. Set it in your .env or server environment.")

login(token=HF_TOKEN)

# ── Settings: change these and rerun once per condition ───────────────────────
model_id = "meta-llama/Llama-3.1-8B-Instruct"
ORDER = "Xbox or PlayStation"   # or "PlayStation or Xbox"
TEMPERATURE = 1.0               # my runs: 0.2, 0.7, 1.0

prompt = f"Choose between {ORDER}, reply with only one of the two, without any punctuation"

# number of iterations
num_iter = 100

# To store results
responses = []

# Create pipeline once
pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16, "use_safetensors": True},
            device_map="auto",
            trust_remote_code = True
        )

messages = [
            {"role": "user", "content": prompt},
        ]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# --- Loop 100 times ---
for i in range(num_iter):
    print(f"Iteration {i+1}...")
    print(prompt)

    try:
        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.9,
        )

        output = outputs[0]["generated_text"][-1]['content']
        responses.append({"iteration": i+1, "response": output})
        print(output)

    except Exception as e:
        print(f"Error at iteration {i+1}: {e}")
        responses.append({"iteration": i+1, "response": None})

    # Optional: a short pause to avoid rate limits
    time.sleep(1)

# --- Save to CSV ---
df = pd.DataFrame(responses)
model_name = model_id.split("/")[-1]
file_path = f"{ORDER.replace(' ', '_')}_{model_name}_temp{float(TEMPERATURE)}.csv"
df.to_csv(file_path, index=False)
print(f"✅ Done! Responses saved to {file_path}")
