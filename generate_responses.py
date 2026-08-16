import pandas as pd
import time
import json
from groq import Groq

# Load the dataset saved from Phase 1
sample_df = pd.read_pickle('truthfulqa_sample.pkl')
print(f"Loaded {len(sample_df)} questions")

client = Groq()

# ---- FUNCTION DEFINITION (this was missing) ----
def get_response(question, model_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": question}],
                temperature=0.8,
                max_tokens=150
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Retry {attempt+1}, error: {e}")
            time.sleep(2 ** attempt)
    return None

N_RESPONSES = 10
MODEL_NAME = "llama-3.3-70b-versatile"
OUTPUT_FILE = "responses_model_b.jsonl"

with open(OUTPUT_FILE, "w") as f:
    for idx, row in sample_df.iterrows():
        for i in range(N_RESPONSES):
            answer = get_response(row['question'], MODEL_NAME)
            record = {
                "question_id": row['question_id'],
                "question": row['question'],
                "response_num": i,
                "response": answer
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
        print(f"Done question {idx+1}/{len(sample_df)} (id={row['question_id']})")
        time.sleep(1)
