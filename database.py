from datasets import load_dataset
import pandas as pd

# Load
dataset = load_dataset("domenicrosati/TruthfulQA")
df = dataset["train"].to_pandas()

# Rename columns
clean_df = df.rename(columns={
    'Question': 'question',
    'Best Answer': 'best_answer',
    'Correct Answers': 'correct_answers',
    'Incorrect Answers': 'incorrect_answers',
    'Category': 'category',
    'Type': 'type',
    'Source': 'source'
})

# Split semicolon-separated strings into lists
clean_df['correct_answers'] = clean_df['correct_answers'].apply(
    lambda x: [a.strip() for a in x.split(';')] if isinstance(x, str) else []
)
clean_df['incorrect_answers'] = clean_df['incorrect_answers'].apply(
    lambda x: [a.strip() for a in x.split(';')] if isinstance(x, str) else []
)

# Filter out rows with no correct answers
clean_df = clean_df[clean_df['correct_answers'].apply(len) > 0].reset_index(drop=True)
print(f"After filtering: {len(clean_df)} rows")

# Stratified sampling — manual loop (safest across pandas versions)
TARGET_SAMPLE_SIZE = 250
sample_parts = []

for cat, group in clean_df.groupby('category'):
    n = max(1, round(len(group) * TARGET_SAMPLE_SIZE / len(clean_df)))
    n = min(n, len(group))  # can't sample more than exists in that category
    sample_parts.append(group.sample(n=n, random_state=42))

sample_df = pd.concat(sample_parts, ignore_index=True)

# Assign clean, permanent question_id — this ID is what you'll join all future
# data against (responses, embeddings, consistency scores, labels), so lock it in now
sample_df['question_id'] = range(len(sample_df))

# Reorder columns for readability
sample_df = sample_df[['question_id', 'category', 'type', 'question', 'best_answer',
                        'correct_answers', 'incorrect_answers', 'source']]

# Save — pickle preserves list columns properly, CSV is for quick manual inspection only
sample_df.to_pickle('truthfulqa_sample.pkl')
sample_df.to_csv('truthfulqa_sample_readable.csv', index=False)  # lists will show as strings here, that's fine, just for eyeballing

print(f"Final sample size: {len(sample_df)}")
print(sample_df['category'].value_counts())
print(sample_df['type'].value_counts())

sample_df = pd.read_pickle('truthfulqa_sample.pkl')
print(sample_df.shape)
print(sample_df.isnull().sum())  # check no unexpected nulls crept in
print(sample_df.iloc[0])  # eyeball one full row