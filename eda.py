import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sample_df = pd.read_pickle('truthfulqa_sample.pkl')

# ---- Feature engineering for EDA (derived columns) ----
sample_df['question_length'] = sample_df['question'].apply(lambda x: len(x.split()))
sample_df['best_answer_length'] = sample_df['best_answer'].apply(lambda x: len(x.split()))
sample_df['num_correct_answers'] = sample_df['correct_answers'].apply(len)
sample_df['num_incorrect_answers'] = sample_df['incorrect_answers'].apply(len)

# ================================================================
# 1. Category distribution — bar chart
# ================================================================
plt.figure(figsize=(12, 8))
sample_df['category'].value_counts().plot(kind='barh')
plt.title('Question Count by Category')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('eda_category_distribution.png', dpi=150)
plt.savefig('category_distribution.png', bbox_inches='tight', dpi=300)
print("Graph successfully saved as category_distribution.png!")

# ================================================================
# 2. Type distribution (Adversarial vs Non-Adversarial) — pie/bar
# ================================================================
plt.figure(figsize=(6, 6))
sample_df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Adversarial vs Non-Adversarial Split')
plt.ylabel('')
plt.tight_layout()
plt.savefig('eda_type_split.png', dpi=150)
plt.show()

# ================================================================
# 3. Question length — histogram + descriptive stats
# ================================================================
plt.figure(figsize=(10, 5))
plt.hist(sample_df['question_length'], bins=20, edgecolor='black')
plt.axvline(sample_df['question_length'].mean(), color='red', linestyle='--', label=f"Mean: {sample_df['question_length'].mean():.1f}")
plt.axvline(sample_df['question_length'].median(), color='green', linestyle='--', label=f"Median: {sample_df['question_length'].median():.1f}")
plt.title('Distribution of Question Length (words)')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.savefig('eda_question_length_hist.png', dpi=150)
plt.show()

print("Question length stats:")
print(sample_df['question_length'].describe())
print(f"25th percentile: {sample_df['question_length'].quantile(0.25)}")
print(f"75th percentile: {sample_df['question_length'].quantile(0.75)}")

# ================================================================
# 4. Question length by category — box plot
# ================================================================
plt.figure(figsize=(14, 8))
top_categories = sample_df['category'].value_counts().head(10).index
subset = sample_df[sample_df['category'].isin(top_categories)]
sns.boxplot(data=subset, x='category', y='question_length')
plt.xticks(rotation=45, ha='right')
plt.title('Question Length by Category (Top 10 Categories)')
plt.tight_layout()
plt.savefig('eda_boxplot_length_by_category.png', dpi=150)
plt.show()

# ================================================================
# 5. Question length by type — box plot (Adversarial vs Non-Adversarial)
# ================================================================
plt.figure(figsize=(8, 6))
sns.boxplot(data=sample_df, x='type', y='question_length')
plt.title('Question Length: Adversarial vs Non-Adversarial')
plt.tight_layout()
plt.savefig('eda_boxplot_length_by_type.png', dpi=150)
plt.show()

# ================================================================
# 6. Number of correct vs incorrect answers — distribution plot
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(sample_df['num_correct_answers'], bins=15, kde=True, ax=axes[0])
axes[0].set_title('Distribution: Number of Correct Answers per Question')
sns.histplot(sample_df['num_incorrect_answers'], bins=15, kde=True, ax=axes[1], color='orange')
axes[1].set_title('Distribution: Number of Incorrect Answers per Question')
plt.tight_layout()
plt.savefig('eda_answer_counts_distribution.png', dpi=150)
plt.show()

# ================================================================
# 7. Scatter plot — question length vs number of incorrect answers
# ================================================================
plt.figure(figsize=(8, 6))
sns.scatterplot(data=sample_df, x='question_length', y='num_incorrect_answers', hue='type', alpha=0.6)
plt.title('Question Length vs Number of Incorrect (Trap) Answers')
plt.tight_layout()
plt.savefig('eda_scatter_length_vs_traps.png', dpi=150)
plt.show()

# ================================================================
# 8. Correlation heatmap — numeric features
# ================================================================
numeric_cols = ['question_length', 'best_answer_length', 'num_correct_answers', 'num_incorrect_answers']
corr_matrix = sample_df[numeric_cols].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap: Question/Answer Features')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150)
plt.show()

# ================================================================
# 9. Summary statistics table — for your report
# ================================================================
summary_stats = sample_df[numeric_cols].describe().T
summary_stats['skewness'] = sample_df[numeric_cols].skew()
summary_stats['variance'] = sample_df[numeric_cols].var()
print("\nFull summary statistics table:")
print(summary_stats)
summary_stats.to_csv('eda_summary_statistics.csv')