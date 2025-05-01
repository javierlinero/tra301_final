import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths
file_paths = {
    'Ideal Tagging': 'evaluation_results.csv',
    'Automatic Tagging': 'evaluation_results_tagged.csv',
    'Default Output': 'pretrained_bart_results.csv'
}

# Metrics to average
metrics = ['bert_similarity', 'rouge1', 'rouge2', 'rougeL', 'rougeLsum']

# 1) Load & average
averages = {}
for label, path in file_paths.items():
    df = pd.read_csv(path)
    averages[label] = df[metrics].mean()

avg_df = pd.DataFrame(averages, index=metrics)

# 2) Plot with custom aesthetics
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, avg_df['Ideal Tagging'], width, label='Ideal Tagging', color='#1f77b4', alpha=0.9)  # Blue
bars2 = ax.bar(x, avg_df['Automatic Tagging'], width, label='Automatic Tagging', color='#ff7f0e', alpha=0.9)  # Orange
bars3 = ax.bar(x + width, avg_df['Default Output'], width, label='Default Output', color='#2ca02c', alpha=0.9)  # Green

# Labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylabel('Average Score', fontsize=12)
ax.set_title('Evaluation Metrics by Category', fontsize=14)

# Legend outside the plot
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

plt.tight_layout()
plt.show()

# save to file
fig.savefig('evaluation_metrics_comparison.png', bbox_inches='tight', dpi=300)
