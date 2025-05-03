import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import textstat
import os

# Increase all font sizes for better readability when scaled in LaTeX
mpl.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
    'figure.titlesize': 24,
})

# Function to compute Flesch scores for any DataFrame's specified columns
def compute_flesch_scores(df, columns):
    for col in columns:
        df[f'flesch_score_{col}'] = df[col].astype(str).apply(textstat.flesch_reading_ease)
    return df

# Function to plot Flesch score histograms
def plot_flesch_histograms(df, columns, title_prefix, save_prefix):
    for col in columns:
        score_col = f'flesch_score_{col}'
        plt.figure(figsize=(10, 6), dpi=150)
        plt.hist(df[score_col], bins=20, edgecolor='black')

        mean_score = df[score_col].mean()
        plt.axvline(x=mean_score, linestyle='--', label=f'Mean Score: {mean_score:.2f}')
        plt.axvline(x=90, linestyle=':', label='Very Easy (90+)')
        plt.axvline(x=70, linestyle=':', label='Easy (70-90)')
        plt.axvline(x=50, linestyle=':', label='Fairly Difficult (50-70)')
        plt.axvline(x=30, linestyle=':', label='Difficult (30-50)')

        plt.xlabel('Flesch Reading Ease Score')
        plt.ylabel('Frequency')
        plt.title(title_prefix)
        plt.legend()
        plt.grid(axis='y', alpha=0.75)
        plt.tight_layout()

        filename = f'{save_prefix}_{col}_histogram.png'
        plt.savefig(filename)
        plt.show()


# Load evaluation files
eval1_path = 'evaluation_results.csv'
eval2_path = 'evaluation_results_tagged.csv'
eval3_path = 'pretrained_bart_results.csv'

eval1_df = pd.read_csv(eval1_path)
eval2_df = pd.read_csv(eval2_path)
eval3_df = pd.read_csv(eval3_path)

# Identify 'generated' columns
gen_cols_eval1 = [col for col in eval1_df.columns if 'generated' in col]
gen_cols_eval2 = [col for col in eval2_df.columns if 'generated' in col]
gen_cols_eval3 = [col for col in eval3_df.columns if 'generated_summary' in col]

# Compute scores and plot
eval1_df = compute_flesch_scores(eval1_df, gen_cols_eval1)
plot_flesch_histograms(eval1_df, gen_cols_eval1, 'Generated Summary Flesch Scores on Ideal Tagging', 'eval1')

eval2_df = compute_flesch_scores(eval2_df, gen_cols_eval2)
plot_flesch_histograms(eval2_df, gen_cols_eval2, 'Generated Summary Flesch Scores on Automatic Tagging', 'eval2')

eval3_df = compute_flesch_scores(eval3_df, gen_cols_eval3)
plot_flesch_histograms(eval3_df, gen_cols_eval3, 'Generated Summary Flesch Scores on Default Output', 'eval3')
