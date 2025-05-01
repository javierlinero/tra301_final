import pandas as pd
import matplotlib.pyplot as plt
import textstat

# Load the CSV file
def analyze_flesch_scores(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure the OUTPUT column exists
    if 'OUTPUT' not in df.columns:
        raise ValueError("CSV file must contain an 'OUTPUT' column")
    
    # Calculate Flesch reading score for each entry in the OUTPUT column
    df['flesch_score'] = df['OUTPUT'].apply(lambda text: textstat.flesch_reading_ease(str(text)))
    
    # Create a histogram of the Flesch scores
    plt.figure(figsize=(10, 6))
    plt.hist(df['flesch_score'], bins=20, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel('Flesch Reading Ease Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Flesch Reading Ease Scores')
    
    # Add grid lines for better readability
    plt.grid(axis='y', alpha=0.75)
    
    # Add a vertical line for the mean score
    mean_score = df['flesch_score'].mean()
    plt.axvline(x=mean_score, color='red', linestyle='--', 
                label=f'Mean Score: {mean_score:.2f}')
    
    # Add reference lines for readability levels
    plt.axvline(x=90, color='green', linestyle=':', label='Very Easy (90+)')
    plt.axvline(x=70, color='lightgreen', linestyle=':', label='Easy (70-90)')
    plt.axvline(x=50, color='orange', linestyle=':', label='Fairly Difficult (50-70)')
    plt.axvline(x=30, color='red', linestyle=':', label='Difficult (30-50)')
    
    plt.legend()
    plt.tight_layout()
    
    # Save the histogram
    plt.savefig('flesch_score_histogram.png')
    
    # Show the histogram
    plt.show()
    
    # Return the DataFrame with scores for further analysis if needed
    return df

# Example usage
if __name__ == "__main__":
    results = analyze_flesch_scores('data.csv')
    
    # Print some basic statistics
    print(f"Average Flesch Score: {results['flesch_score'].mean():.2f}")
    print(f"Minimum Flesch Score: {results['flesch_score'].min():.2f}")
    print(f"Maximum Flesch Score: {results['flesch_score'].max():.2f}")