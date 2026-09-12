import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    results_dir = base_dir / "results"
    figures_dir = base_dir / "figures" / "temporal_plots"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TEMPORAL ANALYSIS: aspect_eval_corpus.csv")
    print("=" * 60)
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Parsing dates and extracting Year-Month...")
    df['post_datetime'] = pd.to_datetime(df['post_datetime'], utc=True)
    df['year_month'] = df['post_datetime'].dt.to_period('M')
    
    MIN_REVIEWS = 5
    
    # -----------------------------------------------------
    # 1. Global Temporal Trends
    # -----------------------------------------------------
    print("\nCalculating Global Trends...")
    global_grouped = df.groupby(['year_month', 'aspect']).agg(
        review_count=('review_id', 'count'),
        avg_score=('score', 'mean'),
        avg_sentiment=('sentiment_label', 'mean')
    ).reset_index()
    
    # Apply sample size threshold
    global_valid = global_grouped[global_grouped['review_count'] >= MIN_REVIEWS].copy()
    
    # Sort and calculate delta (period-over-period change)
    global_valid = global_valid.sort_values(['aspect', 'year_month'])
    global_valid['score_delta'] = global_valid.groupby('aspect')['avg_score'].diff()
    global_valid['sentiment_delta'] = global_valid.groupby('aspect')['avg_sentiment'].diff()
    
    global_out_path = results_dir / "global_aspect_trends.csv"
    global_valid.to_csv(global_out_path, index=False)
    print(f"Saved global trends to {global_out_path}")
    
    # Plotting Global Trends
    plt.figure(figsize=(14, 8))
    # Convert period back to datetime for plotting compatibility
    global_valid['date'] = global_valid['year_month'].dt.to_timestamp()
    
    sns.lineplot(data=global_valid, x='date', y='avg_score', hue='aspect', marker='o')
    plt.title('Global Average Aspect Score over Time (≥5 reviews/month)')
    plt.ylabel('Average Score (1-5)')
    plt.xlabel('Date')
    plt.legend(title='Aspect', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(figures_dir / "global_trends_score.png")
    plt.close()
    print(f"Saved global trend chart to {figures_dir / 'global_trends_score.png'}")
    
    # -----------------------------------------------------
    # 2. Per-Hotel Temporal Trends
    # -----------------------------------------------------
    print("\nCalculating Per-Hotel Trends...")
    hotel_grouped = df.groupby(['hotel_name', 'year_month', 'aspect']).agg(
        review_count=('review_id', 'count'),
        avg_score=('score', 'mean'),
        avg_sentiment=('sentiment_label', 'mean')
    ).reset_index()
    
    hotel_valid = hotel_grouped[hotel_grouped['review_count'] >= MIN_REVIEWS].copy()
    
    hotel_valid = hotel_valid.sort_values(['hotel_name', 'aspect', 'year_month'])
    hotel_valid['score_delta'] = hotel_valid.groupby(['hotel_name', 'aspect'])['avg_score'].diff()
    hotel_valid['sentiment_delta'] = hotel_valid.groupby(['hotel_name', 'aspect'])['avg_sentiment'].diff()
    
    hotel_out_path = results_dir / "hotel_aspect_trends.csv"
    hotel_valid.to_csv(hotel_out_path, index=False)
    print(f"Saved per-hotel trends to {hotel_out_path}")
    
    # Identify the biggest absolute upgrades and degrades globally and per-hotel
    print("\n--- Largest Global Upgrades (Month-over-Month) ---")
    top_up = global_valid.nlargest(3, 'score_delta')
    for _, r in top_up.iterrows():
        print(f"  {r['year_month']} | {r['aspect']}: +{r['score_delta']:.2f} (Count: {r['review_count']})")
        
    print("\n--- Largest Global Degrades (Month-over-Month) ---")
    top_down = global_valid.nsmallest(3, 'score_delta')
    for _, r in top_down.iterrows():
        print(f"  {r['year_month']} | {r['aspect']}: {r['score_delta']:.2f} (Count: {r['review_count']})")
        
    print("\nDone with Step 2.")

if __name__ == "__main__":
    main()
