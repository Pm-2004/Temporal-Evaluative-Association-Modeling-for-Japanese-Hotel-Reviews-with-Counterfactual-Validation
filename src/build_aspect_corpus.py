import pandas as pd
import json
from pathlib import Path
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "research_corpus.jsonl"
    output_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    
    print("=" * 60)
    print("BUILDING ASPECT-EVALUATION CORPUS")
    print("=" * 60)
    
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    df = pd.json_normalize(data)
    
    initial_len = len(df)
    
    # 1. Clean Text
    # Drop empty or extremely short comments (e.g. less than 5 characters)
    df = df.dropna(subset=['comment'])
    df = df[df['comment'].str.strip().str.len() >= 5]
    
    # 2. Filter missing core aspect blocks
    # As seen in the profile, 7,728 records completely lack aspect scores.
    df = df.dropna(subset=['scores.room'])
    
    # 3. Melt to Aspect-Level (Long Format)
    # This transforms the dataset so each row is a single (Review, Aspect, Score) pair
    aspects = ['room', 'hospitality', 'location', 'bath', 'facility', 'cleanliness', 'breakfast', 'dinner']
    score_cols = [f"scores.{a}" for a in aspects]
    
    keep_cols = ['hotel_id', 'hotel_name', 'review_id', 'comment', 'post_datetime']
    available_score_cols = [c for c in score_cols if c in df.columns]
    
    df_melt = pd.melt(
        df[keep_cols + available_score_cols],
        id_vars=keep_cols,
        value_vars=available_score_cols,
        var_name='aspect',
        value_name='score'
    )
    
    # Clean up aspect names (e.g. "scores.room" -> "room")
    df_melt['aspect'] = df_melt['aspect'].str.replace('scores.', '')
    
    # 4. Filter out unrated aspects
    # In Rakuten data, 0.0 means the reviewer didn't rate that specific aspect
    df_melt = df_melt.dropna(subset=['score'])
    df_melt = df_melt[df_melt['score'] > 0.0]
    
    # 5. Map 1-5 stars to Sentiment Label (0=Negative, 1=Neutral, 2=Positive)
    # This aligns our target labels with standard 3-way sequence classification
    def map_sentiment(score):
        if score <= 2.0: return 0
        elif score == 3.0: return 1
        else: return 2
        
    df_melt['sentiment_label'] = df_melt['score'].apply(map_sentiment)
    
    # Sort logically
    df_melt = df_melt.sort_values(['hotel_id', 'post_datetime'])
    
    # Save to disk
    df_melt.to_csv(output_path, index=False)
    
    print(f"Original reviews: {initial_len}")
    print(f"Valid reviews after dropping missing/short texts: {df['review_id'].nunique()}")
    print(f"Final aspect-level records generated: {len(df_melt)}\n")
    
    print("--- Aspect Distribution ---")
    print(df_melt['aspect'].value_counts().to_string())
    
    print("\n--- Sentiment Label Distribution ---")
    label_counts = df_melt['sentiment_label'].value_counts().sort_index()
    print(f"0 (Negative): {label_counts.get(0, 0)}")
    print(f"1 (Neutral):  {label_counts.get(1, 0)}")
    print(f"2 (Positive): {label_counts.get(2, 0)}")
    
    print(f"\nSaved research-ready dataset to: {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
