import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    
    print("=" * 60)
    print("DATA QUALITY CHECK: aspect_eval_corpus.csv")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    total_rows = len(df)
    print(f"Total Rows: {total_rows:,}\n")
    
    # 1. Check post_datetime
    print("--- post_datetime Check ---")
    null_date_count = df['post_datetime'].isnull().sum()
    print(f"Null post_datetime: {null_date_count:,} ({(null_date_count/total_rows)*100:.2f}%)")
    
    # Try parsing to find malformed dates
    parsed_dates = pd.to_datetime(df['post_datetime'], errors='coerce', utc=True)
    malformed_date_count = parsed_dates.isnull().sum() - null_date_count
    print(f"Malformed post_datetime (fails to parse): {malformed_date_count:,} ({(malformed_date_count/total_rows)*100:.2f}%)\n")
    
    # 2. Check hotel_name
    print("--- hotel_name Check ---")
    null_hotel_count = df['hotel_name'].isnull().sum()
    print(f"Null hotel_name: {null_hotel_count:,} ({(null_hotel_count/total_rows)*100:.2f}%)\n")
    
    # 3. Check aspects
    print("--- aspect Check ---")
    aspect_counts = df['aspect'].value_counts(dropna=False)
    print(f"Distinct aspect values found: {len(aspect_counts)}")
    for aspect_val, count in aspect_counts.items():
        print(f"  - '{aspect_val}': {count:,} ({(count/total_rows)*100:.2f}%)")
    
    expected_aspects = {'room', 'hospitality', 'location', 'bath', 'facility', 'cleanliness', 'breakfast', 'dinner'}
    found_aspects = set(df['aspect'].dropna().unique())
    if found_aspects != expected_aspects:
        print("\nWARNING: Aspect names do not match the expected exactly 8 lowercase values.")
        print(f"Expected: {expected_aspects}")
        print(f"Found:    {found_aspects}")
    else:
        print("\nPerfect! Exactly 8 expected aspect values found, no typos or case mismatches.")
        
    # 4. Check score / sentiment_label nulls per aspect
    print("\n--- score & sentiment_label Check ---")
    print(f"{'Aspect':<15} | {'Null score':<15} | {'Null sentiment_label'}")
    print("-" * 55)
    
    for aspect_val in found_aspects:
        aspect_df = df[df['aspect'] == aspect_val]
        aspect_total = len(aspect_df)
        null_score = aspect_df['score'].isnull().sum()
        null_sentiment = aspect_df['sentiment_label'].isnull().sum()
        
        pct_score = (null_score / aspect_total) * 100 if aspect_total > 0 else 0
        pct_sentiment = (null_sentiment / aspect_total) * 100 if aspect_total > 0 else 0
        
        print(f"{aspect_val:<15} | {null_score:,} ({pct_score:.2f}%)   | {null_sentiment:,} ({pct_sentiment:.2f}%)")

if __name__ == "__main__":
    main()
