import pandas as pd
import json
from pathlib import Path
import warnings

# Suppress pandas FutureWarnings for clean output
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    # Adjust path assuming script is run from project root
    base_dir = Path(__file__).parent.parent
    corpus_path = base_dir / "data" / "processed" / "research_corpus.jsonl"
    
    print("=" * 60)
    print("PROFILING RESEARCH CORPUS (UPDATED SCHEMA)")
    print("=" * 60)
    print(f"Loading data from: {corpus_path}")
    
    # Load JSONL into a flattened pandas dataframe to easily access nested fields
    data = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
            
    df = pd.json_normalize(data)
    
    print(f"\nTotal records loaded: {len(df)}")
    print(f"Columns found: {df.columns.tolist()}")
    
    # 1. Text Metrics & Duplicate Detection
    print("\n" + "-" * 30)
    print("1. TEXT METRICS & DUPLICATES")
    print("-" * 30)
    
    if 'review_id' in df.columns:
        dup_ids = df['review_id'].duplicated().sum()
        print(f"Duplicate 'review_id' count: {dup_ids}")
    
    if 'comment' in df.columns:
        dup_comments = df['comment'].duplicated().sum()
        missing_comments = df['comment'].isna().sum()
        print(f"Duplicate 'comment' texts: {dup_comments} (includes empty/generic reviews)")
        print(f"Missing 'comment' fields: {missing_comments}")
        
        df['comment_len'] = df['comment'].astype(str).str.len()
        print("\nComment Character Length Distribution:")
        print(df['comment_len'].describe(percentiles=[.25, .5, .75, .90, .99]).round(2))
    
    # 2. Temporal Coverage
    print("\n" + "-" * 30)
    print("2. TEMPORAL COVERAGE")
    print("-" * 30)
    
    if 'post_datetime' in df.columns:
        df['post_datetime_parsed'] = pd.to_datetime(df['post_datetime'], errors='coerce')
        missing_dates = df['post_datetime_parsed'].isna().sum()
        print(f"post_datetime Min: {df['post_datetime_parsed'].min()}")
        print(f"post_datetime Max: {df['post_datetime_parsed'].max()}")
        print(f"post_datetime Missing/Invalid: {missing_dates}")
        
    if 'check_in_date' in df.columns:
        missing_checkins = df['check_in_date'].isna().sum()
        print(f"check_in_date Missing: {missing_checkins}")
    
    # 3. Aspect Score Distributions
    print("\n" + "-" * 30)
    print("3. ASPECT SCORE DISTRIBUTIONS")
    print("-" * 30)
    
    aspects = ['room', 'hospitality', 'location', 'bath', 'facility', 'cleanliness', 'breakfast', 'dinner']
    for aspect in aspects:
        col = f"scores.{aspect}"
        if col in df.columns:
            print(f"\n{aspect.upper()} Scores:")
            counts = df[col].value_counts(dropna=False).sort_index()
            for val, count in counts.items():
                print(f"  {val}: {count}")
        else:
            print(f"\n{aspect.upper()}: Column not found!")

    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
