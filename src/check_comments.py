import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    
    print("=" * 60)
    print("TEXT DATA QUALITY CHECK: aspect_eval_corpus.csv")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    total_rows = len(df)
    
    # 1. Null or empty comments
    null_comments = df['comment'].isnull().sum()
    empty_comments = (df['comment'].str.strip() == '').sum() if null_comments < total_rows else 0
    total_missing = null_comments + empty_comments
    print(f"Null or empty comments: {total_missing:,} ({(total_missing/total_rows)*100:.2f}%)")
    
    # 2. Short comments (< 5 chars)
    # Need to handle NaN for str.len() by filling with empty string or dropping
    short_comments = (df['comment'].dropna().str.strip().str.len() < 5).sum()
    print(f"Very short comments (< 5 chars): {short_comments:,} ({(short_comments/total_rows)*100:.2f}%)")
    
    # 3. Exact duplicates
    # Since we melted, 1 review becomes up to 8 rows. We should check for duplicate reviews 
    # based on review_id first, but the user asked about duplicate comment text.
    # Let's count unique comment strings vs total unique review_ids.
    unique_review_ids = df['review_id'].nunique()
    unique_comments = df['comment'].nunique()
    
    print(f"\nUnique review IDs: {unique_review_ids:,}")
    print(f"Unique comment strings: {unique_comments:,}")
    
    if unique_comments < unique_review_ids:
        diff = unique_review_ids - unique_comments
        print(f"WARNING: There are {diff:,} fewer unique comments than review IDs, suggesting identical review text across different reviews.")
    elif unique_comments == unique_review_ids:
        print("Perfect: Every unique review ID has exactly one unique comment string.")
    else:
        print("Note: There are more unique comment strings than review IDs. This is unusual but could happen if review_id is not globally unique.")

if __name__ == "__main__":
    main()
