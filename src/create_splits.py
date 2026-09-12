import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    processed_dir = base_dir / "data" / "processed"
    
    print("=" * 60)
    print("CREATING TRAIN/VAL/TEST SPLITS")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} aspect-level records from {df['review_id'].nunique()} unique reviews.")
    
    # Fill missing hotel identifiers using hotel_name since hotel_id might be null
    if df['hotel_name'].isna().any():
        df['hotel_name'] = df['hotel_name'].fillna("Unknown_Hotel")
    
    # ---------------------------------------------------------
    # 1. Review-Level Baseline Splits
    # ---------------------------------------------------------
    print("\n--- Generating Review-Level Splits ---")
    
    unique_reviews = df['review_id'].unique()
    
    train_ids, temp_ids = train_test_split(unique_reviews, test_size=0.2, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)
    
    baseline_train = df[df['review_id'].isin(train_ids)]
    baseline_val = df[df['review_id'].isin(val_ids)]
    baseline_test = df[df['review_id'].isin(test_ids)]
    
    print(f"Train: {len(baseline_train)} records")
    print(f"Val:   {len(baseline_val)} records")
    print(f"Test:  {len(baseline_test)} records")
    
    baseline_train.to_csv(processed_dir / "baseline_train.csv", index=False)
    baseline_val.to_csv(processed_dir / "baseline_val.csv", index=False)
    baseline_test.to_csv(processed_dir / "baseline_test.csv", index=False)
    
    # ---------------------------------------------------------
    # 2. Hotel-Grouped Splits (Main Evaluation)
    # ---------------------------------------------------------
    print("\n--- Generating Hotel-Grouped Splits ---")
    
    # Use hotel_name instead of hotel_id for groups, as hotel_id contained NaNs
    group_col = 'hotel_name'
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df[group_col]))
    
    hotel_train = df.iloc[train_idx]
    hotel_temp = df.iloc[temp_idx]
    
    gss_temp = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_temp.split(hotel_temp, groups=hotel_temp[group_col]))
    
    hotel_val = hotel_temp.iloc[val_idx]
    hotel_test = hotel_temp.iloc[test_idx]
    
    print(f"Train: {len(hotel_train)} records ({hotel_train[group_col].nunique()} hotels)")
    print(f"Val:   {len(hotel_val)} records ({hotel_val[group_col].nunique()} hotels)")
    print(f"Test:  {len(hotel_test)} records ({hotel_test[group_col].nunique()} hotels)")
    
    hotel_train.to_csv(processed_dir / "hotel_train.csv", index=False)
    hotel_val.to_csv(processed_dir / "hotel_val.csv", index=False)
    hotel_test.to_csv(processed_dir / "hotel_test.csv", index=False)
    
    print("\n" + "=" * 60)
    print("SPLITS GENERATED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
