import pandas as pd
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent
    input_path = base_dir / "data" / "processed" / "aspect_eval_corpus.csv"
    
    print("=" * 60)
    print("CROSS-HOTEL DUPLICATE CHECK")
    print("=" * 60)
    
    df = pd.read_csv(input_path)
    
    # We only care about unique reviews for this check (1 row per review_id)
    reviews_df = df[['review_id', 'comment', 'hotel_name']].drop_duplicates(subset=['review_id'])
    
    # Find comments that appear in more than one review_id
    comment_counts = reviews_df['comment'].value_counts()
    duplicate_comments = comment_counts[comment_counts > 1].index
    
    print(f"Total unique comments with duplicates: {len(duplicate_comments)}")
    
    # Check how many of these cross hotel boundaries
    cross_hotel_count = 0
    cross_hotel_examples = []
    
    for comment in duplicate_comments:
        hotels = reviews_df[reviews_df['comment'] == comment]['hotel_name'].unique()
        if len(hotels) > 1:
            cross_hotel_count += 1
            if len(cross_hotel_examples) < 5:
                cross_hotel_examples.append((comment, list(hotels)))
                
    print(f"\nComments appearing in multiple DIFFERENT hotels: {cross_hotel_count}")
    
    if cross_hotel_count > 0:
        print("\n--- Examples of cross-hotel duplicates ---")
        for comment, hotels in cross_hotel_examples:
            short_comment = comment[:50] + "..." if len(comment) > 50 else comment
            print(f"Text: '{short_comment}'")
            print(f"Hotels: {hotels}\n")
    else:
        print("\nExcellent! None of the duplicate comments cross hotel boundaries. No leakage risk.")

if __name__ == "__main__":
    main()
