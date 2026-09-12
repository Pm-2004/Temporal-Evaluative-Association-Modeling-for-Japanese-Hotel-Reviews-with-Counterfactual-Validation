import pandas as pd
from pathlib import Path
import sentencepiece as spm

def main():
    # Adjust path assuming script is run from project root or src/
    base_dir = Path(__file__).parent.parent
    processed_dir = base_dir / "data" / "processed"
    
    print("=" * 60)
    print("1. INSPECTING PROCESSED CSV SPLITS")
    print("=" * 60)
    
    splits = ["train.csv", "dev.csv", "test.csv"]
    for split in splits:
        file_path = processed_dir / split
        if file_path.exists():
            df = pd.read_csv(file_path)
            print(f"\n--- {split} ---")
            print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"Columns: {df.columns.tolist()}")
            
            if 'label' in df.columns:
                print(f"Label distribution: {df['label'].value_counts().to_dict()}")
            
            # Check for aspects and hotel groupings
            aspect_keywords = ['room', 'hospitality', 'location', 'bath', 'facility', 'cleanliness', 'score']
            aspect_cols = [c for c in df.columns if any(k in c.lower() for k in aspect_keywords)]
            hotel_cols = [c for c in df.columns if 'hotel' in c.lower() or 'provider' in c.lower()]
            
            print(f"Aspect-related columns: {aspect_cols if aspect_cols else 'None'}")
            print(f"Hotel identifier columns: {hotel_cols if hotel_cols else 'None'}")
            
            print("\nFirst row sample:")
            sample_dict = df.iloc[0].to_dict() if len(df) > 0 else "Empty"
            for k, v in sample_dict.items():
                # Truncate text if it's too long
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                print(f"  {k}: {v_str}")
        else:
            print(f"\n--- {split} ---")
            print("FILE NOT FOUND")

    print("\n" + "=" * 60)
    print("2. INSPECTING SENTENCEPIECE MODEL")
    print("=" * 60)
    
    sp_model_path = processed_dir / "sp_japanese.model"
    if sp_model_path.exists():
        try:
            sp = spm.SentencePieceProcessor()
            sp.load(str(sp_model_path))
            print(f"Model path: {sp_model_path}")
            print(f"Vocab size: {sp.get_piece_size()}")
            
            # Sample some common tokens
            sample_ids = [10, 50, 100, 500, 1000]
            sample_tokens = {i: sp.id_to_piece(i) for i in sample_ids if i < sp.get_piece_size()}
            print(f"Sample token mapping: {sample_tokens}")
            
            # Test encoding a sample string
            test_str = "部屋はとてもきれいでした"
            encoded_ids = sp.encode_as_ids(test_str)
            encoded_pieces = sp.encode_as_pieces(test_str)
            print(f"\nTest Encoding: '{test_str}'")
            print(f"Pieces: {encoded_pieces}")
            print(f"IDs: {encoded_ids}")
            
        except Exception as e:
            print(f"Failed to load SentencePiece model: {e}")
    else:
        print(f"Model not found at {sp_model_path}")

if __name__ == "__main__":
    main()
