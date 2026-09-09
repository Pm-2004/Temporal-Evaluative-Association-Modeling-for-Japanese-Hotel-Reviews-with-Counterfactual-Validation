import pandas as pd
import sentencepiece as spm
from pathlib import Path


# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "sp_japanese.model"


# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# Load SentencePiece model
# ==========================================

sp = spm.SentencePieceProcessor()

sp.load(str(MODEL_PATH))


# ==========================================
# Test several Japanese sentences
# ==========================================

print("=" * 70)
print("SENTENCEPIECE SUBWORD TOKENIZATION")
print("=" * 70)

for i in range(10):

    text = str(df.iloc[i]["text"])

    pieces = sp.encode(
        text,
        out_type=str
    )

    print(f"\nExample {i + 1}")
    print("-" * 70)

    print("Original:")
    print(text)

    print("\nSubwords:")
    print(" | ".join(pieces))


# ==========================================
# Vocabulary information
# ==========================================

print("\n" + "=" * 70)
print("TOKENIZER INFORMATION")
print("=" * 70)

print("Vocabulary size:", sp.get_piece_size())

print("\nFirst 20 vocabulary pieces:")

for i in range(20):
    print(i, "→", sp.id_to_piece(i))

print("\n" + "=" * 70)
print("SUBWORD TOKENIZATION COMPLETE")
print("=" * 70)