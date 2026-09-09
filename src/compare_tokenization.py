import pandas as pd
import sentencepiece as spm
from fugashi import Tagger
from pathlib import Path


# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "sp_japanese.model"


# ==========================================
# Load data
# ==========================================

df = pd.read_csv(DATA_PATH)


# ==========================================
# Load tokenizers
# ==========================================

sp = spm.SentencePieceProcessor()
sp.load(str(MODEL_PATH))

tagger = Tagger()


# ==========================================
# Calculate token counts
# ==========================================

character_counts = []
subword_counts = []
word_counts = []


for text in df["text"].astype(str):

    # Character tokens
    characters = list(text)

    # Subword tokens
    subwords = sp.encode(
        text,
        out_type=str
    )

    # Word/morphological tokens
    words = [word.surface for word in tagger(text)]

    character_counts.append(len(characters))
    subword_counts.append(len(subwords))
    word_counts.append(len(words))


# ==========================================
# Add statistics to dataframe
# ==========================================

df["character_tokens"] = character_counts
df["subword_tokens"] = subword_counts
df["word_tokens"] = word_counts


# ==========================================
# Print results
# ==========================================

print("=" * 70)
print("TOKENIZATION COMPARISON")
print("=" * 70)

print("\nNumber of training examples:", len(df))

print("\n--- Average Token Counts ---")

print(
    f"Character : {df['character_tokens'].mean():.2f}"
)

print(
    f"Subword   : {df['subword_tokens'].mean():.2f}"
)

print(
    f"Word      : {df['word_tokens'].mean():.2f}"
)


print("\n--- Median Token Counts ---")

print(
    f"Character : {df['character_tokens'].median():.2f}"
)

print(
    f"Subword   : {df['subword_tokens'].median():.2f}"
)

print(
    f"Word      : {df['word_tokens'].median():.2f}"
)


print("\n--- Minimum Token Counts ---")

print(
    f"Character : {df['character_tokens'].min()}"
)

print(
    f"Subword   : {df['subword_tokens'].min()}"
)

print(
    f"Word      : {df['word_tokens'].min()}"
)


print("\n--- Maximum Token Counts ---")

print(
    f"Character : {df['character_tokens'].max()}"
)

print(
    f"Subword   : {df['subword_tokens'].max()}"
)

print(
    f"Word      : {df['word_tokens'].max()}"
)


# ==========================================
# Compression ratios
# ==========================================

print("\n--- Average Compression Ratios ---")

print(
    f"Subword / Character : "
    f"{df['subword_tokens'].mean() / df['character_tokens'].mean():.3f}"
)

print(
    f"Word / Character    : "
    f"{df['word_tokens'].mean() / df['character_tokens'].mean():.3f}"
)


print("\n" + "=" * 70)
print("TOKENIZATION COMPARISON COMPLETE")
print("=" * 70)