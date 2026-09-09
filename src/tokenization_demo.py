import pandas as pd
import sentencepiece as spm
from fugashi import Tagger
from pathlib import Path


# ==========================================
# 1. Load one Japanese sentence
# ==========================================

# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to processed training data
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"

df = pd.read_csv(DATA_PATH)

# ==========================================
# Show several real Japanese examples
# ==========================================

tagger = Tagger()

print("=" * 70)
print("10 JAPANESE TRAINING EXAMPLES")
print("=" * 70)

for i in range(10):

    text = str(df.iloc[i]["text"])

    # Character-level tokens
    characters = list(text)

    # Word / morphological tokens
    words = [word.surface for word in tagger(text)]

    print(f"\nExample {i + 1}")
    print("-" * 70)

    print("Original:")
    print(text)

    print("\nCharacters:")
    print(" | ".join(characters))

    print("\nWords:")
    print(" | ".join(words))

print("=" * 60)
print("ORIGINAL JAPANESE TEXT")
print("=" * 60)
print(text)


# ==========================================
# 2. CHARACTER-LEVEL REPRESENTATION
# ==========================================

characters = list(text)

print("\n" + "=" * 60)
print("CHARACTER-LEVEL TOKENS")
print("=" * 60)

print(characters)


# ==========================================
# 3. WORD / MORPHOLOGICAL REPRESENTATION
# ==========================================

tagger = Tagger()

words = [word.surface for word in tagger(text)]

print("\n" + "=" * 60)
print("WORD / MORPHOLOGICAL TOKENS")
print("=" * 60)

print(words)


# ==========================================
# 4. SUBWORD TOKENIZATION
# ==========================================
#
# IMPORTANT:
# We need a trained SentencePiece model.
# We will create one from our Japanese
# training data in the next step.
#
# For now we only check that SentencePiece
# is installed correctly.
# ==========================================

print("\n" + "=" * 60)
print("SUBWORD TOKENIZATION")
print("=" * 60)

print("SentencePiece installation: OK")
print("We will train our Japanese SentencePiece")
print("tokenizer in the next step.")


print("\n" + "=" * 70)
print("TOKENIZATION DEMO COMPLETE")
print("=" * 70)