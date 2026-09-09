import sentencepiece as spm
from pathlib import Path


# ==========================================
# Paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_TEXT = PROJECT_ROOT / "data" / "processed" / "sp_train.txt"
MODEL_PREFIX = PROJECT_ROOT / "data" / "processed" / "sp_japanese"


# ==========================================
# Train SentencePiece
# ==========================================

spm.SentencePieceTrainer.train(
    input=str(TRAIN_TEXT),
    model_prefix=str(MODEL_PREFIX),

    # Vocabulary size
    vocab_size=1600,

    # Unigram language model
    model_type="unigram",

    # Special tokens
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,

    # Character coverage
    character_coverage=1.0,

    # Random seed / reproducibility
    seed_sentencepiece_size=1000000
)


print("=" * 60)
print("SENTENCEPIECE MODEL TRAINED")
print("=" * 60)

print(f"\nTraining data:")
print(TRAIN_TEXT)

print("\nVocabulary size: 1600")

print("\nFiles created:")
print(MODEL_PREFIX.with_suffix(".model"))
print(MODEL_PREFIX.with_suffix(".vocab"))

print("\n" + "=" * 60)