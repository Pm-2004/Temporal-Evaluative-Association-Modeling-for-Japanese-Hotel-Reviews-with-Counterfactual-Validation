import pandas as pd
from pathlib import Path

# ----------------------------------------
# Paths
# ----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sp_train.txt"

# ----------------------------------------
# Load training data
# ----------------------------------------

df = pd.read_csv(TRAIN_PATH)

# ----------------------------------------
# Extract Japanese text
# ----------------------------------------

texts = df["text"].astype(str)

# ----------------------------------------
# Save training text
# ----------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")

print("=" * 50)
print("SENTENCEPIECE TRAINING TEXT CREATED")
print("=" * 50)

print(f"Number of sentences: {len(texts)}")
print(f"Output file: {OUTPUT_PATH}")

print("=" * 50)