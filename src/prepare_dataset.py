import pandas as pd
from pathlib import Path

# ----------------------------------------
# File locations
# ----------------------------------------

RAW_PATH = Path("../data/raw/pn.tsv")
PROCESSED_DIR = Path("../data/processed")

# Create processed folder if it doesn't exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------
# Load the raw dataset
# ----------------------------------------

df = pd.read_csv(
    RAW_PATH,
    sep="\t",
    header=None,
    names=["id", "label", "text", "annotation", "split"]
)

# ----------------------------------------
# Keep only the columns needed
# ----------------------------------------

df = df[["id", "label", "text", "split"]].copy()

# ----------------------------------------
# Convert labels for model training
#
# Original:
# -1 = Negative
#  0 = Neutral
#  1 = Positive
#
# New:
# 0 = Negative
# 1 = Neutral
# 2 = Positive
# ----------------------------------------

label_mapping = {
    -1: 0,
     0: 1,
     1: 2
}

df["label"] = df["label"].map(label_mapping)

# ----------------------------------------
# Separate train / dev / test
# ----------------------------------------

train_df = df[df["split"] == "train"].copy()
dev_df = df[df["split"] == "dev"].copy()
test_df = df[df["split"] == "test"].copy()

# ----------------------------------------
# Save processed files
# ----------------------------------------

train_df.to_csv(
    PROCESSED_DIR / "train.csv",
    index=False
)

dev_df.to_csv(
    PROCESSED_DIR / "dev.csv",
    index=False
)

test_df.to_csv(
    PROCESSED_DIR / "test.csv",
    index=False
)

# ----------------------------------------
# Display result
# ----------------------------------------

print("=" * 50)
print("PROCESSED DATASET CREATED")
print("=" * 50)

print(f"\nTrain examples: {len(train_df)}")
print(f"Dev examples:   {len(dev_df)}")
print(f"Test examples:  {len(test_df)}")

print("\nLabel mapping:")
print("0 = Negative")
print("1 = Neutral")
print("2 = Positive")

print("\nFiles created:")
print(PROCESSED_DIR / "train.csv")
print(PROCESSED_DIR / "dev.csv")
print(PROCESSED_DIR / "test.csv")

print("\n" + "=" * 50)