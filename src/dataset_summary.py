import pandas as pd
from pathlib import Path

# Path to the raw JRTE dataset
DATA_PATH = Path("../data/raw/pn.tsv")

# Load the dataset
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["id", "label", "text", "annotation", "split"]
)

print("=" * 50)
print("JRTE DATASET SUMMARY")
print("=" * 50)

# Total number of examples
print(f"\nTotal examples: {len(df)}")

# Dataset splits
print("\n--- Dataset Splits ---")
split_counts = df["split"].value_counts()

for split, count in split_counts.items():
    percentage = count / len(df) * 100
    print(f"{split:5s}: {count:4d} ({percentage:.2f}%)")

# Labels
print("\n--- Labels ---")
label_counts = df["label"].value_counts().sort_index()

for label, count in label_counts.items():
    percentage = count / len(df) * 100
    print(f"{label:2d}: {count:4d} ({percentage:.2f}%)")

# Label distribution by split
print("\n--- Labels by Split ---")
print(pd.crosstab(df["split"], df["label"]))

# Text length statistics
df["char_length"] = df["text"].astype(str).str.len()

print("\n--- Text Length (Characters) ---")
print(f"Minimum : {df['char_length'].min()}")
print(f"Maximum : {df['char_length'].max()}")
print(f"Average : {df['char_length'].mean():.2f}")
print(f"Median  : {df['char_length'].median():.2f}")

# Empty text check
empty_text = (df["text"].astype(str).str.strip() == "").sum()

print("\n--- Text Quality ---")
print(f"Empty texts: {empty_text}")

print("\n" + "=" * 50)
print("SUMMARY COMPLETE")
print("=" * 50)