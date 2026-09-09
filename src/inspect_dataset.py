import pandas as pd
from pathlib import Path

# Location of our raw dataset
DATA_PATH = Path("../data/raw/pn.tsv")

# Read the TSV file
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["id", "label", "text", "annotation", "split"]
)

# Basic information
print("\n===== DATASET SHAPE =====")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== UNIQUE VALUES =====")
for column in df.columns:
    print(f"\n{column}:")
    print(df[column].unique()[:20])
    
print("\n===== SPLIT DISTRIBUTION =====")
print(df["split"].value_counts())

print("\n===== LABEL DISTRIBUTION =====")
print(df["label"].value_counts())

print("\n===== LABEL DISTRIBUTION BY SPLIT =====")
print(pd.crosstab(df["split"], df["label"]))

print("\n===== PERCENTAGE OF EACH LABEL =====")
print(df["label"].value_counts(normalize=True).mul(100).round(2))

print("\n===== PERCENTAGE OF EACH SPLIT =====")
print(df["split"].value_counts(normalize=True).mul(100).round(2))