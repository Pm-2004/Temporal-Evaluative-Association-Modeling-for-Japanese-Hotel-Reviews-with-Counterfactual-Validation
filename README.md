# Temporal Evaluative Association Modeling for Japanese Hotel Reviews with Counterfactual Validation (AMGT)

## Overview
This project implements an **Aspect-based Sentiment Evaluation System** for Japanese hotel reviews using a novel **AMGT (Asymmetric Multi-Granularity Tokenization)** approach. The goal is to accurately predict sentiment across 8 distinct hotel aspects (e.g., room, hospitality, location, cleanliness) by fusing multiple levels of linguistic representation.

## Architecture & AMGT Model
The core of this project is the `AMGTForAspectEvaluation` model which employs a **tri-granular tokenization strategy** specifically designed for the complexities of the Japanese language:
1. **Subword/Contextual Level:** Uses `cl-tohoku/bert-base-japanese-v3` to capture deep semantic context.
2. **Morphological Level:** Uses `Fugashi` (MeCab wrapper) to extract structural tokens and grammatical boundaries.
3. **Character Level:** Captures individual logographic (Kanji) and phonetic (Hiragana/Katakana) traits using a custom character embedding space.

These three granularities are rigorously aligned to a fixed max sequence length and fused using feature concatenations, standard convolutions (CNNs), and max pooling to produce robust aspect-sentiment predictions.

## Dataset & Preprocessing
The model operates on a custom scraped dataset of Japanese hotel reviews spanning roughly ~160k aspect evaluation rows across 25 distinct hotels. 

### Data Preparation Pipeline
1. **Extraction & Parsing:** Scraping and cleaning raw review comments (`extract_review.py`).
2. **Aspect Filtering:** Isolating labels mapped to 8 fixed categories: `room`, `hospitality`, `location`, `bath`, `facility`, `cleanliness`, `breakfast`, `dinner`.
3. **Leak-Proof Splits:** `create_splits.py` partitions the dataset into Train/Validation/Test splits grouped safely by **hotel** rather than just by row, preventing stylistic data leakage.
4. **Tokenization:** Custom `SentencePiece` models alongside BERT embeddings are prepared for morphological analysis.

## Training
- **Framework:** Hugging Face `Trainer` integrated with PyTorch.
- **Script:** `src/train.py`
- **Techniques:** 
  - **Handling Class Imbalance (16:1 Skew):** The dataset is heavily skewed toward positive reviews (~82%). Initial attempts using static class-weighted CrossEntropyLoss resulted in model collapse (always predicting the majority class) due to infrequent exposure to the minority classes per mini-batch. 
  - **OversamplingTrainer:** To solve this, a custom `OversamplingTrainer` subclass overrides the training dataloader to inject a PyTorch `WeightedRandomSampler`. This ensures every mini-batch dynamically balances the negative, neutral, and positive examples using inverse-frequency weights computed from the full 162K training set.
  - **Model Selection:** `macro_f1` is utilized as the primary metric (`metric_for_best_model`) to prevent minority class overfitting, ensuring that the final saved model genuinely discriminates between all three classes.

## Ongoing Work & Updates
- **Baseline Achieved:** The model successfully trains across 3 epochs without collapsing, achieving a baseline validation macro F1 of ~0.44. It demonstrates genuine multi-class generalization on held-out hotel datasets.
- **Counterfactual Validation:** Upcoming rigorous validations to assess model reliability on stylistic perturbations.
- **Temporal Analysis:** Evaluating review trends across Year-Month thresholds (handled in `temporal_analysis.py`).
- **Aspect Profiling:** Next steps involve profiling per-aspect performance (e.g., `cleanliness`, `bath`) to decide between iterating on the architecture or moving forward with change-point detection.

*This README is an active, living document and will be updated as the pipeline evolves.*

## Project File Structure

```text
AMGT Project
├── data
│   ├── collected/          # Raw scraped JSONL review files for various hotels
│   ├── processed/          # Processed CSVs, leak-proof splits, and SentencePiece models
│   ├── rakuten_raw/        # Rakuten specific raw reviews
│   └── raw/                # Base raw datasets
├── models
│   └── amgt_aspect_eval/   # Saved PyTorch checkpoints for the AMGT model
├── results                 # Output CSVs like temporal aspect trends
├── figures                 # Plots and visualizations (e.g., temporal trends)
├── scripts
│   ├── collect_pages.py
│   └── preprocess/
│       └── build_research_corpus.py
├── src
│   ├── amgt_dataset.py                 # PyTorch Dataset for AMGT tokenization
│   ├── amgt_model.py                   # AMGT tri-granular neural network
│   ├── build_aspect_corpus.py          # Data ingestion script
│   ├── create_splits.py                # Leak-proof hotel-level dataset splitting
│   ├── data_quality_check.py           # Aspect and label validation
│   ├── temporal_analysis.py            # Extracts Year-Month aggregated trends
│   ├── train.py                        # Hugging Face Trainer execution script
│   ├── train_sentencepiece.py          # SP model training wrapper
│   └── verify_processed_assets.py      # Post-processing data integrity validation
├── EXTRA                               # External research papers and reference corpora (e.g., JRTE)
├── experiments                         # Scratchpad for modeling experiments
└── notebooks                           # Jupyter notebooks for interactive analysis
```
