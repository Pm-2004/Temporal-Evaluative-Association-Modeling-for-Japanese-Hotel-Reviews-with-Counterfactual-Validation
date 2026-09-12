import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from src.amgt_model import AMGTForAspectEvaluation
from src.amgt_dataset import AMGTDataset, ASPECT_MAP
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
import argparse

from torch.utils.data import WeightedRandomSampler, DataLoader
# (AMGTTrainer subclass removed, class weights now handled natively by model)
def compute_metrics(eval_pred):
    """
    Computes global Accuracy and Macro F1, and provides a per-aspect breakdown
    of Macro F1 to ensure minority aspects aren't being silently masked.
    """
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
        
    preds = np.array(preds)
    
    # Check if flattened and reshape
    if len(preds.shape) == 1:
        if preds.shape[0] % 4 == 0:
            preds = preds.reshape(-1, 4)
        elif preds.shape[0] % 3 == 0:
            preds = preds.reshape(-1, 3)
            
    # Extract logits and aspect_idx
    if preds.shape[-1] == 4:
        logits = preds[:, :3]
        aspect_ids = preds[:, 3]
    else:
        logits = preds[:, :3]
        aspect_ids = None
        
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    
    # Global metrics
    metrics = {
        "accuracy": (predictions == labels).mean(),
        "macro_f1": f1_score(labels, predictions, average='macro')
    }
    
    # Per-aspect Macro F1 breakdown
    idx_to_aspect = {v: k for k, v in ASPECT_MAP.items()}
    if aspect_ids is not None:
        for aspect_id in np.unique(aspect_ids):
            mask = (aspect_ids == aspect_id)
            if mask.sum() > 0:
                aspect_name = idx_to_aspect.get(aspect_id, str(aspect_id))
                metrics[f"macro_f1_{aspect_name}"] = f1_score(
                    labels[mask], predictions[mask], average='macro'
                )
    return metrics

def print_split_stats(df, name):
    counts = df['sentiment_label'].value_counts().sort_index()
    total = len(df)
    hotels = df['hotel_name'].nunique()
    print(f"\n{name} SPLIT ({total} rows across {hotels} hotels):")
    print(f"  0 (Negative): {counts.get(0,0)} ({counts.get(0,0)/total:.1%})")
    print(f"  1 (Neutral):  {counts.get(1,0)} ({counts.get(1,0)/total:.1%})")
    print(f"  2 (Positive): {counts.get(2,0)} ({counts.get(2,0)/total:.1%})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-test', action='store_true', help='Run a quick smoke test on a subset of data')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    train_path = base_dir / "data" / "processed" / "hotel_train.csv"
    val_path = base_dir / "data" / "processed" / "hotel_val.csv"
    
    print("=" * 60)
    print("AMGT ASPECT EVALUATION TRAINING")
    print("=" * 60)
    
    print("\nLoading datasets...")
    train_dataset = AMGTDataset(train_path, max_length=64)
    val_dataset = AMGTDataset(val_path, max_length=64)
    
    if args.smoke_test:
        print("\n[SMOKE TEST] Trimming datasets to 200 rows and setting epochs to 1.")
        train_dataset.data = train_dataset.data.head(200)
        val_dataset.data = val_dataset.data.head(200)
        num_epochs = 1
    else:
        num_epochs = 3
    
    # Print Split Statistics to explicitly monitor small-N variance
    print_split_stats(train_dataset.data, "TRAIN")
    print_split_stats(val_dataset.data, "VAL")
    
    # 1. Sample Weights Calculation (Inverse Frequency)
    print("\nCalculating sample weights for oversampling from training set...")
    df = train_dataset.data
    counts = df['sentiment_label'].value_counts().to_dict()
    
    train_labels = df['sentiment_label'].values
    sample_weights = [1.0 / counts[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(train_labels), 
        replacement=True
    )
    
    # 2. Model Initialization
    print("\nInitializing AMGT Model (no loss weights)...")
    model = AMGTForAspectEvaluation(num_labels=3, class_weights=None)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(base_dir / "models" / "amgt_aspect_eval"),
        eval_strategy="epoch",      # Eval at the end of each epoch
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        fp16=True,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1", # Model selection on macro_f1
        greater_is_better=True,
        logging_steps=100,
        report_to="none",
        remove_unused_columns=False,
        save_safetensors=False
    )
    
    # 4. Trainer Initialization
    class OversamplingTrainer(Trainer):
        def get_train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    trainer = OversamplingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print("\nStarting training loop...")
    trainer.train()
    
    print("\nTraining complete. Saving best model...")
    trainer.save_model(str(base_dir / "models" / "amgt_aspect_eval" / "best_model"))
    print("Done!")

if __name__ == "__main__":
    main()
