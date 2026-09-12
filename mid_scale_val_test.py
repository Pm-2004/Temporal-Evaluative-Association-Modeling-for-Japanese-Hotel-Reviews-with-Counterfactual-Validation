import sys
from pathlib import Path
sys.path.append(str(Path("d:/AMGT Project")))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from src.amgt_model import AMGTForAspectEvaluation
from src.amgt_dataset import AMGTDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
base_dir = Path("d:/AMGT Project")

print("=== MID-SCALE TEST (OVERSAMPLING WITH VAL SPLIT) ===")
train_path = base_dir / "data" / "processed" / "hotel_train.csv"
train_dataset = AMGTDataset(train_path, max_length=64)

df = train_dataset.data
pos_indices = df[df['sentiment_label'] == 2].index.tolist()[:240]
neu_indices = df[df['sentiment_label'] == 1].index.tolist()[:45]
neg_indices = df[df['sentiment_label'] == 0].index.tolist()[:15]
indices = pos_indices + neu_indices + neg_indices

labels = [df.iloc[i]['sentiment_label'] for i in indices]
train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)

subset_train = torch.utils.data.Subset(train_dataset, train_idx)
subset_val = torch.utils.data.Subset(train_dataset, val_idx)

train_labels = [df.iloc[i]['sentiment_label'] for i in train_idx]
class_counts = {c: train_labels.count(c) for c in set(train_labels)}
sample_weights = [1.0 / class_counts[label] for label in train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_labels), replacement=True)

class OversamplingTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

model = AMGTForAspectEvaluation(num_labels=3, class_weights=None)

subset_trainer = OversamplingTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="tmp_mid_scale_oversample_val",
        num_train_epochs=15, 
        per_device_train_batch_size=8,
        logging_steps=50,
        report_to="none",
        save_strategy="no"
    ),
    train_dataset=subset_train,
    eval_dataset=subset_val
)

print(f"Training on {len(subset_train)} examples, validating on {len(subset_val)} examples...")
subset_trainer.train()

print("\nEvaluating mid-scale model on held-out validation set...")
preds_subset = subset_trainer.predict(subset_val)
logits = preds_subset.predictions[0] if isinstance(preds_subset.predictions, tuple) else preds_subset.predictions
if logits.shape[-1] == 4:
    logits = logits[:, :3]
pred_subset_labels = np.argmax(logits, axis=-1)
true_subset_labels = preds_subset.label_ids
subset_cm = confusion_matrix(true_subset_labels, pred_subset_labels, labels=[0, 1, 2])
print("3x3 Confusion Matrix (Validation):")
print(subset_cm)


print("\n=== FULL SCALE BATCH COMPOSITION CHECK ===")
full_labels = df['sentiment_label'].values
full_counts = df['sentiment_label'].value_counts().to_dict()
full_sample_weights = [1.0 / full_counts[label] for label in full_labels]
full_sampler = WeightedRandomSampler(weights=full_sample_weights, num_samples=len(full_labels), replacement=True)

# Custom collate function to avoid triggering full BERT tokenization just to check labels
def mock_collate(batch):
    labels = [b['labels'] for b in batch]
    return {'labels': torch.stack(labels)}

full_dataloader = DataLoader(train_dataset, batch_size=8, sampler=full_sampler, collate_fn=mock_collate)

batch_comps = {0: 0, 1: 0, 2: 0}
print("Sampling 20 batches from full DataLoader...")
for i, batch in enumerate(full_dataloader):
    if i >= 20: break
    lbls = batch['labels'].numpy()
    unique, counts = np.unique(lbls, return_counts=True)
    comp = dict(zip(unique, counts))
    print(f"Batch {i+1:02d}: {comp}")
    for k, v in comp.items():
        batch_comps[k] += v

print(f"\nTotal composition over 20 batches (160 samples): {batch_comps}")
