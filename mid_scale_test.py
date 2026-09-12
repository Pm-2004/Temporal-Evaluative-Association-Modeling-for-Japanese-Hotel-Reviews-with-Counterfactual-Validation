import sys
from pathlib import Path
sys.path.append(str(Path("d:/AMGT Project")))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import Trainer, TrainingArguments
from src.amgt_model import AMGTForAspectEvaluation
from src.amgt_dataset import AMGTDataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)
base_dir = Path("d:/AMGT Project")

print("=== MID-SCALE TEST (OVERSAMPLING) ===")
train_path = base_dir / "data" / "processed" / "hotel_train.csv"
train_dataset = AMGTDataset(train_path, max_length=64)

df = train_dataset.data
pos_indices = df[df['sentiment_label'] == 2].index.tolist()[:240]
neu_indices = df[df['sentiment_label'] == 1].index.tolist()[:45]
neg_indices = df[df['sentiment_label'] == 0].index.tolist()[:15]
indices = pos_indices + neu_indices + neg_indices

subset = torch.utils.data.Subset(train_dataset, indices)

# Calculate sample weights for oversampling (inversely proportional to class frequencies in the subset)
subset_labels = [train_dataset.data.iloc[i]['sentiment_label'] for i in indices]
class_counts = {c: subset_labels.count(c) for c in set(subset_labels)}
sample_weights = [1.0 / class_counts[label] for label in subset_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(subset_labels), replacement=True)

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

# Remove loss-level class weights as requested
model = AMGTForAspectEvaluation(num_labels=3, class_weights=None)

subset_trainer = OversamplingTrainer(
    model=model,
    args=TrainingArguments(
        output_dir="tmp_mid_scale_oversample",
        num_train_epochs=15, 
        per_device_train_batch_size=8,
        logging_steps=50,
        report_to="none",
        save_strategy="no"
    ),
    train_dataset=subset
)

print("Training mid-scale model with WeightedRandomSampler...")
subset_trainer.train()

print("Evaluating mid-scale model on the same 300 examples...")
preds_subset = subset_trainer.predict(subset)
logits = preds_subset.predictions[0] if isinstance(preds_subset.predictions, tuple) else preds_subset.predictions
if logits.shape[-1] == 4:
    logits = logits[:, :3]
pred_subset_labels = np.argmax(logits, axis=-1)
true_subset_labels = preds_subset.label_ids
subset_cm = confusion_matrix(true_subset_labels, pred_subset_labels, labels=[0, 1, 2])
print("3x3 Confusion Matrix:")
print(subset_cm)
