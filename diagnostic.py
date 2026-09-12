import sys
from pathlib import Path
sys.path.append(str(Path("d:/AMGT Project")))

import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import Trainer
from src.amgt_model import AMGTForAspectEvaluation
from src.amgt_dataset import AMGTDataset

base_dir = Path("d:/AMGT Project")

print("\n=== DIAGNOSTIC 1: Confusion Matrix on Full Validation Set ===")
val_path = base_dir / "data" / "processed" / "hotel_val.csv"
val_dataset = AMGTDataset(val_path, max_length=64)
model_path = base_dir / "models" / "amgt_aspect_eval" / "best_model"

model = AMGTForAspectEvaluation(num_labels=3, class_weights=None)
try:
    model.load_state_dict(torch.load(model_path / "pytorch_model.bin", map_location="cpu", weights_only=True))
except Exception as e:
    print(f"Failed to load checkpoint: {e}")
    sys.exit(1)

model.eval()
trainer = Trainer(model=model, eval_dataset=val_dataset)
preds = trainer.predict(val_dataset)

logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
if logits.shape[-1] == 4:
    logits = logits[:, :3]
pred_labels = np.argmax(logits, axis=-1)
true_labels = preds.label_ids

cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1, 2])
print("3x3 Confusion Matrix:")
print(cm)
