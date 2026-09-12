# AMGT Model: Class Imbalance Debugging & Training Summary

This document summarizes the investigation, diagnosis, and resolution of a major training pathology (model collapse) encountered during the initial full-scale training of the AMGT (Asymmetric Multi-Granularity Tokenization) model.

## 1. The Initial Problem: Model Collapse
The training dataset consists of ~162,000 aspect-level Japanese hotel reviews, which are heavily skewed towards positive sentiment (roughly 16:1 ratio):
- **Positive:** ~82.0%
- **Neutral:** ~13.4%
- **Negative:** ~4.6%

During initial full-scale training, the model experienced a total **majority-class collapse**. It learned to exclusively predict the "Positive" class for every single input. 
- **Symptoms:** `eval_macro_f1` and `eval_accuracy` were mathematically identical and completely flat across all 3 epochs.
- **Confusion Matrix:** All validation predictions landed in the "Positive" column, with zero predictions for Negative or Neutral.

## 2. Investigation & False Positives
Initial diagnostics revealed that the inverse-frequency class weights tensor passed to the `CrossEntropyLoss` was residing on the CPU, while the model and logits were on the GPU. 

We initially hypothesized that this device mismatch was causing a silent fallback to unweighted loss. However, after properly registering the class weights as a PyTorch buffer within the model (`self.register_buffer`), the model *still* collapsed under full-scale training conditions. 

## 3. The Root Cause: Frequency vs. Magnitude
The true root cause was a fundamental limitation of static class weighting in extreme batch-level imbalance.
- With a batch size of 8 and only ~5% negative examples, the vast majority of mini-batches contained **zero** minority examples.
- While static class weights amplify the *magnitude* of the gradient when a minority example finally appears, they do not fix the *frequency*. The rare gradient spikes were simply diluted and overwhelmed by thousands of consecutive "Positive-only" mini-batches, trapping the model in a local minimum where predicting "Positive" was the safest path to lowering the loss.

## 4. The Solution: Dynamic Oversampling
To ensure the model consistently encountered minority examples, we replaced static loss weighting with **dynamic oversampling** at the DataLoader level.

**Implementation Details:**
1. **Inverse-Frequency Sample Weights:** We computed weights for all 162K rows in the training set based on their exact class frequency (`1.0 / total_class_count`).
2. **`WeightedRandomSampler`:** We initialized a PyTorch `WeightedRandomSampler` with these sample weights to construct batches.
3. **`OversamplingTrainer`:** Because we were using the Hugging Face `Trainer`, we created a custom subclass that specifically overrides `get_train_dataloader` to inject the sampler. 
4. **Validation Integrity:** Crucially, we did *not* override `get_eval_dataloader`. The validation set continues to use a standard sequential sampler, ensuring that our `eval_macro_f1` reflects real-world, naturally-skewed performance rather than an artificially balanced one.
5. **Disabled Loss Weights:** To prevent double-correction (which would overly penalize the majority class), we removed the static class weights from the model's loss function.

## 5. Verification & Final Results
Before committing to another 2.5-hour run, we verified the sampler via a mid-scale test (160 samples), which confirmed near-perfectly balanced batches (e.g., 52 Negative, 53 Neutral, 55 Positive).

**The Full-Scale 3-Epoch Run proved highly successful:**
- **Monotonic F1 Growth:** The `eval_macro_f1` steadily climbed across the epochs (`0.4258 -> 0.4339 -> 0.4388`), proving the model was actively learning decision boundaries.
- **Accuracy Trade-off:** Overall `eval_accuracy` correctly dropped from ~73% to ~68%. The model stopped safely guessing the majority class and started making riskier, nuanced predictions.
- **Genuine Multi-Class Discrimination:** The final 3x3 confusion matrix on the 5,924-example held-out validation set demonstrated true discrimination:
  - It successfully caught ~38% of all true Negatives (49/130).
  - It successfully caught ~46% of all true Neutrals (298/647).
  - It retained ~70% recall on the Positives (3622/5147).

## 6. Next Steps
The model has successfully escaped majority-class collapse, establishing a solid ~0.44 macro F1 baseline on a very difficult task. The immediate next steps involve **Aspect Profiling**: evaluating the per-aspect macro F1 scores (specifically on low-volume aspects like `cleanliness` and `bath`) to determine whether further architectural iteration is needed before proceeding to temporal change-point detection.
