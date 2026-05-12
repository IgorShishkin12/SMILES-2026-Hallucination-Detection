# SMILES-2026 Hallucination Detection - Solution

## Reproducibility

The pipeline produces:
- `results.json` - 5-fold cross-validated metrics
- `predictions.csv` - predicted labels for the 100 held-out test samples

## Final Results

| Metric | Baseline (skeleton) | Final |
|--------|---------------------|-------|
| Test Accuracy | 70.19% | **74.02%** |
| Test F1 | 82.49% | **83.89%** |
| Test AUROC | 75.83% | **76.54%** |
| Feature dimension | 896 | 9 088 |
| Extract time | 13.1 s | 14.7 s |

*Primary ranking metric - Test AUROC: 76.54%*

## Architecture Choices & Rationale

### 1. Aggregation (`aggregation.py`)

**Multi-layer fusion**
- We concatenate the **last token** from the final **4 transformer layers**.
- Rationale: Hallucination signals are distributed across late layers, not just the final one. Concatenating multiple layers gives the probe access to depth-varying representations.

**Token-aware pooling**
- `solution.py` tokenises each prompt separately to determine the exact prompt length.
- The aggregation receives `response_start` and computes:
  - **Mean-pool over response tokens** (per layer)
  - **Max-pool over response tokens** (last layer only)
  - **Mean-pool over all real tokens** (last layer only)
- Rationale: The hallucination signal lives in the model’s generated response, not the user prompt. Isolating response tokens sharpens the feature vector.

**Geometric features**
- Enabled (`USE_GEOMETRIC = True`).
- Features extracted (~128 dims):
  - Sequence length & response-length ratio
  - Layer-wise L2 norms of the last token
  - Mean / std of token L2 norms per layer
  - Cosine-similarity drift between consecutive layers (last token & mean over all tokens)
  - Response-only norm statistics in the last layer
- Rationale: These capture representation drift and activation magnitude patterns that are complementary to the raw hidden-state vectors.

### 2. Probe (`probe.py`)

**Network architecture**
```
input_dim → 128 → 64 → 1
```
- BatchNorm after each hidden layer
- ReLU activations
- Dropout 0.6 after each hidden layer

**Training regime**
- Optimiser: `AdamW`, lr = 1e‑3, weight_decay = 1e‑2
- Scheduler: CosineAnnealingLR (T_max = 100, eta_min = 1e‑5)
- Loss: `BCEWithLogitsLoss` with class weighting (neg/pos ratio)
- Early stopping: internal 15 % stratified validation split, patience = 30 epochs, monitored on **validation loss**
- Max epochs: 200

**Dimensionality reduction**
- PCA to **256 components** (fitted on the training fold only) is applied before the MLP whenever the input dimension exceeds 256.
- Rationale: With only ~470 training samples per fold, reducing 9 088 raw features to 256 keeps the MLP tractable and reduces overfitting.

### 3. Splitting (`splitting.py`)

- **Stratified 5-fold cross-validation** (`random_state=42`).
- Within each fold, the training portion is further split into train / validation (85 % / 15 %) for threshold tuning.
- Rationale: 5-fold gives a more reliable generalisation estimate than a single hold-out and exposes variance across data subsets.

## Hyperparameter Search

A systematic ablation was run over 9 configurations varying:
- Number of concatenated layers: 1, 2, 4
- Geometric features: ON / OFF
- Probe depth & width: linear, (128,64), (256,128), (512,256,128)
- Regularisation strength: dropout 0.4 - 0.6, weight_decay 1e‑3 - 1e‑2
- PCA: none, 256, 512 components

**Top 3 ablation results (mean 5-fold test AUROC):**

| Rank | Config | Test AUROC |
|------|--------|------------|
| 1 | 4 layers, geo=ON, probe=(128,64), dropout=0.6, wd=1e‑2, PCA=256 | **0.7616 +- 0.0364** |
| 2 | 4 layers, geo=OFF, same probe | 0.7567 +- 0.0373 |
| 3 | 4 layers, geo=OFF, linear probe, wd=1e‑1 | 0.7534 +- 0.0370 |

The winning configuration (rank 1) was locked in as the final model. The geometric features add a small but consistent gain (~0.5 % AUROC).

## Log of Failed / Discarded Experiments

| Experiment | Why it failed / was discarded |
|------------|------------------------------|
| Very deep MLP `(512, 256, 128)` + no PCA | Severe overfitting: train AUROC ~ 97 %, test AUROC ~ 75 %. Too many parameters for 689 samples. |
| 2 concatenated layers | Lower mean test AUROC (~74.6 %) than 4 layers. Less depth information. |
| 1 concatenated layer, no PCA | Test AUROC dropped to 72.5 %. Single-layer last-token features are too coarse. |
| Early stopping on AUROC | More noisy than loss-based ES on small internal validation sets (~70 samples). |

## Files Modified

- `aggregation.py` - multi-layer fusion, response-aware pooling, geometric features
- `probe.py` - deeper MLP with BatchNorm/Dropout, AdamW, cosine LR schedule, early stopping, optional PCA
- `splitting.py` - stratified 5-fold cross-validation
- `solution.py` - `USE_GEOMETRIC = True`, prompt-length pre-computation passed to aggregation
