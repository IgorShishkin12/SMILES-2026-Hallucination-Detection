# SMILES-2026 Hallucination Detection - Solution

## Reproducibility

The pipeline produces:
- `results.json` - 5-fold cross-validated metrics
- `predictions.csv` - predicted labels for the 100 held-out test samples

## Final Results

| Metric | Baseline (skeleton) | Final |
|--------|---------------------|-------|
| Test Accuracy | 70.19% | **74.14%** |
| Test F1 | 82.49% | **83.74%** |
| Test AUROC | 75.83% | **77.63%** |
| Feature dimension | 896 | 13008 |
| Extract time | 13.1 s | 14.7 s |

*Primary ranking metric - Test AUROC: 77.63%*

## Architecture Choices & Rationale

### 1. Aggregation (`aggregation.py`)

**Multi-layer fusion**
- Concatenate **last token** and **mean-pooled response tokens** from the final **4 transformer layers** (layers 21-24).
- Append **max-pool over response tokens** and **mean-pool over all real tokens** from the very last layer.
- Total: 4×896×2 + 896 + 896 = 8960 hidden-state dims.

- Rationale: Hallucination signals are distributed across late layers, not just the final one. Concatenating multiple layers gives the probe access to depth-varying representations.

**Mid-layer checkpoints**
- Additionally extract **mean** and **max** pooling of response tokens from transformer **layers 12-13** (mid-network).
- Adds 2×2×896 = 3584 dims.
- Rationale: Hallucination errors form in mid-network layers before they propagate to later layers. These features capture the model's "forming" representations and are complementary to the late-layer features.


**Response-aware pooling**
- `solution.py` tokenises each prompt separately to determine the exact prompt token length.
- Aggregation receives `response_start` and restricts pooling to response tokens only.
- Rationale: The hallucination signal lives in the generated response, not the input prompt.

**Geometric features**
- Enabled (`USE_GEOMETRIC = True`).
- Features extracted (~128 dims):
  - Sequence length & response-length ratio
  - Layer-wise L2 norms of the last token (all 24 layers)
  - Mean / std of token L2 norms per layer
  - Cosine-similarity drift between consecutive layers (last token & mean over all tokens)
  - Response-only norm statistics in the last layer
- Rationale: These capture representation drift and activation magnitude patterns that are complementary to the raw hidden-state vectors.


**Lookback-Lens attention features (336 dims)**
- For each of the 24 transformer layers and each of the 14 attention heads, compute the mean fraction of attention mass that response tokens direct toward prompt tokens.
- Total: 24 × 14 = 336 dims.
- Rationale: Hallucinated responses attend less to source context (lower lookback ratio). This is the Lookback-Lens signal from the literature.
- Requires `attn_implementation="eager"` in the model (set in `solution.py`).

**Total feature dim: 8960 + 3584 + 128 + 336 = 13008**

### 2. Probe (`probe.py`)

**Network architecture**
```
input_dim → 128 → 64 → 1
```
- BatchNorm after each hidden layer
- ReLU activations
- Dropout 0.6 after each hidden layer

**Training regime**
- Optimiser: `AdamW`, lr = 1e-3, weight_decay = 1e-2
- Scheduler: CosineAnnealingLR (T_max = 100, eta_min = 1e-5)
- Loss: `BCEWithLogitsLoss` with class-weighting (neg/pos ratio)
- Early stopping: internal 15% stratified validation split, patience = 30 epochs, monitored on **validation loss**
- Max epochs: 200

**Dimensionality reduction**
- PCA to **256 components** (fitted on the training fold only) whenever input dim > 256.
- Rationale: With only ~470 training samples per fold, reducing 13008 features to 256 keeps the MLP tractable and reduces overfitting.

### 3. Splitting (`splitting.py`)

- **Stratified 5-fold cross-validation** (`random_state=42`).
- Within each fold, the training portion is further split into train / validation (85 % / 15 %) for threshold tuning.

## Experiment Log

All experiments use 5-fold stratified CV on 689 samples. AUROC is mean test AUROC across folds.

| # | Name | Change from previous | Feature dim | Test AUROC | delta |
|---|------|----------------------|-------------|------------|---|
| 0 | Baseline | Skeleton code (4-layer hidden + geometry) | 9088 | 75.49% | 0 |
| 1 | Exp1 | Last-token only top-3 layers + compact geometry | ~2800 | 72.12% | -3.37% |
| 2 | Exp2 | Restore baseline features + add Lookback-Lens attention (336 dims) | 9424 | 76.94% | +1.45% |
| 3 | Exp3 | Attention-only features, no hidden states (no PCA) | 336 | 74.95% | -0.54% |
| 4 | Exp4 | Exp2 + mixed BCE+ranking loss (ranking_weight=0.3) | 9424 | 76.39% | -0.55% |
| 4b | Exp4b | Reduce ranking_weight to 0.1 | 9424 | 75.80% | -1.14% |
| 5 | Exp5 | Exp2 + attention entropy features (2×336 attn dims) | 9760 | 76.79% | -0.15% |
| 6 | Exp6 | Exp2 + mid-layer 12-13 mean/max response features | 13008 | 77.63% | +0.69% |
| 7 | Exp7 | Replace layers 12-13 with 4 checkpoints (6,9,12,15) | 16592 | 77.39% | -0.24% |
| 7b | Exp7b | Exp6 + PCA 512 instead of 256 | 13008 | 77.44% | -0.19% |
| 8 | Exp8 | Exp6 + early stopping on val AUROC instead of val loss | 13008 | 77.29% | -0.34% |

### Key findings

- **Attention features are additive**: Lookback-Lens attention (+1.45%) complements hidden-state features - they carry independent signal.
- **Mid-layer representations matter**: Layers 12-13 carry hallucination-specific features that late-layer representations do not fully capture.
- **Optimal layer spread is narrow**: Adding more mid-layer checkpoints (6,9,12,15) or earlier layers dilutes the signal rather than adding to it.
- **Ranking loss hurts**: BPR-style ranking loss destabilises calibration even at low weight (0.1). Pure BCE is more stable.
- **Entropy features are redundant**: Attention entropy per head adds no signal beyond the lookback ratio.
- **PCA 256 is the sweet spot**: Both 128 and 512 components perform worse on this dataset size (~470 training samples/fold).
- **BCE stopping beats AUROC stopping**: On small internal validation sets (~70 samples), AUROC estimates are too noisy for reliable early stopping.

## Files Modified

- `aggregation.py` - multi-layer fusion, response-aware pooling, geometric features, Lookback-Lens attention, mid-layer checkpoints
- `probe.py` - deeper MLP with BatchNorm/Dropout, AdamW, cosine LR schedule, early stopping, PCA, BPR ranking loss (weight=0.0 in final)
- `splitting.py` - stratified 5-fold cross-validation
- `solution.py` - `USE_GEOMETRIC = True`, `USE_ATTENTION = True`, eager attention reload, prompt-length pre-computation passed to aggregation