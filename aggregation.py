"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).
  3. ``extract_attention_features`` — Lookback-Lens attention ratios
     (enabled when attentions are passed to aggregation_and_feature_extraction).

All stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

N_LAYERS_TO_CONCAT = 4          # number of final layers to concatenate
USE_RESPONSE_POOL = True        # pool over response tokens when response_start is given


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int | None = None,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Strategy:
      * Concatenate the **last token** from the final ``N_LAYERS_TO_CONCAT``
        layers.
      * Concatenate the **mean-pooled response tokens** from the same layers.
      * Append the **max-pooled response tokens** from the *very last* layer.
      * Append the **mean-pooled all-real tokens** from the *very last* layer.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        response_start: Index of the first response token (optional).

    Returns:
        A 1-D feature tensor.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_real = real_positions.numel()
    last_pos = int(real_positions[-1].item())

    # 1. Select the final K layers (embedding = index 0, transformers = 1..)
    selected = hidden_states[-N_LAYERS_TO_CONCAT:]   # (K, seq_len, hidden_dim)

    # 2. Response token positions
    if USE_RESPONSE_POOL and response_start is not None and 0 < response_start < n_real:
        resp_positions = real_positions[real_positions >= response_start]
    else:
        resp_positions = real_positions

    # 3. Last-token features for the K selected layers
    last_token_feats = selected[:, last_pos, :]        # (K, hidden_dim)

    # 4. Mean-pool over response tokens for the K selected layers
    mean_resp_feats = selected[:, resp_positions, :].mean(dim=1)   # (K, hidden_dim)

    # 5. Max-pool over response tokens for the *last* layer only
    max_resp_last = selected[-1, resp_positions, :].max(dim=0).values   # (hidden_dim,)

    # 6. Mean-pool over ALL real tokens for the *last* layer only
    mean_all_last = selected[-1, real_positions, :].mean(dim=0)    # (hidden_dim,)

    # 7. Middle layers (transformer 12-13, indices 12-13 in hidden_states):
    #    mean and max pool over response tokens - captures mid-network representations
    mid = hidden_states[12:14]                                     # (2, seq_len, hidden_dim)
    mid_mean_resp = mid[:, resp_positions, :].mean(dim=1)          # (2, hidden_dim)
    mid_max_resp = mid[:, resp_positions, :].max(dim=1).values     # (2, hidden_dim)

    feature = torch.cat([
        last_token_feats.flatten(),      # K * hidden_dim
        mean_resp_feats.flatten(),       # K * hidden_dim
        max_resp_last,                   # hidden_dim
        mean_all_last,                   # hidden_dim
        mid_mean_resp.flatten(),         # 2 * hidden_dim
        mid_max_resp.flatten(),          # 2 * hidden_dim
    ], dim=0)

    return feature


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    response_start: int | None = None,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        response_start: Index of the first response token (optional).

    Returns:
        A 1-D float tensor of fixed length.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_real = real_positions.numel()
    last_pos = int(real_positions[-1].item())

    feats: list[float] = []

    # 1. Sequence length
    feats.append(float(n_real))

    # 2. Response length and ratio
    if response_start is not None and 0 < response_start < n_real:
        resp_len = float(n_real - response_start)
    else:
        resp_len = float(n_real)
    feats.append(resp_len)
    feats.append(resp_len / max(n_real, 1))

    # 3. Layer-wise L2 norm of the last token (all layers)
    last_token_all = hidden_states[:, last_pos, :]          # (n_layers, hidden_dim)
    layer_norms = torch.norm(last_token_all, dim=1)         # (n_layers,)
    feats.extend(layer_norms.tolist())

    # 4. Mean L2 norm across all real tokens per layer
    token_norms = torch.norm(hidden_states[:, real_positions, :], dim=2)   # (n_layers, n_real)
    mean_norms = token_norms.mean(dim=1)                     # (n_layers,)
    feats.extend(mean_norms.tolist())

    # 5. Std of L2 norms across tokens per layer
    std_norms = token_norms.std(dim=1)                       # (n_layers,)
    feats.extend(std_norms.tolist())

    # 6. Cosine similarity drift between consecutive layers (last token)
    last_token_normed = F.normalize(last_token_all, dim=1)   # (n_layers, hidden_dim)
    cos_drift_last = (last_token_normed[:-1] * last_token_normed[1:]).sum(dim=1)  # (n_layers-1,)
    feats.extend(cos_drift_last.tolist())

    # 7. Mean cosine similarity drift between consecutive layers (averaged over all real tokens)
    normalized = F.normalize(hidden_states[:, real_positions, :], dim=2)   # (n_layers, n_real, hidden_dim)
    token_cos = (normalized[:-1] * normalized[1:]).sum(dim=2)             # (n_layers-1, n_real)
    mean_token_cos = token_cos.mean(dim=1)                                # (n_layers-1,)
    feats.extend(mean_token_cos.tolist())

    # 8. Response-only statistics (if boundary known)
    if response_start is not None and 0 < response_start < n_real:
        resp_positions = real_positions[real_positions >= response_start]
        # Mean L2 norm of response tokens in last layer
        resp_norms_last = torch.norm(hidden_states[-1, resp_positions, :], dim=1)
        feats.append(float(resp_norms_last.mean()))
        feats.append(float(resp_norms_last.std()))
    else:
        feats.extend([0.0, 0.0])

    return torch.tensor(feats, dtype=hidden_states.dtype, device=hidden_states.device)


def extract_attention_features(
    attentions: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    response_start: int | None = None,
) -> torch.Tensor:
    """Lookback-Lens: per-head ratio of response attention directed at prompt tokens.

    For each transformer layer and each attention head, computes the mean
    fraction of attention mass that response tokens direct toward the prompt
    (context grounding). Hallucinated responses attend less to source context.

    Args:
        attentions:     Tuple of (num_heads, seq_len, seq_len) attention weight
                        tensors, one per transformer layer (already per-sample).
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real tokens.
        response_start: Token index where the response begins.

    Returns:
        Feature tensor of shape ``(n_layers * num_heads,)``.
    """
    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_real = real_positions.numel()
    n_layers = len(attentions)
    n_heads = attentions[0].shape[0]

    if response_start is None or response_start >= n_real:
        return torch.zeros(n_layers * n_heads, dtype=torch.float32)

    resp_positions = real_positions[real_positions >= response_start]
    if resp_positions.numel() == 0:
        return torch.zeros(n_layers * n_heads, dtype=torch.float32)

    feats: list[torch.Tensor] = []
    for attn_l in attentions:
        # attn_l: (num_heads, seq_len, seq_len) - attn_l[h, q, k] = weight from q to k
        resp_attn = attn_l[:, resp_positions, :]          # (heads, n_resp, seq_len)
        prompt_mass = resp_attn[:, :, :response_start].sum(dim=-1)    # (heads, n_resp)
        total_mass = resp_attn.sum(dim=-1).clamp(min=1e-8)            # (heads, n_resp)
        lookback = (prompt_mass / total_mass).mean(dim=1)             # (heads,)
        feats.append(lookback)

    return torch.cat(feats, dim=0).float()    # (n_layers * heads,)


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
    response_start: int | None = None,
    attentions: tuple[torch.Tensor, ...] | None = None,
    use_hidden: bool = True,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric/attention features.

    Main entry point called from ``solution.py`` for each sample.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real tokens.
        use_geometric:  Whether to append geometric features.
        response_start: Index of the first response token (optional).
        attentions:     Optional tuple of per-layer attention tensors
                        ``(num_heads, seq_len, seq_len)`` for Lookback-Lens features.
        use_hidden:     Whether to include hidden-state aggregate features.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)``.
    """
    parts = []

    if use_hidden:
        parts.append(aggregate(hidden_states, attention_mask, response_start=response_start))

    if use_geometric and use_hidden:
        parts.append(extract_geometric_features(hidden_states, attention_mask, response_start=response_start))

    if attentions is not None:
        parts.append(extract_attention_features(attentions, attention_mask, response_start=response_start))

    return torch.cat(parts, dim=0)
