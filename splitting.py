"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    n_splits: int = 5,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Return stratified k-fold splits.

    Each fold reserves ``1/n_splits`` of the data as the test set.
    The remaining ``(n_splits-1)/n_splits`` is further split into train
    (minus a small stratified validation hold-out) and validation.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional full DataFrame (same row order as ``y``).
        n_splits:     Number of folds (default 5).
        val_size:     Fraction of the *training* portion reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples.
    """
    idx = np.arange(len(y))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for train_val_idx, test_idx in skf.split(idx, y):
        # Further split train_val into train / val (stratified)
        if val_size > 0 and len(train_val_idx) > 10:
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size,
                random_state=random_state,
                stratify=y[train_val_idx],
            )
        else:
            train_idx = train_val_idx
            val_idx = None

        splits.append((train_idx, val_idx, test_idx))

    return splits
