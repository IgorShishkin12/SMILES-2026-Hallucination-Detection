"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a binary MLP that classifies feature
vectors as truthful (0) or hallucinated (1).  Called from ``solution.py``
via ``evaluate.run_evaluation``.  All four public methods (``fit``,
``fit_hyperparameters``, ``predict``, ``predict_proba``) must be implemented
and their signatures must not change.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module``; the default architecture is a single
    hidden-layer MLP with ``StandardScaler`` pre-processing.  The network is
    built lazily in ``fit()`` once the feature dimension is known.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.6,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_epochs: int = 200,
        patience: int = 30,
        es_val_size: float = 0.15,
        scheduler_tmax: int = 100,
        use_pca: bool = True,
        pca_components: int = 256,
    ) -> None:
        super().__init__()
        self._net: nn.Sequential | None = None  # built lazily in fit()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.es_val_size = es_val_size
        self.scheduler_tmax = scheduler_tmax
        self.use_pca = use_pca
        self.pca_components = pca_components

    # ------------------------------------------------------------------
    # Network builder
    # ------------------------------------------------------------------
    def _build_network(self, input_dim: int) -> None:
        """Instantiate the network layers.

        Called once at the start of ``fit()`` when ``input_dim`` is known.

        Args:
            input_dim: Feature vector dimensionality.
        """
        layers: list[nn.Module] = []
        dims = [input_dim, *self.hidden_dims, 1]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # not the final output layer
                layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout))
        self._net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass — returns raw logits of shape ``(n_samples,)``.

        Args:
            x: Float tensor of shape ``(n_samples, feature_dim)``.

        Returns:
            1-D tensor of raw (pre-sigmoid) logits.
        """
        if self._net is None:
            raise RuntimeError(
                "Network has not been built yet. Call fit() before forward()."
            )
        return self._net(x).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors with early stopping.

        Scales features with ``StandardScaler``, builds the network if needed,
        and optimises with Adam + ``BCEWithLogitsLoss``.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_scaled = self._scaler.fit_transform(X)

        # Optional PCA for dimensionality reduction
        if self.use_pca and X_scaled.shape[1] > self.pca_components:
            n_comp = min(self.pca_components, X_scaled.shape[0] - 1, X_scaled.shape[1])
            self._pca = PCA(n_components=n_comp, random_state=42)
            X_scaled = self._pca.fit_transform(X_scaled)

        # Internal train / early-stopping split (stratified)
        if self.es_val_size > 0 and len(y) > 20:
            X_tr, X_es, y_tr, y_es = train_test_split(
                X_scaled,
                y,
                test_size=self.es_val_size,
                random_state=42,
                stratify=y,
            )
        else:
            X_tr, y_tr = X_scaled, y
            X_es, y_es = X_scaled, y

        self._build_network(X_tr.shape[1])

        X_tr_t = torch.from_numpy(X_tr).float()
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32))
        X_es_t = torch.from_numpy(X_es).float()
        y_es_t = torch.from_numpy(y_es.astype(np.float32))

        # Class-weighted loss
        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.scheduler_tmax, eta_min=self.lr * 1e-2
        )

        best_loss = float("inf")
        best_state: dict | None = None
        epochs_no_improve = 0

        self.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            logits = self(X_tr_t)
            loss = criterion(logits, y_tr_t)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Early-stopping check on validation loss
            self.eval()
            with torch.no_grad():
                es_logits = self(X_es_t)
                es_loss = criterion(es_logits, y_es_t).item()

                if es_loss < best_loss:
                    best_loss = es_loss
                    best_state = copy.deepcopy(self.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        break
            self.train()

        # Restore best weights
        if best_state is not None:
            self.load_state_dict(best_state)

        self.eval()
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]

        # Candidate thresholds: unique predicted probabilities plus a coarse grid.
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            score = f1_score(y_val, y_pred_t, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        X_scaled = self._scaler.transform(X)
        if self._pca is not None:
            X_scaled = self._pca.transform(X_scaled)
        X_t = torch.from_numpy(X_scaled).float()
        with torch.no_grad():
            logits = self(X_t)
            prob_pos = torch.sigmoid(logits).numpy()
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)

