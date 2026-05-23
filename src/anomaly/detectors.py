"""Anomaly detectors over journal-entry features.

Three independent detectors and an ensemble that fuses their scores:

* :class:`IsolationForestDetector` — scikit-learn baseline.
* :class:`AutoencoderDetector` — small 3-layer MLP in PyTorch, reconstruction
  error is the anomaly score.
* :class:`KMeansDetector` — distance to nearest cluster centroid.
* :class:`EnsembleDetector` — weighted average of normalised scores.

All detectors share a small protocol::

    fit(X)
    score(X)   -> float array, higher = more anomalous, scale unspecified

Higher scores are always more anomalous. The ensemble normalises each
component to ``[0, 1]`` via rank-based scaling before averaging.
"""

# ruff: noqa: N803  # X is the standard sklearn name for a feature matrix.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch import nn

logger = logging.getLogger(__name__)


class Detector(Protocol):
    """Minimal anomaly-detector interface."""

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit the detector on the training matrix."""
        ...

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-row anomaly scores (higher = more anomalous)."""
        ...


def _rank_normalise(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map raw scores to ``[0, 1]`` by rank order.

    Args:
        scores: 1-D float array.

    Returns:
        Array of the same length where the smallest score maps to 0.0 and the
        largest to 1.0. Ties get the average rank.
    """
    n = len(scores)
    if n == 0:
        return scores.astype(float)
    order = scores.argsort()
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)
    if n > 1:
        ranks /= n - 1
    return ranks


class IsolationForestDetector:
    """scikit-learn IsolationForest wrapper.

    Attributes:
        model: The underlying ``IsolationForest`` estimator.
        scaler: Standardiser applied before fitting.
        random_state: Seed for reproducibility.
    """

    def __init__(self, *, random_state: int = 42, n_estimators: int = 200) -> None:
        """Initialise the detector.

        Args:
            random_state: Seed propagated to ``IsolationForest``.
            n_estimators: Number of trees in the forest.
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            random_state=random_state,
            contamination="auto",
            n_jobs=1,
        )

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit the scaler and forest on ``X``."""
        scaled = self.scaler.fit_transform(X)
        self.model.fit(scaled)

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-row anomaly scores; higher = more anomalous."""
        scaled = self.scaler.transform(X)
        # ``score_samples`` returns the opposite of the anomaly score; negate.
        return np.asarray(-self.model.score_samples(scaled), dtype=np.float64)


class _AutoencoderModule(nn.Module):
    """3-layer MLP autoencoder."""

    def __init__(self, in_dim: int, hidden_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.decoder(self.encoder(x))
        return out


class AutoencoderDetector:
    """PyTorch autoencoder; reconstruction error is the anomaly score.

    Attributes:
        scaler: Standardiser applied before training.
        model: The trained autoencoder module (``None`` until ``fit``).
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        hidden_dim: Hidden-layer width.
        bottleneck_dim: Bottleneck width.
        seed: Seed used for torch + numpy initialisation.
    """

    def __init__(
        self,
        *,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 1e-3,
        hidden_dim: int = 32,
        bottleneck_dim: int = 8,
        seed: int = 42,
    ) -> None:
        """Initialise hyperparameters; the network is built in ``fit``."""
        self.scaler = StandardScaler()
        self.model: _AutoencoderModule | None = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.seed = seed

    def fit(self, X: NDArray[np.float64]) -> None:
        """Train the autoencoder on ``X`` with MSE reconstruction loss."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        scaled = self.scaler.fit_transform(X).astype(np.float32)
        in_dim = scaled.shape[1]
        model = _AutoencoderModule(in_dim, self.hidden_dim, self.bottleneck_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        tensor = torch.from_numpy(scaled)
        dataset = torch.utils.data.TensorDataset(tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )

        model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                output = model(batch)
                loss = loss_fn(output, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * batch.shape[0]
            epoch_loss /= max(1, len(dataset))
            logger.debug("autoencoder epoch", extra={"epoch": epoch, "loss": epoch_loss})

        model.eval()
        self.model = model

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return per-row reconstruction error."""
        if self.model is None:
            raise RuntimeError("AutoencoderDetector.score called before fit")
        scaled = self.scaler.transform(X).astype(np.float32)
        tensor = torch.from_numpy(scaled)
        with torch.no_grad():
            recon = self.model(tensor)
        err = (recon - tensor).pow(2).mean(dim=1).cpu().numpy()
        return np.asarray(err, dtype=np.float64)


class KMeansDetector:
    """KMeans clustering; distance to the nearest centroid is the score.

    Attributes:
        n_clusters: Number of clusters.
        random_state: Seed propagated to ``KMeans``.
        scaler: Standardiser applied before clustering.
        model: The fitted ``KMeans`` instance (``None`` until ``fit``).
    """

    def __init__(self, *, n_clusters: int = 8, random_state: int = 42) -> None:
        """Initialise the clusterer.

        Args:
            n_clusters: Number of clusters.
            random_state: Seed propagated to ``KMeans``.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | None = None

    def fit(self, X: NDArray[np.float64]) -> None:
        """Cluster the standardised feature matrix."""
        scaled = self.scaler.fit_transform(X)
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(scaled)

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return distance to nearest centroid for each row."""
        if self.model is None:
            raise RuntimeError("KMeansDetector.score called before fit")
        scaled = self.scaler.transform(X)
        # transform() returns distance to each centroid.
        distances = self.model.transform(scaled)
        return np.asarray(distances.min(axis=1), dtype=np.float64)

    def predict_clusters(self, X: NDArray[np.float64]) -> NDArray[np.intp]:
        """Return the assigned cluster id per row.

        Args:
            X: Feature matrix.

        Returns:
            Integer cluster ids, shape ``(n,)``.
        """
        if self.model is None:
            raise RuntimeError("KMeansDetector.predict_clusters called before fit")
        scaled = self.scaler.transform(X)
        return np.asarray(self.model.predict(scaled), dtype=np.intp)


@dataclass
class EnsembleDetector:
    """Weighted ensemble of detectors with rank-normalised scores.

    Attributes:
        detectors: Mapping of detector name to instance.
        weights: Mapping of detector name to weight (need not sum to 1.0).
    """

    detectors: dict[str, Detector] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)

    def fit(self, X: NDArray[np.float64]) -> None:
        """Fit every component detector on ``X``."""
        for name, det in self.detectors.items():
            logger.info("fitting detector", extra={"detector": name})
            det.fit(X)

    def score(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return weighted-average normalised ensemble score per row."""
        if not self.detectors:
            return np.zeros(len(X), dtype=float)
        total_weight = sum(self.weights.get(name, 1.0) for name in self.detectors)
        if total_weight <= 0:
            raise ValueError("ensemble weights sum to zero")
        accum = np.zeros(len(X), dtype=float)
        for name, det in self.detectors.items():
            normalised = _rank_normalise(det.score(X))
            w = self.weights.get(name, 1.0)
            accum += (w / total_weight) * normalised
        return accum

    def component_scores(self, X: NDArray[np.float64]) -> dict[str, NDArray[np.float64]]:
        """Return raw per-detector scores keyed by detector name."""
        return {name: det.score(X) for name, det in self.detectors.items()}


def combine_normalised(
    component_scores: dict[str, NDArray[np.float64]],
    weights: dict[str, float],
) -> NDArray[np.float64]:
    """Combine pre-computed component scores into an ensemble score.

    Each component is rank-normalised to ``[0, 1]`` before averaging by
    ``weights``. Useful for weight calibration where the detectors have
    already been scored on a validation split.

    Args:
        component_scores: Mapping of detector name to raw score array.
        weights: Mapping of detector name to non-negative weight.

    Returns:
        Combined score array of the same length as the input components.

    Raises:
        ValueError: If ``weights`` sum to zero across the provided components.
    """
    if not component_scores:
        first = next(iter(component_scores.values()), None)
        return np.zeros(0 if first is None else len(first), dtype=float)
    total_weight = sum(weights.get(name, 0.0) for name in component_scores)
    if total_weight <= 0:
        raise ValueError("ensemble weights sum to zero")
    length = len(next(iter(component_scores.values())))
    accum = np.zeros(length, dtype=float)
    for name, scores in component_scores.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        accum += (w / total_weight) * _rank_normalise(scores)
    return accum


def calibrate_weights(
    component_scores: dict[str, NDArray[np.float64]],
    labels: NDArray[np.bool_],
    *,
    k: int,
    grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    objective: str = "pr_auc",
) -> tuple[dict[str, float], float]:
    """Pick ensemble weights maximising the chosen objective on validation data.

    Performs an exhaustive grid search over ``grid`` values for each detector
    weight. Skips the all-zero combination.

    Args:
        component_scores: Validation-split component scores.
        labels: Ground-truth boolean labels parallel to the scores.
        k: Cutoff used when ``objective='precision_at_k'``.
        grid: Weight values to try for each detector.
        objective: ``'pr_auc'`` (default) or ``'precision_at_k'``. PR-AUC is
            preferred because it rewards good ranking across all thresholds,
            not just at one cutoff.

    Returns:
        ``(weights, best_score)`` — the selected weight mapping and the
        validation score it achieved on ``objective``.
    """
    from itertools import product

    from src.anomaly.eval import precision_at_k

    names = list(component_scores.keys())
    best_score = -1.0
    best_weights: dict[str, float] = dict.fromkeys(names, 1.0)
    labels_int = labels.astype(int)
    for combo in product(grid, repeat=len(names)):
        if sum(combo) == 0:
            continue
        weights = dict(zip(names, combo, strict=True))
        scores = combine_normalised(component_scores, weights)
        if objective == "precision_at_k":
            value = precision_at_k(scores, labels_int, k=k)
        elif objective == "pr_auc":
            from sklearn.metrics import average_precision_score

            value = float(average_precision_score(labels_int, scores))
        else:
            raise ValueError(f"unknown objective: {objective}")
        if value > best_score:
            best_score = value
            best_weights = weights
    return best_weights, best_score
