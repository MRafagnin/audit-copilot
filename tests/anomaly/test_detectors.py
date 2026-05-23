"""Tests for ``src.anomaly.detectors``."""

# ruff: noqa: N806  # X is the standard sklearn name for a feature matrix.

from __future__ import annotations

import numpy as np
import pytest

from src.anomaly.detectors import (
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestDetector,
    KMeansDetector,
    _rank_normalise,
    calibrate_weights,
    combine_normalised,
)


@pytest.fixture()
def synthetic_matrix() -> tuple[np.ndarray, int]:
    """Return a feature matrix where the last row is a clear outlier."""
    rng = np.random.default_rng(0)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    outlier = np.array([[20.0, 20.0, 20.0, 20.0, 20.0]])
    X = np.vstack([inliers, outlier])
    return X, X.shape[0] - 1


def test_rank_normalise_basic() -> None:
    out = _rank_normalise(np.array([10.0, 30.0, 20.0]))
    assert out.min() == 0.0
    assert out.max() == 1.0
    assert out.argmax() == 1


def test_isolation_forest_ranks_outlier_highest(
    synthetic_matrix: tuple[np.ndarray, int],
) -> None:
    X, outlier_idx = synthetic_matrix
    det = IsolationForestDetector(random_state=42, n_estimators=100)
    det.fit(X)
    scores = det.score(X)
    assert int(np.argmax(scores)) == outlier_idx


def test_kmeans_ranks_outlier_highly(
    synthetic_matrix: tuple[np.ndarray, int],
) -> None:
    X, outlier_idx = synthetic_matrix
    # n_clusters=1 collapses to a single centroid (mean) so the outlier is
    # unambiguously the farthest row.
    det = KMeansDetector(n_clusters=1, random_state=42)
    det.fit(X)
    scores = det.score(X)
    assert int(np.argmax(scores)) == outlier_idx
    clusters = det.predict_clusters(X)
    assert clusters.shape == (X.shape[0],)


def test_kmeans_requires_fit_before_score() -> None:
    det = KMeansDetector()
    with pytest.raises(RuntimeError):
        det.score(np.zeros((1, 3)))


def test_autoencoder_ranks_outlier_highly(
    synthetic_matrix: tuple[np.ndarray, int],
) -> None:
    X, outlier_idx = synthetic_matrix
    det = AutoencoderDetector(seed=42, epochs=5, batch_size=64, hidden_dim=8, bottleneck_dim=4)
    det.fit(X)
    scores = det.score(X)
    ranked = np.argsort(-scores)
    # Autoencoder is stochastic; require outlier in top 3
    assert outlier_idx in ranked[:3]


def test_autoencoder_requires_fit_before_score() -> None:
    det = AutoencoderDetector()
    with pytest.raises(RuntimeError):
        det.score(np.zeros((1, 3)))


def test_ensemble_combines_detectors(
    synthetic_matrix: tuple[np.ndarray, int],
) -> None:
    X, outlier_idx = synthetic_matrix
    ensemble = EnsembleDetector(
        detectors={
            "iforest": IsolationForestDetector(random_state=42, n_estimators=50),
            "ae": AutoencoderDetector(
                seed=42, epochs=5, batch_size=64, hidden_dim=8, bottleneck_dim=4
            ),
        },
        weights={"iforest": 1.0, "ae": 1.0},
    )
    ensemble.fit(X)
    scores = ensemble.score(X)
    assert scores.shape == (X.shape[0],)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
    assert int(np.argmax(scores)) == outlier_idx
    components = ensemble.component_scores(X)
    assert set(components.keys()) == {"iforest", "ae"}


def test_ensemble_empty_returns_zeros() -> None:
    ensemble = EnsembleDetector()
    out = ensemble.score(np.zeros((4, 3)))
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_ensemble_zero_weights_raise() -> None:
    ensemble = EnsembleDetector(
        detectors={"iforest": IsolationForestDetector(random_state=0)},
        weights={"iforest": 0.0},
    )
    ensemble.fit(np.random.default_rng(0).normal(size=(20, 3)))
    with pytest.raises(ValueError):
        ensemble.score(np.zeros((2, 3)))


def test_combine_normalised_zero_weight_skipped() -> None:
    comps = {
        "a": np.array([0.1, 0.2, 0.9]),
        "b": np.array([0.9, 0.2, 0.1]),
    }
    out = combine_normalised(comps, {"a": 1.0, "b": 0.0})
    # Only 'a' contributes; rank-normalised: 0.1->0, 0.2->0.5, 0.9->1.0
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_combine_normalised_zero_weights_raise() -> None:
    comps = {"a": np.array([0.1, 0.2])}
    with pytest.raises(ValueError):
        combine_normalised(comps, {"a": 0.0})


def test_calibrate_weights_prefers_signal_detector() -> None:
    # 'good' ranks the two positives at the top; 'noisy' ranks them at the
    # bottom. A grid search should down-weight 'noisy'.
    comps = {
        "good": np.array([0.1, 0.2, 0.3, 0.9, 0.95]),
        "noisy": np.array([0.95, 0.9, 0.3, 0.2, 0.1]),
    }
    labels = np.array([False, False, False, True, True])
    weights, score = calibrate_weights(comps, labels, k=2)
    assert score == 1.0
    assert weights["good"] > weights["noisy"]


def test_calibrate_weights_precision_at_k_objective() -> None:
    comps = {
        "good": np.array([0.1, 0.2, 0.3, 0.9, 0.95]),
        "noisy": np.array([0.95, 0.9, 0.3, 0.2, 0.1]),
    }
    labels = np.array([False, False, False, True, True])
    weights, p = calibrate_weights(
        comps, labels, k=2, objective="precision_at_k"
    )
    assert p == 1.0
    assert weights["good"] > weights["noisy"]


def test_calibrate_weights_unknown_objective() -> None:
    comps = {"a": np.array([0.1, 0.9])}
    labels = np.array([False, True])
    with pytest.raises(ValueError):
        calibrate_weights(comps, labels, k=1, objective="nonsense")
