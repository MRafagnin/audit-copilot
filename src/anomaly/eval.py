"""Evaluation metrics for anomaly detectors.

Computes precision@k, ROC-AUC and PR-AUC against ground-truth labels and
persists the results as JSON for the README/CI.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalResult:
    """Anomaly-detector evaluation metrics.

    Attributes:
        precision_at_k: Fraction of the top-``k`` scored rows that are true
            anomalies.
        roc_auc: Area under the ROC curve.
        pr_auc: Area under the precision-recall curve (average precision).
        k: ``k`` used for precision@k.
        n: Total rows evaluated.
        anomalies: Number of ground-truth anomalies in ``labels``.
    """

    precision_at_k: float
    roc_auc: float
    pr_auc: float
    k: int
    n: int
    anomalies: int


def precision_at_k(scores: NDArray[np.float64], labels: NDArray[np.float64], *, k: int) -> float:
    """Fraction of the top-``k`` scored rows that are labelled anomalous.

    Args:
        scores: Anomaly scores, higher = more anomalous.
        labels: Boolean (or 0/1) labels parallel to ``scores``.
        k: Number of top rows to inspect.

    Returns:
        ``hits / k`` in ``[0, 1]``. Returns 0.0 when ``k <= 0`` or ``len(scores) == 0``.
    """
    if k <= 0 or len(scores) == 0:
        return 0.0
    k = min(k, len(scores))
    top_idx = np.argpartition(scores, -k)[-k:]
    hits = int(labels[top_idx].astype(bool).sum())
    return hits / k


def evaluate(scores: NDArray[np.float64], labels: NDArray[np.float64], *, k: int) -> EvalResult:
    """Compute precision@k, ROC-AUC and PR-AUC.

    Args:
        scores: Anomaly scores.
        labels: Ground-truth boolean labels.
        k: Cutoff for precision@k.

    Returns:
        :class:`EvalResult` with all metrics filled in. ROC-AUC and PR-AUC
        fall back to ``0.0`` when only one class is present in ``labels``.
    """
    labels_int = labels.astype(int)
    n = len(scores)
    anomalies = int(labels_int.sum())
    p_at_k = precision_at_k(scores, labels_int, k=k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        warnings.simplefilter("ignore", UserWarning)
        try:
            roc = float(roc_auc_score(labels_int, scores))
        except ValueError:
            roc = 0.0
        try:
            pr = float(average_precision_score(labels_int, scores))
        except ValueError:
            pr = 0.0
    if not np.isfinite(roc):
        roc = 0.0
    if not np.isfinite(pr):
        pr = 0.0
    return EvalResult(
        precision_at_k=float(p_at_k),
        roc_auc=roc,
        pr_auc=pr,
        k=k,
        n=n,
        anomalies=anomalies,
    )


def persist(results: dict[str, EvalResult], out_path: Path) -> None:
    """Write a ``{detector: metrics}`` mapping to JSON.

    Args:
        results: Mapping of detector name to :class:`EvalResult`.
        out_path: Output file path. Parent directories are created.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(metrics) for name, metrics in results.items()}
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    logger.info(
        "metrics persisted",
        extra={"path": str(out_path), "detectors": list(results.keys())},
    )
