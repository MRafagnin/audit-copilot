"""Train the anomaly-detection ensemble and persist metrics + flagged rows.

Pipeline:

1. Load the synthetic GL CSV.
2. Build the engineered feature matrix.
3. Train IsolationForest, Autoencoder and KMeans on an 80% split.
4. Score the 20% holdout; evaluate per-detector and the ensemble.
5. Persist metrics JSON and the top-k flagged rows CSV.

Run::

    uv run python scripts/train_anomaly.py
"""

# ruff: noqa: N806  # X_train/X_test follow the sklearn convention.

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.anomaly.detectors import (
    AutoencoderDetector,
    EnsembleDetector,
    IsolationForestDetector,
    KMeansDetector,
    calibrate_weights,
    combine_normalised,
)
from src.anomaly.eval import EvalResult, evaluate, persist
from src.anomaly.features import FEATURE_COLUMNS, build_features
from src.core.config import settings
from src.core.logging_config import configure_logging
from src.fusion.explain import derive_feature_flags

logger = logging.getLogger(__name__)

_UNUSUAL_PAIR_QUANTILE: float = 0.05
_ZSCORE_OUTLIER: float = 3.0
_NEAR_DUP_WINDOW_MIN: int = 15

# Severity-ordered list of flag names; the order is preserved when joining
# into the persisted ``feature_flags`` column so the UI badges render with
# the most damning signal leftmost.
_FLAG_ORDER: tuple[str, ...] = (
    "is_near_duplicate",
    "is_amount_outlier_for_account",
    "is_unusual_user_account",
    "is_round_credit_to_revenue",
    "is_benford_first_digit_9",
    "is_large_amount",
    "is_sensitive_account",
    "is_weekend",
    "is_after_hours",
    "is_round_amount",
)


def enrich_feature_flags(
    df: pd.DataFrame,
    X: np.ndarray,  # noqa: N803
    top_idx: np.ndarray,
) -> list[str]:
    """Compute single-row + cross-row feature flags for the top-k rows.

    Args:
        df: Full GL DataFrame (used for groupby-based cross-row signals).
        X: Feature matrix aligned with ``df`` row order; columns follow
            :data:`FEATURE_COLUMNS`.
        top_idx: Integer positions (relative to ``df``) of the rows to enrich.

    Returns:
        List of semicolon-joined flag strings, one entry per ``top_idx`` row,
        in severity-first order.
    """
    pair_col = FEATURE_COLUMNS.index("user_account_freq")
    z_col = FEATURE_COLUMNS.index("amount_zscore_per_account")

    pair_threshold = float(np.quantile(X[:, pair_col], _UNUSUAL_PAIR_QUANTILE))

    # Near-duplicate detection over the full frame: same (account, rounded
    # amount) with another row within ``_NEAR_DUP_WINDOW_MIN`` minutes.
    amount_abs = np.abs(df["debit"].to_numpy() - df["credit"].to_numpy())
    work = pd.DataFrame(
        {
            "account": df["account"].to_numpy(),
            "amount_key": np.round(amount_abs, 2),
            "ts": pd.to_datetime(df["posting_ts"]),
        }
    )
    near_dup = np.zeros(len(df), dtype=bool)
    window = pd.Timedelta(minutes=_NEAR_DUP_WINDOW_MIN)
    for _, group in work.groupby(["account", "amount_key"], sort=False):
        if len(group) < 2:
            continue
        ts_sorted = group.sort_values("ts")
        ts_vals = ts_sorted["ts"].to_numpy()
        diffs_next = np.diff(ts_vals)
        is_dup = np.zeros(len(ts_vals), dtype=bool)
        for i in range(len(ts_vals)):
            if i < len(diffs_next) and diffs_next[i] <= window:
                is_dup[i] = True
                is_dup[i + 1] = True
        near_dup[ts_sorted.index.to_numpy()] = is_dup

    out: list[str] = []
    for pos in top_idx:
        row = df.iloc[pos]
        flags = set(
            derive_feature_flags(
                posting_ts=str(row["posting_ts"]),
                debit=float(row["debit"]),
                credit=float(row["credit"]),
                account=str(row["account"]),
            )
        )
        if X[pos, pair_col] <= pair_threshold:
            flags.add("is_unusual_user_account")
        if abs(float(X[pos, z_col])) > _ZSCORE_OUTLIER:
            flags.add("is_amount_outlier_for_account")
        if near_dup[pos]:
            flags.add("is_near_duplicate")
        ordered = [f for f in _FLAG_ORDER if f in flags]
        out.append(";".join(ordered))
    return out


def _three_way_split(
    n: int, *, train_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shuffled train/val/test integer index arrays."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    train_cut = int(train_frac * n)
    val_cut = train_cut + int(val_frac * n)
    return perm[:train_cut], perm[train_cut:val_cut], perm[val_cut:]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Train the anomaly ensemble.")
    parser.add_argument("--gl", default="data/gl/journal_entries.csv")
    parser.add_argument("--metrics", default="data/metrics/anomaly_eval.json")
    parser.add_argument("--flagged", default="data/flagged/top_k.csv")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=settings.anomaly_seed)
    parser.add_argument("--epochs", type=int, default=12)
    args = parser.parse_args(argv)

    configure_logging()
    gl_path = Path(args.gl)
    if not gl_path.exists():
        logger.error("gl csv missing", extra={"path": str(gl_path)})
        return 1

    df = pd.read_csv(gl_path)
    logger.info("gl loaded", extra={"rows": len(df), "path": str(gl_path)})

    fm = build_features(df)
    if fm.labels is None:
        logger.error("labels missing in gl; expected is_anomaly column")
        return 1

    train_idx, val_idx, test_idx = _three_way_split(
        len(df),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    X_train = fm.X[train_idx]
    X_val = fm.X[val_idx]
    X_test = fm.X[test_idx]
    y_val = fm.labels[val_idx]
    y_test = fm.labels[test_idx]

    ensemble = EnsembleDetector(
        detectors={
            "isolation_forest": IsolationForestDetector(random_state=args.seed),
            "autoencoder": AutoencoderDetector(seed=args.seed, epochs=args.epochs),
            "kmeans": KMeansDetector(random_state=args.seed),
        },
        weights={"isolation_forest": 1.0, "autoencoder": 1.0, "kmeans": 1.0},
    )
    ensemble.fit(X_train)

    val_components = ensemble.component_scores(X_val)
    best_weights, best_val_score = calibrate_weights(val_components, y_val, k=args.top_k)
    ensemble.weights = best_weights
    logger.info(
        "ensemble weights calibrated",
        extra={
            "weights": {n: round(w, 3) for n, w in best_weights.items()},
            "val_pr_auc": round(best_val_score, 3),
        },
    )

    components = ensemble.component_scores(X_test)
    ensemble_scores = combine_normalised(components, best_weights)

    results: dict[str, EvalResult] = {}
    for name, scores in components.items():
        results[name] = evaluate(scores, y_test, k=args.top_k)
    results["ensemble"] = evaluate(ensemble_scores, y_test, k=args.top_k)

    for name, metrics in results.items():
        logger.info(
            "detector metrics",
            extra={
                "detector": name,
                "precision_at_k": round(metrics.precision_at_k, 3),
                "roc_auc": round(metrics.roc_auc, 3),
                "pr_auc": round(metrics.pr_auc, 3),
            },
        )

    persist(results, Path(args.metrics))

    # Top-k flagged rows from ensemble scoring on the holdout split
    k = min(args.top_k, len(ensemble_scores))
    top_idx_local = np.argpartition(ensemble_scores, -k)[-k:]
    top_idx_local = top_idx_local[np.argsort(-ensemble_scores[top_idx_local])]
    original_idx = test_idx[top_idx_local]
    flagged = df.iloc[original_idx].copy()
    flagged["ensemble_score"] = ensemble_scores[top_idx_local]
    for name, scores in components.items():
        flagged[f"{name}_score"] = scores[top_idx_local]
    flagged["feature_flags"] = enrich_feature_flags(df, fm.X, original_idx)

    flagged_path = Path(args.flagged)
    flagged_path.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_csv(flagged_path, index=False)
    logger.info(
        "flagged rows written",
        extra={"path": str(flagged_path), "k": k},
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
