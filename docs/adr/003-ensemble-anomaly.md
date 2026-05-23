# ADR 003 — Ensemble anomaly detection (IsolationForest + Autoencoder + KMeans)

* **Status**: Accepted
* **Date**: 2026-05-22
* **Owner**: Matheus Rafagnin

## Context

The journal-entry anomaly detector must surface the kinds of patterns ASA 240
flags as fraud-risk indicators: round amounts, weekend / after-hours
postings, unusual user-account pairs, and Benford deviations. The synthetic
GL is 50 000 rows with ~1 % seeded anomalies of varying type.

No single unsupervised model catches all of those:

* `IsolationForest` is strong on tabular outliers but weak when fraud
  patterns are dense (clustered).
* An autoencoder picks up subtle reconstruction errors in the encoded
  feature space but is sensitive to feature scaling.
* `KMeans` doesn't score anomalies directly, but cluster membership is a
  useful **explanatory** feature for the fusion narrative.

## Decision

Run three detectors and **score-fuse** the two that produce per-row anomaly
scores; use the third for explanation only:

* `IsolationForest` (sklearn, `n_estimators=200`).
* `Autoencoder` — 3-layer MLP in PyTorch on standardised features;
  reconstruction error is the score.
* `KMeans` — `n_clusters=8`; produces a `cluster_id` feature carried into
  the explainer's prompt context (not part of the anomaly score).

Each per-row score is min-max normalised to `[0, 1]`. Final score is a
weighted average. Weights are **calibrated on a held-out split against
PR-AUC** (not ROC-AUC) because the class is imbalanced. Calibrated weights:

```
iso     = 0.25
ae      = 1.00
kmeans  = 0.00   # explanation only
```

Evaluation metrics persisted to `data/metrics/anomaly_eval.json`:

* Precision@100 = **0.98**
* ROC-AUC = **0.884**
* PR-AUC = **0.646**

## Alternatives considered

| Option | Why not |
|---|---|
| Supervised classifier (e.g. XGBoost) | Requires labelled fraud — not realistic for an audit demo. The seeded labels exist only for evaluation, not training. |
| Single autoencoder | Misses outliers that the AE under-reconstructs only marginally; IsolationForest's tree structure adds diversity. |
| LOF / OneClassSVM | LOF is `O(n²)`; OneClassSVM requires careful kernel tuning. Marginal gain at this scale. |
| Deep tabular model (TabNet, etc.) | Over-engineered for a 50k-row demo and not interpretable. |

## Consequences

**Positive**

* Two-model fusion catches both broad outliers and reconstruction-error
  outliers without needing labels.
* PR-AUC calibration matches the real ranking objective (top-`k` review
  queue, not threshold accuracy).
* KMeans cluster id is human-interpretable and feeds the narrative
  explainer directly.

**Negative / accepted trade-offs**

* Three models to train. All three fit in <60 s on CPU, so the cost is paid
  once during `make bootstrap`.
* The autoencoder dominates the score (weight 1.0). This is intentional
  from PR-AUC calibration but means the ensemble is essentially "AE plus
  an iso safety net".

## Azure mapping

The detectors fit comfortably into **Azure ML** as a training pipeline
producing a registered model. Inference becomes a managed endpoint;
training data lands in **Azure Data Lake Storage Gen2**. The fusion
explainer reads the model output and routes through the same `LLMClient`
abstraction as the RAG pipeline.
