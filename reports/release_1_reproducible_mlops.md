# Release 1: Reproducible MLOps Foundation

## Summary

This release establishes the first reproducible MLOps foundation for AEGIS-HGX.

The current system trains and serves a simple logistic-regression baseline on synthetic cyber event data. The goal of this release is not model sophistication. The goal is to make the baseline repeatable, testable, tracked, documented, and inspectable before moving to public IDS data and stronger models.

## Scope

Included in this release:

* config-driven baseline training
* synthetic data generation
* MLflow experiment tracking
* FastAPI inference endpoint
* unit tests for data generation and model input shape
* API contract tests
* GitHub Actions CI
* model card and data card templates
* DVC tracking for the synthetic dataset
* model/data lineage manifest
* basic Evidently drift report
* threshold tuning report

Not included:

* real IDS dataset training
* graph models
* temporal modeling
* production SOC deployment
* autonomous blocking
* adversarial robustness

## Current Architecture

```text
synthetic events
  -> config-driven training
  -> logistic baseline model
  -> MLflow run + metrics
  -> model/data lineage manifest
  -> drift monitoring
  -> threshold tuning
  -> tested inference endpoint
```

## Reproducibility Controls

The baseline is controlled through `configs/baseline_logistic.yaml`.

The synthetic dataset is DVC-tracked, so the data artifact is tied to a versioned metadata file instead of relying only on a mutable CSV path.

The lineage manifest records the model artifact, dataset hash, config hashes, Git metadata, MLflow run metadata, metrics, and training command.

## Testing

Current test coverage includes:

* data generation checks
* model input shape checks
* inference API contract checks
* lineage manifest smoke test
* drift monitoring smoke test
* threshold tuning smoke test

CI runs the project tests through GitHub Actions.

## Experiment Tracking

MLflow is used to record baseline parameters, metrics, config artifacts, metrics artifacts, model artifacts, and lineage artifacts.

The main baseline metrics are written to:

```text
reports/baseline_logistic_metrics.json
```

## Monitoring

A basic Evidently drift report is generated at:

```text
reports/monitoring/basic_drift_report.html
```

The current monitoring flow uses the lineage-backed synthetic dataset as the reference source and compares a reference/current split. This is a local monitoring loop, not production monitoring.

## Threshold Tuning

Threshold tuning writes an operating-point report to:

```text
reports/evaluation/threshold_tuning_report.json
```

The report evaluates multiple thresholds and records precision, recall, F1, false positives, false negatives, and alert count. The initial recommended threshold is selected using max F1, then recall, then precision as tie-breakers.

## Release Decision

Accepted as a reproducible synthetic-data MLOps foundation.

Not accepted as a validated cyber anomaly detection system.

## Known Limitations

The current model uses synthetic data only. The baseline is intentionally simple. Drift monitoring is local. Threshold tuning is based on the synthetic test split. The inference API does not yet load threshold policy metadata. The system has not yet been evaluated on CICIDS, CSE-CIC-IDS, graph data, temporal data, or adversarial examples.

## Next Step

The next release should move from synthetic data to a public IDS dataset and build a baseline comparison laboratory. That work should include stronger tabular baselines, calibration analysis, threshold policy refinement, and clearer comparison metrics before graph modeling begins.
