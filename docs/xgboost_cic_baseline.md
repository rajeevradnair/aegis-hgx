# XGBoost CIC Baseline

## Purpose

This document describes the first XGBoost baseline for AEGIS-HGX using the cleaned CICIDS2017 tabular feature dataset.

The goal is to create a stronger tabular baseline after the logistic regression CIC baseline. Logistic regression gives a simple linear benchmark. XGBoost gives a more powerful nonlinear benchmark that can learn interactions between network-flow features.

This work belongs to the training and evaluation phase. It does not change the inference API.

## Input Data

The model trains on:

```text
data/processed/cicids2017/cic_tabular_features.csv
```

This file is produced by the CIC feature-building pipeline:

```text
pipelines/build_tabular_features.py
```

Expected structure:

```text
feature_1, feature_2, feature_3, ..., target
```

The `target` column is binary:

```text
0 = benign
1 = attack
```

The dataset must contain both classes. Training should fail if only one class is present.

## Model

The model is an XGBoost binary classifier.

Implementation:

```text
src/aegis_hgx/models/baselines/train_xgboost_cic.py
```

Configuration:

```text
configs/xgboost_cic.yaml
```

Output artifacts:

```text
reports/xgboost_cic_metrics.json
artifacts/models/xgboost_cic_baseline.joblib
artifacts/lineage/xgboost_cic_manifest.json
```

## Why XGBoost?

XGBoost is a strong baseline for tabular datasets because it can learn nonlinear feature interactions.

For example, a single feature such as destination port may not be suspicious by itself. But a combination of unusual port, high packet count, short flow duration, and abnormal flag behavior may indicate attack traffic.

## Training Flow

The training script performs the following steps:

```text
load config
load CIC tabular feature dataset
validate target column
validate two target classes
split features and target
create stratified train/test split
train XGBoost classifier
compute metrics
save metrics JSON
save model artifact
log MLflow evidence
write lineage manifest
```

## Metrics

The baseline reports:

```text
accuracy
precision
recall
F1
ROC-AUC
PR-AUC
confusion matrix
test row count
positive row count
```

## MLOps Evidence

The XGBoost trainer logs evidence to MLflow, including:

```text
run parameters
scalar metrics
training config
metrics report
model artifact
MLflow-formatted model
confusion matrix figure
lineage manifest
```

The lineage manifest connects the model artifact to:

```text
training data path
training config
metrics file
MLflow run ID
model artifact path
training command
feature snapshot identifier
```

## CI Coverage

https://github.com/rajeevradnair/aegis-hgx.git

GitHub Actions now validates the CIC baseline path using a small committed CIC-style fixture dataset.

The CI flow includes:

```text
prepare CIC fixture data
ingest CIC fixture data
build CIC tabular features
train CIC logistic model
train CIC XGBoost model
run tests
```

This proves the training path works without requiring CI to download the full CICIDS2017 dataset.

## Important Limitations

This baseline is not a production-ready cyber detector.

Current limitations:

```text
uses a sampled CIC-derived feature file
may still contain dataset artifacts
uses random stratified split
does not yet perform duplicate analysis
does not yet perform leakage audit
does not yet include calibration analysis
does not yet compare against multiple seeds
does not yet include temporal validation
```

High metrics should be treated cautiously. XGBoost is powerful and can exploit leakage or dataset shortcuts if they exist.

## How to Run

Build CIC tabular features:

```bash
python pipelines/build_tabular_features.py
```

Train XGBoost baseline:

```bash
python -m aegis_hgx.models.baselines.train_xgboost_cic --config configs/xgboost_cic.yaml
```

Run the smoke test:

```bash
python -m pytest tests/test_train_xgboost_cic.py -ra
```

Run all tests:

```bash
python -m pytest -ra
```

## Next Step

The next model baseline is the MLP baseline. After logistic regression, XGBoost, and MLP are available, AEGIS-HGX can produce a baseline comparison report.

The comparison should evaluate whether stronger models actually improve useful detection metrics such as PR-AUC, recall, false positives, and false negatives.
