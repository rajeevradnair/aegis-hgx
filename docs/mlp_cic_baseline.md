# MLP CIC Baseline

## Purpose

This document describes the first neural-network baseline for AEGIS-HGX using the cleaned CICIDS2017 tabular feature dataset.

The goal is to add a neural tabular baseline after the logistic regression and XGBoost baselines. This gives AEGIS-HGX three comparable supervised baselines before moving into graph construction, GNNs, temporal modeling, and anomaly-specific architectures.

This work belongs to the training and evaluation phase. It does not change the inference API.

## Input Data

The model trains on:

```text
data/processed/cicids2017/cic_tabular_features.csv
```

This file is produced by:

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

The training script validates that the dataset exists, contains the target column, and includes both benign and attack classes.

## Model

The model is a scikit-learn MLP classifier wrapped in a pipeline:

```text
StandardScaler
→ MLPClassifier
```

Implementation:

```text
src/aegis_hgx/models/baselines/train_mlp_cic.py
```

Configuration:

```text
configs/mlp_cic.yaml
```

Output artifacts:

```text
reports/mlp_cic_metrics.json
artifacts/models/mlp_cic_baseline.joblib
artifacts/lineage/mlp_cic_manifest.json
```

## Why an MLP?

An MLP is a feed-forward neural network that learns nonlinear combinations of input features.

For CIC flow data, this allows the model to combine features such as flow duration, bytes per second, packet counts, protocol, port, and flag behavior into learned intermediate representations.

The MLP baseline is useful because it answers a specific research question:

```text
Does a simple neural network improve over linear and tree-based tabular baselines?
```

It may not outperform XGBoost on tabular data. That is acceptable. The purpose is disciplined comparison, not forcing the neural model to win.

## Why Scaling Is Required

MLPs are sensitive to feature scale because they are optimized with gradient-based learning.

If one feature has a range from 0 to 1 and another has a range from 0 to 1,000,000, the large-scale feature can dominate optimization.

For this reason, the MLP pipeline uses:

```text
StandardScaler
```

This standardizes features before training the neural network.

## Training Flow

The training script performs the following steps:

```text
load config
load CIC tabular feature dataset
validate target column
validate two target classes
split features and target
create stratified train/test split
build StandardScaler + MLPClassifier pipeline
train model
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

PR-AUC is especially important because cyber attack detection is usually class-imbalanced. Accuracy alone can be misleading.

## MLOps Evidence

The MLP trainer logs evidence to MLflow, including:

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

GitHub Actions validates the MLP CIC baseline using the committed CIC-style fixture dataset.

The CI flow includes:

```text
prepare CIC fixture data
ingest CIC fixture data
build CIC tabular features
train CIC logistic model
train CIC XGBoost model
train CIC MLP model
run tests
```

This proves the MLP training path works without requiring CI to download the full CICIDS2017 dataset.

## Important Limitations

This baseline is not a production-ready cyber detector.

Current limitations:

```text
uses a sampled CIC-derived feature file
uses random stratified split
does not yet include training curves
does not yet include calibration analysis
does not yet include multi-seed stability
does not yet include leakage audit
does not yet include temporal validation
does not yet model graph structure
```

High metrics should be treated cautiously. Neural networks can overfit, especially on small or artifact-heavy datasets.

## How to Run

Build CIC tabular features:

```bash
python pipelines/build_tabular_features.py
```

Train MLP baseline:

```bash
python -m aegis_hgx.models.baselines.train_mlp_cic --config configs/mlp_cic.yaml
```

Run the smoke test:

```bash
python -m pytest tests/test_train_mlp_cic.py -ra
```

Run all tests:

```bash
python -m pytest -ra
```
