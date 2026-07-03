# AEGIS-HGX PyTorch MLP Training Diagnostics

## Purpose

This document describes the PyTorch MLP diagnostic trainer for AEGIS-HGX using the cleaned CICIDS2017 tabular feature dataset.

The goal is to add training-curve diagnostics after the logistic regression, XGBoost, and scikit-learn MLP baselines. Instead of only plotting a post-hoc training loss curve, this implementation uses a more research-grade design:

```text
train split
validation split
test split
per-epoch validation metrics
epoch x threshold operating surface
final untouched test-set metrics
```

This work belongs to the training and evaluation phase. It does not change the inference API.

## Why This Trainer Exists

The earlier scikit-learn MLP baseline is useful as a quick neural baseline, but it does not expose a fully transparent training loop.

The PyTorch trainer gives direct control over batching, forward pass, loss computation, backpropagation, optimizer steps, validation evaluation, test evaluation, probability thresholds, and training-history artifacts.

This is more interview-defensible because it demonstrates the mechanics of neural-network training instead of relying only on high-level library behavior.

## Implementation Files

Trainer:

```text
src/aegis_hgx/models/baselines/train_torch_mlp_cic.py
```

Config:

```text
configs/torch_mlp_cic.yaml
```

Smoke test:

```text
tests/test_train_torch_mlp_cic.py
```

CI workflow:

```text
.github/workflows/ci.yml
```

## Input Data

The trainer uses the CIC tabular feature file:

```text
data/processed/cicids2017/cic_tabular_features.csv
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

The trainer validates that the dataset exists, is not empty, contains the target column, and includes both benign and attack classes.

## Train / Validation / Test Split

The trainer uses an explicit train/validation/test split.

The test split is held out and evaluated only after training is complete. The validation split is used during training to diagnose generalization behavior. The train split is the only split used for weight updates.

```text
train = model learns from this data
validation = model behavior is monitored during training
test = final unbiased evaluation after training
```

## Leakage Prevention

The `StandardScaler` is fit only on the train split.

Correct behavior:

```text
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_validation)
scaler.transform(X_test)
```

Incorrect behavior:

```text
scaler.fit(full_dataset)
```

Fitting the scaler on the full dataset leaks validation/test distribution information into training.

## Model Architecture

The model is a simple feed-forward PyTorch MLP:

```text
input features
-> Linear
-> ReLU
-> Dropout
-> Linear
-> ReLU
-> Dropout
-> Linear
-> logit
```

The model outputs logits, not probabilities. Probabilities are computed with sigmoid only during evaluation:

```text
probability = sigmoid(logit)
```

## Loss Function

The trainer uses:

```text
BCEWithLogitsLoss
```

This is the correct binary-classification loss for logits because it combines sigmoid and binary cross entropy in a numerically stable way.

The trainer also computes positive-class weight:

```text
pos_weight = negative_count / positive_count
```

This helps account for class imbalance, which is common in cyber attack detection.

## Optimizer

The trainer uses Adam with configurable learning rate and weight decay:

```text
optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

## Main Training Flow

```text
load config
load CIC tabular dataset
validate target
split features and target
create train/validation/test split
fit scaler on train only
transform train/validation/test
create PyTorch tensors
create train DataLoader
build PyTorch MLP
build BCEWithLogitsLoss
build Adam optimizer

for each epoch:
    train on train batches
    evaluate on validation split
    compute validation loss
    compute validation PR-AUC
    compute validation ROC-AUC
    compute threshold-grid metrics
    append epoch history
    append threshold history

after training:
    evaluate once on untouched test split
    compute final test metrics
    save model checkpoint
    save scaler
    save diagnostic histories
    save plots
    save MLflow evidence
    save lineage manifest
```

## Artifact Design

The trainer separates final evidence from training diagnostics.

### Final Metrics

```text
reports/torch_mlp_cic_train_test_validation_metrics.json
```

Purpose:

```text
final train/validation/test summary and untouched test-set evidence
```

This file stores model name, dataset name, split type, train rows, validation rows, test rows, test positive rows, test negative rows, test PR-AUC, test ROC-AUC, test loss, default-threshold test metrics, test confusion matrix, best validation epoch, selected threshold, and artifact paths.

This file does not store every epoch or every threshold row.

### Epoch History

```text
reports/torch_mlp_cic_epoch_history.json
```

Purpose:

```text
compact learning dynamics across epochs
```

Each row contains:

```text
epoch
train_loss
validation_loss
validation_pr_auc
validation_roc_auc
```

This file supports classic training-curve diagnostics.

### Epoch Probability-Threshold History

```text
reports/torch_mlp_cic_epoch_probability_threshold_history.csv
```

Purpose:

```text
epoch x threshold operating surface
```

Each row contains:

```text
epoch
threshold
tp
fp
tn
fn
accuracy
precision
recall
f1
```

This is stronger than a simple training curve because it shows how operational behavior changes across probability thresholds and epochs.

## Why Threshold History Matters

Cyber detection is threshold-sensitive.

A model can improve recall by predicting more alerts, but that may create too many false positives for a SOC team.

The threshold history lets us answer:

```text
At which epoch did PR-AUC peak?
Which threshold gave the best F1?
Did recall improve only by exploding false positives?
Did threshold behavior become unstable over training?
Which threshold range looks operationally usable?
```

This gives a more defensible story than only reporting final accuracy or PR-AUC.

## Correct Curve Naming

The trainer distinguishes AUC trends from actual PR and ROC curves.

### AUC-by-epoch charts

These are scalar training diagnostics:

```text
reports/figures/torch_mlp_pr_auc_by_epoch.png
reports/figures/torch_mlp_roc_auc_by_epoch.png
```

They are not precision-recall or ROC curves.

They plot:

```text
x-axis = epoch
y-axis = validation PR-AUC or validation ROC-AUC
```

### Actual Precision-Recall Curve

```text
reports/figures/torch_mlp_final_test_precision_recall_curve.png
```

This plots:

```text
x-axis = recall
y-axis = precision
```

### Actual ROC Curve

```text
reports/figures/torch_mlp_final_test_roc_curve.png
```

This plots:

```text
x-axis = false positive rate
y-axis = true positive rate
```

### Other Diagnostic Plots

```text
reports/figures/torch_mlp_loss_by_epoch.png
reports/figures/torch_mlp_threshold_f1_by_probability_threshold.png
```

The loss plot helps diagnose underfitting and overfitting.

The threshold-F1 plot shows how validation F1 changes across probability thresholds for the final epoch.

## MLflow Evidence

The trainer logs MLflow evidence including:

```text
config file
final metrics JSON
epoch history JSON
threshold history CSV
model checkpoint
scaler artifact
diagnostic figures
lineage manifest
```

MLflow answers:

```text
What happened in this run?
```

Lineage answers:

```text
How was this artifact produced, and why is it traceable?
```

## Lineage Manifest

The trainer writes:

```text
artifacts/lineage/torch_mlp_cic_manifest.json
```

The lineage manifest links the model artifact to training data path, training config path, metrics path, MLflow experiment ID, MLflow run ID, training command, feature snapshot ID, and offline store path.

This makes the trained model auditable.

## CI Coverage

The CI workflow validates the PyTorch MLP diagnostic path using the committed CIC-style fixture dataset.

The CIC CI flow includes:

```text
prepare CIC fixture data
ingest CIC fixture data
build CIC tabular features
train CIC logistic model
train CIC XGBoost model
train CIC scikit-learn MLP model
train CIC PyTorch MLP diagnostic model
run tests
```

The PyTorch smoke test validates that the trainer writes final metrics JSON, epoch history JSON, threshold history CSV, model checkpoint, scaler artifact, lineage manifest, and diagnostic plots.

## How to Run

Build CIC tabular features:

```bash
python pipelines/build_tabular_features.py
```

Train the PyTorch MLP diagnostic model:

```bash
python -m aegis_hgx.models.baselines.train_torch_mlp_cic --config configs/torch_mlp_cic.yaml
```

Run the focused smoke test:

```bash
python -m pytest tests/test_train_torch_mlp_cic.py -ra
```

Run the full test suite:

```bash
python -m pytest -ra
```

## Expected Outputs

```text
reports/torch_mlp_cic_train_test_validation_metrics.json
reports/torch_mlp_cic_epoch_history.json
reports/torch_mlp_cic_epoch_probability_threshold_history.csv
reports/figures/torch_mlp_loss_by_epoch.png
reports/figures/torch_mlp_pr_auc_by_epoch.png
reports/figures/torch_mlp_roc_auc_by_epoch.png
reports/figures/torch_mlp_threshold_f1_by_probability_threshold.png
reports/figures/torch_mlp_final_test_precision_recall_curve.png
reports/figures/torch_mlp_final_test_roc_curve.png
artifacts/models/torch_mlp_cic_baseline.pt
artifacts/models/torch_mlp_cic_scaler.joblib
artifacts/lineage/torch_mlp_cic_manifest.json
```

## Important Limitations

This trainer is still a baseline diagnostic trainer, not a final production detector.

Current limitations:

```text
uses CIC tabular features, not graph structure
uses random train/validation/test split, not temporal split
does not yet implement early stopping
does not yet tune threshold policy
does not yet run multi-seed evaluation
does not yet compute confidence intervals
does not yet include calibration analysis
```

These are intentionally handled later in the roadmap.

## Interview-Defensible Explanation

```text
I built a PyTorch MLP diagnostic trainer with explicit train, validation, and test splits. The train split is used for weight updates, the validation split is used for per-epoch diagnostics, and the test split is evaluated once at the end. I store three evidence layers: final train/validation/test metrics, compact epoch-level learning history, and a full epoch-by-threshold operating surface. This lets me diagnose not only overfitting but also whether apparent recall gains are creating unacceptable false positives. I also separate AUC-by-epoch trend charts from actual PR and ROC curves, which are generated on the untouched test split.
```

## Cheat Sheet

```text
Train split updates weights.
Validation split monitors learning.
Test split is used once after training.
Scaler must be fit on train only.
PyTorch model outputs logits.
BCEWithLogitsLoss expects logits.
Sigmoid converts logits to probabilities.
PR-AUC and ROC-AUC are threshold-independent ranking metrics.
Precision, recall, F1, and confusion matrix depend on a threshold.
PR-AUC by epoch is not a PR curve.
ROC-AUC by epoch is not a ROC curve.
PR curve plots recall vs precision.
ROC curve plots false positive rate vs true positive rate.
Threshold history shows operational behavior across thresholds.
Cyber detection is threshold-sensitive because false positives create SOC fatigue.
```

## Commit Messages

Recommended implementation commit:

```bash
git add configs/torch_mlp_cic.yaml src/aegis_hgx/models/baselines/train_torch_mlp_cic.py tests/test_train_torch_mlp_cic.py .github/workflows/ci.yml
git commit -m "Add PyTorch MLP training diagnostics"
```

Recommended documentation commit:

```bash
git add docs/torch_mlp_cic_training_diagnostics.md
git commit -m "Document PyTorch MLP training diagnostics"
```
