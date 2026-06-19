# Logistic Regression Baseline

## Purpose

This model establishes the simplest supervised baseline for classifying synthetic cyber events as normal or suspicious.

Future models must demonstrate measurable improvement over this baseline.

## Pipeline

1. Load the configured dataset.
2. Separate features and labels.
3. Create a stratified train/test split.
4. standardize numeric features.
5. One-hot encode categorical features.
6. Train logistic regression.
7. Evaluate unseen test data.
8. Save metrics and the trained pipeline.

## Metrics

The evaluation includes:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- PR-AUC
- confusion matrix

PR-AUC and recall are particularly important because suspicious events are the minority class.

## Limitations

High performance is expected because the synthetic dataset contains deliberately clear suspicious patterns.

The results validate pipeline correctness, not real-world detection capability.