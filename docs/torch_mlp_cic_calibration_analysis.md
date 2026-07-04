# Calibration Analysis

## Purpose

This document describes the AEGIS-HGX calibration analysis pipeline for the CICIDS2017 PyTorch MLP baseline.

The goal is to determine whether model scores are trustworthy as risk probabilities.

## Phase

This is an offline training/evaluation/test diagnostic. It does not change the inference API.

## Inputs

- `data/processed/cicids2017/cic_tabular_features.csv`
- `artifacts/models/torch_mlp_cic_baseline.pt`
- `artifacts/models/torch_mlp_cic_scaler.joblib`
- `configs/calibration_cic.yaml`

## Outputs

- `reports/calibration/torch_mlp_cic_calibration_metrics.json`
- `reports/calibration/torch_mlp_cic_calibration_bins.csv`
- `reports/figures/torch_mlp_cic_reliability_diagram.png`
- `reports/figures/torch_mlp_cic_probability_histogram.png`

## Metrics

### Brier Score

Brier score measures the squared error between predicted probability and true label.

Lower is better.

### Expected Calibration Error

Expected Calibration Error measures the weighted average gap between predicted probability and observed positive rate across probability bins.

Lower is better.

### Reliability Diagram

The reliability diagram compares mean predicted probability against observed attack rate.

A perfectly calibrated model follows the diagonal line.

### Probability Histogram

The probability histogram shows how confident the model is across the held-out test split.

## Interpretation

Calibration is different from ranking.

ROC-AUC and PR-AUC tell us whether attacks are generally ranked above benign traffic. Calibration tells us whether a score like 0.80 actually behaves like an 80% probability.

## Current Limitation

This report evaluates raw PyTorch MLP probabilities. It does not yet apply Platt scaling, isotonic regression, temperature scaling, or any post-training calibration method.

## Next Step

The next logical step is to compare calibration across logistic regression, XGBoost, scikit-learn MLP, and PyTorch MLP, then decide whether a post-training calibration method is needed.