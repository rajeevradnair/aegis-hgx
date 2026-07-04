# Baseline Comparison Report Workflow

## Purpose

This document describes the AEGIS-HGX baseline comparison report workflow.

The goal is to consolidate CIC baseline model evidence and define the performance bar that future graph models must beat.

## Phase

This is an offline training/evaluation/test reporting workflow. It does not modify inference behavior.

## Inputs

- `reports/logistic_cic_metrics.json`
- `reports/xgboost_cic_metrics.json`
- `reports/mlp_cic_metrics.json`
- `reports/torch_mlp_cic_train_test_validation_metrics.json`
- `reports/calibration/torch_mlp_cic_calibration_metrics.json`
- `reports/experiments/torch_mlp_cic_multiseed_summary.json`
- `configs/baseline_comparison_report.yaml`

## Output

- `reports/baseline_comparison_report.md`

## What the Report Contains

The report includes:

- purpose
- phase classification
- artifact inventory
- model comparison table
- initial interpretation
- calibration summary
- seed-stability summary
- graph-model target bar
- limitations
- next steps

## Why This Report Matters

The report turns isolated model artifacts into a coherent baseline laboratory.

It answers:

- Which tabular baselines exist?
- Which metrics are strongest?
- Which model has the best PR-AUC?
- Which model has the best F1?
- Are model scores calibrated?
- Is the PyTorch MLP stable across random seeds?
- What must future graph models beat?

## Key Principle

A future GNN is not better simply because it is more complex.

A future graph model must show measurable improvement over tabular baselines on meaningful metrics such as:

- PR-AUC
- false-positive behavior
- false-negative behavior
- calibration quality
- seed stability
- relationship-level anomaly evidence

## Current Limitation

The current report focuses on CIC tabular baselines. It does not yet compare graph models, temporal models, adversarial robustness, drift behavior, or production serving behavior.
