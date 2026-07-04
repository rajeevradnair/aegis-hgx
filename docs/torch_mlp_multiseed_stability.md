# PyTorch MLP Seed Stability

## Purpose

This document describes the PyTorch MLP multi-seed stability experiment for AEGIS-HGX.

The goal is to determine whether the CIC PyTorch MLP baseline is stable across random seeds.

## Phase

This is an offline training/evaluation/test experiment. It does not change inference behavior.

## Inputs

- `data/processed/cicids2017/cic_tabular_features.csv`
- `configs/torch_mlp_seed_runs_cic.yaml`

## Outputs

- `reports/experiments/torch_mlp_cic_seed_runs.csv`
- `reports/experiments/torch_mlp_cic_seed_summary.json`

## Why Multi-Seed Runs Matter

A single neural-network training run can be misleading because performance depends on:

- train/validation/test split
- weight initialization
- mini-batch ordering
- dropout randomness
- optimizer trajectory
- small-sample variance

Multi-seed evaluation turns a single metric into a distribution of metrics.

## Metrics

The experiment records the following test-set metrics for each seed:

- test loss
- ROC-AUC
- PR-AUC
- accuracy
- precision
- recall
- F1
- true positives
- false positives
- true negatives
- false negatives

The summary report computes:

- mean
- standard deviation
- minimum
- maximum
- standard error
- approximate 95% confidence interval

## Interpretation

High mean and low standard deviation means the model is strong and stable.

High mean and high standard deviation means the model may be seed-sensitive.

Low mean and low standard deviation means the model is consistently weak.

Low mean and high standard deviation means the model is weak and unstable.

## Current Limitation

This experiment currently focuses on the PyTorch MLP CIC baseline. Later baseline comparison work can extend the same stability framework to logistic regression, XGBoost, and scikit-learn MLP.