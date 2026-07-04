# AEGIS-HGX Baseline Comparison Report

Generated at: `2026-07-04T04:47:49.852134+00:00`

## Purpose

This report consolidates the CIC baseline laboratory results and defines the performance bar that future graph models must beat.

## Phase Classification

| Topic | Phase | Reason |
|---|---|---|
| Baseline comparison report | Training/evaluation/test | Summarizes offline experiment evidence |
| Model metric artifacts | Training/evaluation/test | Produced by prior offline training runs |
| Calibration evidence | Training/evaluation/test | Evaluates score trustworthiness on held-out data |
| Seed-stability evidence | Training/evaluation/test | Measures variability across repeated offline runs |
| Inference API | Not changed | No serving code is modified by this report |

## Artifact Inventory

| Artifact | Path | Purpose |
|---|---|---|
| Logistic Regression CIC Baseline | `reports/logistic_cic_metrics.json` | Baseline model metrics |
| XGBoost CIC Baseline | `reports/xgboost_cic_metrics.json` | Baseline model metrics |
| Scikit-learn MLP CIC Baseline | `reports/mlp_cic_metrics.json` | Baseline model metrics |
| PyTorch MLP CIC Diagnostic Baseline | `reports/torch_mlp_cic_train_test_validation_metrics.json` | Baseline model metrics |
| PyTorch MLP Calibration Analysis | `reports/calibration/torch_mlp_cic_calibration_metrics.json` | Calibration evidence |
| PyTorch MLP Seed Stability | `reports/experiments/torch_mlp_cic_multiseed_summary.json` | Seed-stability evidence |

## Model Comparison

| Model | Family | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression CIC Baseline | linear | 0.9632 | 0.9899 | 0.8675 | 0.7826 | 0.9730 | 0.9450 | N/A | N/A |
| Scikit-learn MLP CIC Baseline | neural_network | 0.9814 | 0.9864 | 0.9600 | 0.9474 | 0.9730 | 0.9850 | N/A | N/A |
| PyTorch MLP CIC Diagnostic Baseline | neural_network | 0.9758 | 0.9877 | 0.9114 | 0.8571 | 0.9730 | 0.9650 | 6.0000 | 1.0000 |
| XGBoost CIC Baseline | gradient_boosted_trees | 0.9986 | 0.9997 | 0.9600 | 0.9474 | 0.9730 | 0.9850 | N/A | N/A |

## Initial Interpretation

This report compares the current CIC tabular baselines before the project moves into graph construction and GNN modeling.

PR-AUC is treated as the most important ranking metric because cyber attack detection is usually class-imbalanced. ROC-AUC remains useful for separability, but it can look optimistic when the negative class dominates.

Precision, recall, F1, false positives, and false negatives are threshold-dependent metrics. They describe what happens when model scores are converted into operational alerts.

A future graph model should not be considered better just because it is more complex. It should beat the strongest tabular baseline on meaningful metrics such as PR-AUC, false-positive behavior, false-negative behavior, calibration quality, and stability.

## Calibration Summary

Calibration evaluates whether model scores can be interpreted as trustworthy risk probabilities.

| Metric | Value | Interpretation |
|---|---:|---|
| Brier score | 0.0412 | Lower is better; measures squared probability error |
| Expected calibration error | 0.0702 | Lower is better; measures weighted bin-level calibration gap |
| Calibration bins | 10 | Number of probability buckets used in the reliability analysis |
| Strategy | uniform | Probability-bin construction strategy |

### Probability Summary

| Statistic | Value |
|---|---:|
| Minimum probability | 0.0000 |
| Maximum probability | 1.0000 |
| Mean probability | 0.2489 |
| Median probability | 0.0153 |

Calibration is separate from ranking. A model can have strong PR-AUC and ROC-AUC while still producing poorly calibrated probability scores.

## Seed-Stability Summary

Seed-stability analysis evaluates whether the PyTorch MLP baseline is reliable across repeated training runs.

| Metric | Mean | Std | 95% CI | Min | Max |
|---|---:|---:|---:|---:|---:|
| pr_auc | 0.8803 | 0.0859 | [0.8050, 0.9555] | 0.7471 | 0.9612 |
| roc_auc | 0.9503 | 0.0154 | [0.9368, 0.9638] | 0.9279 | 0.9683 |
| f1 | 0.8396 | 0.0431 | [0.8018, 0.8773] | 0.7907 | 0.8861 |
| precision | 0.7709 | 0.0647 | [0.7141, 0.8276] | 0.6939 | 0.8333 |
| recall | 0.9243 | 0.0226 | [0.9045, 0.9441] | 0.8919 | 0.9459 |
| false_positive | 10.4000 | 3.7815 | [7.0853, 13.7147] | 7.0000 | 15.0000 |
| false_negative | 2.8000 | 0.8367 | [2.0666, 3.5334] | 2.0000 | 4.0000 |

Low standard deviation means the baseline is stable across seeds. High standard deviation means the model may be sensitive to train/test split, initialization, dropout, batch ordering, or optimizer trajectory.

## Graph-Model Target Bar

Future graph models must justify their additional complexity. A GNN should not be treated as better simply because it is more advanced.

| Target | Current Baseline Bar |
|---|---|
| Strongest PR-AUC baseline | XGBoost CIC Baseline with PR-AUC 0.9986 |
| Strongest F1 baseline | Scikit-learn MLP CIC Baseline with F1 0.9600 |

A future graph model should aim to improve at least one of the following without severely degrading the others:

- Higher PR-AUC on held-out data.
- Better false-positive behavior at useful recall levels.
- Better false-negative behavior for attack traffic.
- More stable results across random seeds.
- Better calibration or more honest risk scores.
- More explainable relationship-level anomaly evidence.

The strongest future claim will not be: `the graph model is more complex`. The stronger claim will be: `the graph model captures relationship structure that tabular baselines miss, and the evidence shows measurable improvement.`

## Limitations

The current baseline laboratory is intentionally limited.

- The current comparison is based on CIC tabular flow features.
- The current split strategy is random rather than temporal.
- The current graph structure has not yet been constructed.
- The current calibration analysis focuses on the PyTorch MLP baseline.
- The current multi-seed stability analysis focuses on the PyTorch MLP baseline.
- The current report does not yet include graph baselines, temporal models, adversarial robustness, or drift analysis.

These limitations are acceptable at this stage because the purpose of this milestone is to establish a serious tabular baseline bar before graph modeling begins.

## Next Steps

The next milestone is Release 2: Baseline Laboratory.

Immediate next steps:

- Finalize the baseline laboratory release report.
- Confirm that all baseline artifacts are reproducible through CI.
- Use this report to define the minimum bar for graph models.
- Begin graph construction notes and graph schema design.
- Move from isolated tabular rows toward entities, edges, and relationship structure.

The next modeling phase should only claim graph-model value if graph structure produces measurable improvement over this baseline evidence.
