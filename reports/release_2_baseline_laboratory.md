# AEGIS-HGX Release 2 - Baseline Laboratory

Generated at: `2026-07-04T05:02:58.768317+00:00`

## Release Summary

This release closes the CIC baseline laboratory milestone for AEGIS-HGX.

The baseline laboratory moved the project from synthetic experiments into public cyber intrusion-detection data and established a serious tabular-model evidence base before graph modeling begins.

| Field | Value |
|---|---|
| Phase name | Baseline Laboratory |
| Dataset | cicids2017 |
| Primary phase | Training/evaluation/test reporting |
| Inference impact | Indirect; no serving code changed |

The release is complete when ingestion, tabular feature construction, baseline training, training diagnostics, calibration analysis, seed-stability analysis, and baseline comparison are all represented by reproducible artifacts.

## Phase Classification

| Component | Phase | Reason |
|---|---|---|
| CIC ingestion | Data preparation | Converts public CIC files into project-ready data |
| Tabular feature construction | Training/evaluation preparation | Produces model-ready features |
| Logistic, XGBoost, and MLP baselines | Training/evaluation/test | Trains and evaluates offline models |
| PyTorch MLP diagnostics | Training/evaluation/test | Captures training behavior and threshold behavior |
| Calibration analysis | Training/evaluation/test | Evaluates score trustworthiness on held-out data |
| Seed-stability analysis | Training/evaluation/test | Measures run-to-run variability |
| Baseline comparison report | Training/evaluation/test reporting | Summarizes offline evidence |
| Inference API | Not changed | This release does not modify serving behavior |

## Artifact Inventory

| Artifact | Path | Purpose |
|---|---|---|
| Logistic Regression CIC Baseline | `reports/logistic_cic_metrics.json` | Baseline model metrics |
| XGBoost CIC Baseline | `reports/xgboost_cic_metrics.json` | Baseline model metrics |
| Scikit-learn MLP CIC Baseline | `reports/mlp_cic_metrics.json` | Baseline model metrics |
| PyTorch MLP CIC Diagnostic Baseline | `reports/torch_mlp_cic_train_test_validation_metrics.json` | Baseline model metrics |
| PyTorch MLP Calibration Analysis | `reports/calibration/torch_mlp_cic_calibration_metrics.json` | Score calibration evidence |
| PyTorch MLP Seed Stability | `reports/experiments/torch_mlp_cic_multiseed_summary.json` | Multi-seed stability evidence |
| Baseline Comparison Report | `reports/baseline_comparison_report.md` | Baseline comparison and graph-target bar |

## Implementation Journey

This release covers the complete baseline laboratory path:

1. Public CIC data was ingested into the project.
2. Raw CIC flow columns were normalized into clean tabular features.
3. A logistic regression baseline established a simple linear reference point.
4. An XGBoost baseline established a strong tabular tree-based reference point.
5. A scikit-learn MLP baseline established a quick neural-network reference point.
6. A PyTorch MLP diagnostic trainer exposed the training loop, validation behavior, threshold behavior, PR/ROC curves, and model lineage.
7. Calibration analysis tested whether raw model scores behaved like trustworthy risk probabilities.
8. Multi-seed stability analysis tested whether PyTorch MLP results were consistent across random seeds.
9. A baseline comparison report synthesized the evidence and defined what future graph models must beat.

## Baseline Model Summary

| Model | Family | PR-AUC | ROC-AUC | F1 | Precision | Recall | Accuracy | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression CIC Baseline | linear | 0.9632 | 0.9899 | 0.8675 | 0.7826 | 0.9730 | 0.9450 | N/A | N/A |
| Scikit-learn MLP CIC Baseline | neural_network | 0.9814 | 0.9864 | 0.9600 | 0.9474 | 0.9730 | 0.9850 | N/A | N/A |
| PyTorch MLP CIC Diagnostic Baseline | neural_network | 0.9719 | 0.9872 | 0.9114 | 0.8571 | 0.9730 | 0.9650 | 6.0000 | 1.0000 |
| XGBoost CIC Baseline | gradient_boosted_trees | 0.9986 | 0.9997 | 0.9600 | 0.9474 | 0.9730 | 0.9850 | N/A | N/A |

### Baseline Bar

- Strongest PR-AUC baseline: XGBoost CIC Baseline with PR-AUC 0.9986.
- Strongest F1 baseline: Scikit-learn MLP CIC Baseline with F1 0.9600.

PR-AUC is treated as especially important because cyber attack detection is usually class-imbalanced. F1, precision, recall, false positives, and false negatives describe thresholded alert behavior.

## Calibration Summary

Calibration analysis evaluates whether model scores can be interpreted as trustworthy risk probabilities.

| Metric | Value | Interpretation |
|---|---:|---|
| Brier score | 0.0364 | Lower is better; measures squared probability error |
| Expected calibration error | 0.0606 | Lower is better; measures weighted bin-level calibration gap |
| Calibration bins | 10 | Number of probability buckets used in reliability analysis |
| Strategy | uniform | Probability-bin construction strategy |

### Probability Summary

| Statistic | Value |
|---|---:|
| Minimum probability | 0.0000 |
| Maximum probability | 1.0000 |
| Mean probability | 0.2416 |
| Median probability | 0.0103 |

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

## Baseline Comparison Summary

The baseline comparison report was generated and is available as a release input.

The key conclusion is that future graph models must justify their additional complexity. A GNN should not be treated as better simply because it is more advanced; it must show measurable improvement over tabular baselines.

## Graph-Readiness Statement

The project is ready to begin graph construction because the tabular baseline laboratory has established a measurable baseline bar.

| Target | Current Baseline Bar |
|---|---|
| Strongest PR-AUC baseline | XGBoost CIC Baseline with PR-AUC 0.9986 |
| Strongest F1 baseline | Scikit-learn MLP CIC Baseline with F1 0.9600 |

Future graph models should aim to improve at least one of the following without severely degrading the others:

- PR-AUC on held-out data.
- False-positive behavior at useful recall levels.
- False-negative behavior for attack traffic.
- Seed stability.
- Calibration quality.
- Relationship-level anomaly evidence.

The next phase should move from isolated tabular flow rows toward nodes, edges, entity relationships, and graph-structured evidence.

## Limitations

The baseline laboratory is complete for its intended purpose, but it is intentionally limited.

- The release focuses on CIC tabular flow features.
- The release does not yet include graph construction.
- The release does not yet include temporal train/test splitting.
- The release does not yet include graph neural networks.
- Calibration and seed-stability evidence are currently strongest for the PyTorch MLP baseline.
- The release does not yet include adversarial robustness, drift evaluation, or production-scale profiling.

These limitations are acceptable because this milestone is meant to close the tabular baseline lab before the graph research phase begins.

## Next-Phase Handoff

The next phase begins graph construction from events.

The immediate next objectives are:

- Define graph construction concepts from first principles.
- Decide which CIC entities become nodes.
- Decide which flow relationships become edges.
- Build node and edge tables.
- Add graph inspection and visualization.
- Convert the graph into model-ready structures for future GNN baselines.

