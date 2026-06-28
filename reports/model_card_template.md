# Model Card Template
Model overview
Intended use
Out-of-scope use
Training data
Evaluation data
Metrics
Limitations
Failure modes
Monitoring - Concept drift, Distribution drift, Schema drift, latency
Ethical considerations
Operational considerations
Security considerations
Versioning and ownership

## Model Overview

| Field | Value |
|---|---|
| Model name | AEGIS-HGX Logistic Regression Baseline |
| Model type | Logistic regression classifier |
| Model artifact | `artifacts/models/logistic_baseline.joblib` |
| Task | Binary classification of synthetic cyber events as normal or suspicious |
| Current status | Baseline model for experimentation and platform validation |
| Primary users | ML engineers, security ML researchers, platform engineers |
| Deployment context | Local FastAPI inference service and reproducible CI workflow |

## Model Purpose

This model is the first baseline classifier in AEGIS-HGX.

Its purpose is to validate the end-to-end machine learning platform mechanics:

```text
synthetic event generation
→ feature preparation
→ model training
→ experiment tracking
→ model artifact creation
→ inference API serving
→ automated testing
```

## Intended Use

This baseline model is intended for:

- validating the AEGIS-HGX training pipeline
- validating model artifact creation
- validating MLflow experiment tracking
- validating FastAPI inference serving
- validating API contract tests
- validating CI reproducibility
- demonstrating the initial end-to-end MLOps skeleton

The model may be used for local experimentation, engineering demonstrations, and controlled portfolio evaluation.

## Out-of-Scope Use

This model is not intended for:

- real-world incident response
- production security monitoring
- autonomous blocking or containment decisions
- customer-facing threat detection
- legal, compliance, or employment decisions
- high-stakes operational decisions
- claims of real-world attack detection performance

The model has only been validated on synthetic data and should not be treated as a production-grade cyber defense system.

## Training Data

| Field | Value |
|---|---|
| Dataset source | Synthetic cyber event generator |
| Dataset file | `data/processed/synthetic_events.csv` |
| Data type | Tabular cyber event records |
| Label type | Binary labels: `0 = normal`, `1 = suspicious` |
| Current real-world coverage | None |
| Public IDS dataset coverage | Not yet included |

The current training data is generated synthetically to validate platform mechanics and baseline modeling behavior.

The synthetic dataset includes entities such as users, hosts, processes, event types, source IPs, destination IPs, byte counts, event hours, business-hour flags, and binary labels.

## Evaluation Data

| Field | Value |
|---|---|
| Evaluation source | Held-out split from synthetic dataset |
| Split method | Config-driven train/test split |
| Test size | Defined in `configs/baseline_logistic.yaml` |
| Random seed | Defined in `configs/baseline_logistic.yaml` |
| Evaluation status | Baseline evaluation only |

The evaluation data is not independent real-world traffic.

Current evaluation results should be interpreted as a check of pipeline mechanics, not proof of production cyber detection performance.

## Data Assumptions

The current model assumes:

- event rows have the required configured feature columns
- labels are binary
- synthetic suspicious patterns are meaningfully separable from synthetic normal patterns
- training and evaluation data come from the same synthetic generation process
- inference payloads follow the same feature schema used during training

## Metrics

| Metric | Current Status | Notes |
|---|---|---|
| Accuracy | Available from baseline training output | Useful but insufficient for imbalanced anomaly detection |
| Precision | Available from baseline training output | Measures how many suspicious predictions are actually suspicious |
| Recall | Available from baseline training output | Measures how many suspicious events are caught |
| F1 score | Available from baseline training output | Balances precision and recall |
| ROC-AUC | Future metric | Useful for separability but can be misleading under severe imbalance |
| PR-AUC | Future metric | More important for rare cyber anomaly detection |
| Calibration / Brier score | Future metric | Needed before treating probabilities as trustworthy risk scores |
| False positive rate | Future metric | Important for analyst fatigue |
| False negative rate | Future metric | Important for missed attack risk |
| Latency | Available from inference/API testing or deployment checks | Important for runtime serving behavior |

Current metrics should be interpreted as baseline synthetic-data metrics only.

They do not establish real-world cyber detection quality.

## Known Limitations

The current baseline has the following limitations:

- trained only on synthetic data
- evaluated only on a held-out synthetic split
- does not use public IDS datasets yet
- does not model temporal event sequences
- does not model graph relationships between users, hosts, processes, and network entities
- does not include adversarial robustness testing
- does not include drift monitoring yet
- does not include calibrated probability validation
- does not include threshold tuning for operational alerting
- does not include analyst feedback loops
- does not include real SOC workflow integration

## Interpretation Guidance

This model should be interpreted as a baseline engineering artifact.

It proves that AEGIS-HGX can:

- generate data
- train a model
- track experiments
- save a model artifact
- serve predictions through an API
- validate request and response contracts
- run tests through CI

It does not prove that AEGIS-HGX can detect real-world attacks.

## Current Trust Level

| Dimension | Trust Level | Reason |
|---|---|---|
| Pipeline mechanics | Moderate | Data generation, training, API, tests, and CI exist |
| Synthetic evaluation | Moderate | Evaluation is controlled but artificial |
| Real-world detection | Low | No public or production security dataset validation yet |
| Runtime API contract | Moderate | API tests validate expected behavior |
| Probability trust | Low | Calibration has not been validated |
| Operational readiness | Low | Monitoring, thresholding, and real-world evaluation are not complete |

## Failure Modes

The current baseline may fail in the following ways:

### False Positives

Normal behavior may be incorrectly classified as suspicious.

Potential impact:

- unnecessary alerts
- analyst fatigue
- reduced trust in the system
- wasted investigation time

### False Negatives

Suspicious behavior may be incorrectly classified as normal.

Potential impact:

- missed attack behavior
- delayed investigation
- false sense of security

### Distribution Shift

Runtime events may differ from the synthetic training data.

Examples:

- new process names
- new host naming patterns
- different byte-count distributions
- different login-hour patterns
- new event types
- unseen IP address patterns

### Schema Drift

The incoming data schema may change.

Examples:

- missing fields
- renamed fields
- changed data types
- changed categorical values
- different timestamp or event-hour representation

### Synthetic Pattern Overfitting

The model may learn artificial patterns from the synthetic generator rather than general cyber behavior.

This is expected at the current stage and should be treated as a known limitation.

### Probability Misinterpretation

The `suspicious_probability` output should not yet be interpreted as a calibrated real-world risk score.

Probability calibration has not been validated.

## Monitoring Recommendations

Future monitoring should include:

- input schema validation
- missing-field rates
- invalid-payload rates
- feature distribution drift
- prediction distribution drift
- suspicious alert rate
- false-positive review rate
- false-negative review rate when labels become available
- model latency
- API error rate
- model artifact version served by the API

## Required Future Validation

Before this model can support stronger claims, AEGIS-HGX should add:

- public IDS dataset ingestion
- real baseline comparison
- PR-AUC and ROC-AUC reporting
- threshold tuning
- calibration analysis
- drift monitoring
- temporal split evaluation
- graph-based feature evaluation
- adversarial robustness testing
- error analysis with false-positive and false-negative examples

## Versioning and Ownership

| Field | Value |
|---|---|
| Model card version | `0.1` |
| Model version | Baseline logistic regression |
| Model artifact | `artifacts/models/logistic_baseline.joblib` |
| Config file | `configs/baseline_logistic.yaml` |
| Training data source | Synthetic event generator |
| Owner | AEGIS-HGX project maintainer |
| Review status | Draft |
| Approval status | Not approved for production use |

## Production Readiness Decision

Current decision:

```text
Not production ready
```

Reason:

The current model is trained and evaluated only on synthetic data. It is suitable for platform validation, local experimentation, API testing, CI reproducibility, and portfolio demonstration, but not for real-world cyber defense decisions.

## Summary

This model card documents the current baseline honestly.

The baseline is useful because it proves the AEGIS-HGX platform can train, track, serve, test, and reproduce a model.

The baseline is limited because it has not yet been validated on real public security datasets, temporal behavior, graph relationships, adversarial conditions, or calibrated operational thresholds.
