# Config-Driven Experiments

## Objective

AEGIS-HGX uses config-driven experiments so training behavior is controlled by explicit YAML files instead of hardcoded Python values.

This improves reproducibility, experiment comparison, MLflow tracking, and future model development.

## Training/Test Phase

The config controls training and evaluation settings, including:

* input dataset path
* target column
* numeric feature columns
* categorical feature columns
* train/test split size
* random seed
* model hyperparameters
* metrics output path
* model artifact path
* MLflow experiment settings

The current baseline config is:

```text
configs/baseline_logistic.yaml
```

## Inference Phase Relationship

The inference service does not directly use the training config today.

Instead, inference uses the model artifact produced by the training pipeline:

```text
training config -> trained model artifact -> inference service
```

This means the config improves inference trust indirectly by making the model artifact traceable and reproducible.

## Validation

The training config is validated with a typed schema before training starts.

The schema is defined in:

```text
src/aegis_hgx/models/baselines/config_schema.py
```

Validation catches problems such as:

* missing config sections
* invalid train/test split sizes
* invalid model settings
* missing tracking settings
* malformed experiment definitions

For example, this is invalid:

```yaml
split:
  test_size: 2.0
```

The test size must be greater than `0.0` and less than `1.0`.

## MLflow Tracking

MLflow records the experiment evidence, including:

* parameters
* scalar metrics
* config file
* metrics report
* joblib model artifact
* MLflow-formatted model artifact
* confusion matrix figure

This allows future runs to be compared and audited.

## How to Run

Run with the default config:

```bash
python -m aegis_hgx.models.baselines.train_logistic_baseline
```

Run with an explicit config:

```bash
python -m aegis_hgx.models.baselines.train_logistic_baseline \
  --config configs/baseline_logistic.yaml
```

## Why This Matters

As AEGIS-HGX grows into graph models, temporal models, ablations, threshold tuning, and multi-seed studies, experiment discipline becomes essential.

The rule is:

```text
Python code defines the reusable training engine.
YAML config defines one experiment.
MLflow records what happened.
```
