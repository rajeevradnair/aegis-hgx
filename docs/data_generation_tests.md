# Data Generation Tests

## Objective

AEGIS-HGX includes unit tests for synthetic cyber event generation so the data pipeline remains trustworthy before training, evaluation, and inference.

## Phase

These tests primarily protect the training/test phase.

They do not directly test inference behavior, but they indirectly improve inference trust because the inference service later serves models trained on generated data.

## Tested Data Contract

The synthetic event generator must produce a dataset with:

* the configured number of rows
* the required column order
* binary labels
* both normal and suspicious examples
* valid event hours between `0` and `23`
* boolean-compatible business-hour flags
* non-negative byte counts
* reproducible output for the same seed
* anomaly rate close to the configured value
* CSV output that can be saved and reloaded

## Why This Matters

Synthetic data does not prove real-world cyber detection quality.

Its purpose is to validate platform mechanics:

```text id="bvlm9z"
data generation
→ data contract
→ training
→ MLflow tracking
→ model artifact
→ inference service
```

If the generator breaks, downstream model metrics become untrustworthy.

## Reproducibility

The generator should satisfy:

```text id="op1vtt"
same config + same seed = same dataset
```

This makes experiment comparisons more reliable.

## Key Test File

```text id="dajktd"
tests/test_generate_synthetic_events.py
```

## How to Run

Run only data-generation tests:

```bash id="qvzv1f"
python -m pytest tests/test_generate_synthetic_events.py
```

Run the full test suite:

```bash id="xpjc2s"
python -m pytest
```
