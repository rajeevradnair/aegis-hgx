# Model Shape Tests

## Objective

AEGIS-HGX includes model input shape tests to prevent silent ML bugs before training, evaluation, and inference.

These tests verify that the dataset, feature table, target labels, train/test split, and model predictions stay aligned.

## Phase

These tests primarily protect the training/test phase.

They are also indirectly related to inference because the inference service must eventually send payloads that match the same feature contract used during training.

## Tested Contracts

The model shape tests verify:

* feature rows match dataset rows
* target rows match dataset rows
* feature columns match the configured numeric and categorical features
* target column is not included in the feature table
* train/test split preserves total row count
* train feature rows match train target rows
* test feature rows match test target rows
* trained model produces one prediction per test row
* predictions are valid binary class labels
* missing required feature columns raise a clear error

## Why This Matters

ML pipelines can fail silently.

A model may train successfully even when the wrong columns, wrong order, or leaked target values are present.

Shape tests protect against these errors before metrics, MLflow runs, or model artifacts become misleading.

## Training/Test Contract

The expected tabular model contract is:

```text id="cm0grd"
dataset rows = feature rows = target rows
```

and:

```text id="2ldefp"
X shape = rows × configured features
y shape = rows
```

## Inference Relationship

Inference does not directly use these tests today.

However, the trained model artifact expects the same feature schema that was validated during training.

Future API tests should verify that a request payload can be converted into the same feature structure expected by the trained model.

## Key Test File

```text id="coar8m"
tests/test_model_shapes.py
```

## How to Run

Run only model shape tests:

```bash id="p4qzoa"
python -m pytest tests/test_model_shapes.py
```

Run the full test suite:

```bash id="9q44ha"
python -m pytest
```
