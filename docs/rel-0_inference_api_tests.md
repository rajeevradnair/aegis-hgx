# Inference API Tests

## Objective

AEGIS-HGX includes inference API tests to verify that the FastAPI prediction service behaves correctly at the external request boundary.

These tests protect the contract between API callers and the trained baseline model.

## Phase

These tests primarily protect the inference phase.

They are also connected to the training/test phase because the API serves a model artifact trained with a specific feature schema.

## Tested Contracts

The inference API tests verify:

* the health endpoint returns a successful response
* the health endpoint confirms the model is loaded
* a valid prediction payload is accepted
* the prediction response contains stable response keys
* prediction values are valid binary class labels
* classification values match prediction values
* suspicious probability is between `0.0` and `1.0`
* invalid numeric values are rejected
* missing required fields are rejected
* wrong field types are rejected

## Why This Matters

Model correctness is not enough for a deployed ML system.

A trained model can work locally while the deployed API still fails because of bad request validation, unstable response keys, missing fields, or malformed payloads.

API tests protect the runtime boundary before deployment and before automated CI/CD.

## Request Contract

The prediction endpoint expects one cyber event payload containing:

```text
user_id
host_id
process_name
event_type
source_ip
destination_ip
bytes_in
bytes_out
event_hour
is_business_hour
```

These fields must match the model feature contract used during training.

## Response Contract

The prediction endpoint returns:

```text
prediction
classification
suspicious_probability
```

The response must remain stable because future dashboards, monitoring jobs, and client systems may depend on these keys.

## Key Test File

```text
tests/test_inference_api.py
```

## How to Run

Run only inference API tests:

```bash
python -m pytest tests/test_inference_api.py
```

Run the full test suite:

```bash
python -m pytest
```
