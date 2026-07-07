# Continuous Integration

## Objective

AEGIS-HGX uses GitHub Actions CI to automatically run project checks whenever code is pushed or a pull request is opened.

The goal is to make testing reproducible instead of relying only on manual local commands.

## Phase

Continuous integration is an engineering automation layer.

It protects both the training/test phase and the inference phase because it runs tests across the data pipeline, model pipeline, and API boundary.

## Workflow File

```text
.github/workflows/ci.yml
```

## What CI Does

The CI workflow performs the following steps:

```text
check out repository
→ set up Python
→ install project dependencies
→ generate synthetic data
→ train baseline model
→ run pytest
```

## Why Synthetic Data Is Generated in CI

GitHub Actions runners start from a clean machine.

The workflow generates synthetic data during CI so tests do not depend on files that only exist on a developer's laptop.

## Why the Baseline Model Is Trained in CI

The inference API tests require a model artifact.

The workflow trains the baseline model before running tests so the API can load the expected model artifact in a reproducible way.

## Tests Protected by CI

CI runs the full test suite, including:

* data generation tests
* model input shape tests
* inference API contract tests
* baseline training compatibility tests

## Why This Matters

A project can pass tests locally while failing in a clean environment because of hidden local state, missing artifacts, missing dependencies, or environment differences.

CI catches those problems early.

## Current CI Command

```bash
python -m pytest -ra
```

The `-ra` flag shows useful summary information for non-passing or noteworthy test outcomes.

## How to Run Locally

Run the same preparation steps locally:

```bash
python pipelines/generate_synthetic_events.py --config configs/data_generation.yaml
python -m aegis_hgx.models.baselines.train_logistic_baseline --config configs/baseline_logistic.yaml
python -m pytest -ra
```

## Mental Model

```text
Local tests prove the project works on your machine.
CI proves the project can rebuild and test itself on a clean machine.
```
