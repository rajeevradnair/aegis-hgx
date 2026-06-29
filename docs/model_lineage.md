# Model and Data Lineage

## Objective

AEGIS-HGX records model and data lineage so each model artifact can be traced back to the exact data, configuration, code version, metrics, and MLflow run that produced it.

This lineage layer is required before meaningful drift monitoring because a drift report must know which model version and reference dataset it is monitoring.

## Phase

Model and data lineage primarily supports the training/test infrastructure phase.

It is also indirectly connected to inference because the served model artifact should eventually expose its model version, MLflow run ID, and lineage manifest path.

## Why Lineage Matters

A model artifact alone is not enough.

A saved model file answers:

```text
Where is the model?
```

A lineage manifest answers:

```text
Which data trained this model?
Which configuration trained it?
Which code commit produced it?
Which MLflow run owns it?
Which metrics belong to it?
Which model artifact was saved?
Can this model version be reproduced?
```

## Current Lineage Architecture

```text
Git
  tracks code, configs, docs, tests, and DVC metadata

DVC
  tracks dataset content and dataset version metadata

MLflow
  tracks experiment run, parameters, metrics, and artifacts

Lineage manifest
  binds Git metadata, DVC metadata, config hashes, metrics, model artifact hash, and MLflow run metadata
```

## Current Lineage Manifest

The current manifest is written to:

```text
artifacts/lineage/logistic_baseline_manifest.json
```

It is also logged to MLflow under:

```text
run_evidence/lineage
```

## Manifest Sections

The lineage manifest contains:

* lineage manifest version
* creation timestamp
* model metadata
* MLflow experiment and run metadata
* Git commit, branch, and dirty status
* training data path, hash, shape, columns, and DVC metadata
* training config hash
* data generation config hash
* metrics path, hash, and values
* feature-store placeholder metadata
* reproducibility metadata

## Why Hashes Are Used

A file path is not a version.

This path can change over time:

```text
data/processed/synthetic_events.csv
```

A hash identifies the actual file contents:

```text
sha256: ...
```

The lineage manifest stores hashes for:

* model artifact
* training data
* training config
* data generation config
* metrics file

## Git Metadata

The manifest records:

* Git commit
* Git branch
* whether the working tree was dirty

If the working tree is dirty, the model may have been trained using uncommitted code or configuration changes.

## DVC Metadata

DVC tracks the synthetic training dataset.

Git tracks the DVC metadata file:

```text
data/processed/synthetic_events.csv.dvc
```

The lineage manifest records:

* DVC file path
* DVC output path
* DVC output hash
* DVC output size
* whether the data is DVC-tracked

## Feature Store Placeholder

AEGIS-HGX does not use Feast yet.

For now, the manifest records:

```text
provider = local_snapshot
```

This keeps the lineage structure compatible with a future feature-store layer without adding unnecessary complexity too early.

## Relationship to Drift Monitoring

Drift monitoring compares:

```text
reference data
→ current data
```

The lineage manifest defines the reference side:

```text
model version
→ training dataset path/hash
→ DVC dataset metadata
→ config hashes
→ MLflow run ID
```

This prevents drift reports from being disconnected from the model version being monitored.

## Current Tests

Lineage smoke tests verify that:

* the manifest exists
* required top-level sections are present
* the model section links to the model artifact
* model artifact hash is present
* the training data section links to the DVC-tracked dataset
* training data hash is present

## Current Limitations

The current lineage system is local-first.

It does not yet include:

* DVC remote storage
* S3-backed dataset storage
* MLflow Model Registry versioning
* production inference logs
* prediction-level lineage
* OpenLineage integration
* feature-store registry metadata

These can be added later as the project moves toward public datasets, feature stores, and production-like orchestration.

## Mental Model

```text
Git tells us which code existed.
DVC tells us which data existed.
MLflow tells us which experiment ran.
The lineage manifest binds them into one model-version receipt.
```
