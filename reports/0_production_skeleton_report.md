# AEGIS-HGX Production Skeleton Report

## Objective
AEGIS-HGX currently validates the production skeleton for a cyber anomaly detection platform. The purpose of this milestone is not to prove advanced detection quality yet. The purpose is to prove that the project has a reproducible foundation for data generation, baseline training, experiment tracking, model serving, containerization, cloud storage, and cloud deployment.

This production skeleton creates the platform that later graph-based, temporal, and research-grade anomaly detection models can build on.

## Current System State
The current system includes the following working components:

A synthetic cyber event generator that creates tabular security events.
A baseline logistic regression training pipeline.
MLflow experiment tracking for training parameters, metrics, and artifacts.
A FastAPI inference API with validated request and response contracts.
Docker images for training and inference.
Docker Compose orchestration for local containerized execution.
An AWS S3 data lake layout using Bronze, Silver, and Gold zones.
A deployed AWS ECS Fargate inference service behind an Application Load Balancer.
A model-loading path where the deployed service loads the trained model artifact from S3.

The system is intentionally simple at the model layer, but production-oriented at the platform layer.

## Architecture Overview
The current architecture has three major paths.

Local training path

Synthetic cyber events are generated locally and saved as a processed dataset. The baseline training pipeline reads this dataset, trains a logistic regression classifier, evaluates it, writes metrics, saves the model artifact, and records experiment evidence through MLflow.

Flow:

synthetic event generator
    -> processed dataset
    -> logistic regression training pipeline
    -> metrics report
    -> joblib model artifact
    -> MLflow run evidence
Local serving path

The trained model is loaded by a FastAPI application. The API exposes a health endpoint and a prediction endpoint. Request validation is handled before model inference, so malformed payloads are rejected before they reach the model.

Flow:

trained model artifact
    -> FastAPI startup
    -> in-memory sklearn pipeline
    -> /api/v1/baseline_logistic/health
    -> /api/v1/baseline_logistic/predict
Cloud deployment path

The inference container image is pushed to Amazon ECR. AWS ECS Fargate runs the container as a service. An Application Load Balancer exposes the service over HTTP. During startup, the service downloads the trained model from S3 and loads it into memory.

Flow:

Docker inference image
    -> Amazon ECR
    -> ECS Fargate task
    -> FastAPI application
    -> S3 model artifact
    -> Application Load Balancer
    -> public /api/v1/baseline_logistic/health and /api/v1/baseline_logistic/predict endpoints

## Data Pipeline

## Training Pipeline

## Experiment Tracking

## Inference API

## Containerization

## Cloud Data Lake

## AWS Inference Deployment
The deployed service exposes:

/api/v1/baseline_logistic/health
/api/v1/baseline_logistic/predict

The health endpoint confirms that the service is running and that the model was loaded. The prediction endpoint confirms that the deployed API can accept a synthetic cyber event payload and return a structured prediction response.

## Verification Evidence
The following evidence validates the current production skeleton.

Local dataset

Expected local path:

data/processed/synthetic_events.csv

This file confirms that the synthetic data pipeline can produce a usable dataset for early platform validation.

Local model artifact

Expected local path:

artifacts/models/logistic_baseline.joblib

This file confirms that the baseline training pipeline can produce a reusable model artifact.

Metrics report

Expected local path:

reports/baseline_logistic_metrics.json

This file confirms that the baseline model is evaluated and that metrics are persisted outside the training process.

MLflow tracking

local paths:

mlflow/mlflow.db
mlflow/mlruns/

These files confirm that training runs are tracked with experiment metadata, parameters, metrics, and artifacts.

S3 data lake

S3 paths:

s3://aegis-hgx/bronze/
s3://aegis-hgx/silver/
s3://aegis-hgx/gold/
s3://aegis-hgx/metadata/
s3://aegis-hgx/configs/
s3://aegis-hgx/artifacts/
s3://aegis-hgx/experiments/

The Bronze zone contains the uploaded synthetic dataset:

s3://aegis-hgx/bronze/synthetic/events/synthetic_events.csv

The artifacts zone contains the trained model used by the deployed inference service:

s3://aegis-hgx/artifacts/models/logistic_baseline.joblib

## Current Metrics
The current baseline model is a logistic regression classifier trained on synthetic cyber event data. These metrics validate that the training, evaluation, and reporting pipeline works end to end.

Metrics are stored in:

reports/baseline_logistic_metrics.json

The expected metric fields are:

test_rows
positive_rows
accuracy
precision
recall
f1
roc_auc
pr_auc
confusion_matrix

Metric interpretation:

Accuracy shows the percentage of total predictions classified correctly.
Precision shows how many predicted suspicious events were actually suspicious.
Recall shows how many suspicious events were detected.
F1 harmonically balances precision and recall.
ROC-AUC measures general class separability.
PR-AUC is especially important for cyber anomaly detection because suspicious events are usually rare.
The confusion matrix shows true negatives, false positives, false negatives, and true positives.

These metrics should not be interpreted as real-world cyber detection quality yet. They were measured on synthetic data and primarily validate platform correctness, not production detection performance.

## Known Limitations / TODOs
The current system is intentionally limited.

Synthetic data only

The dataset is synthetic and rule-generated. It is useful for testing pipelines, schemas, training logic, MLflow tracking, and deployment mechanics, but it does not prove real-world threat detection quality.

Simple baseline model

The current model is logistic regression. This is appropriate as a first baseline because it is simple, explainable, and fast. It is not yet a graph model, temporal model, or research-grade anomaly detector.

No real public cyber dataset yet

The system has not yet ingested CSE-CIC-IDS2018, CICIDS2017, or another public security dataset. Real dataset ingestion is required before making stronger claims about detection capability.

Local MLflow tracking

MLflow currently uses local tracking storage. Cloud-hosted or shared MLflow tracking is not implemented yet. The S3 prefix for future MLflow artifacts exists, but MLflow itself has not been moved to cloud-backed tracking.

Public HTTP endpoint

The deployed ECS service is exposed through HTTP for early validation. It does not yet use HTTPS, a custom domain, authentication, authorization, WAF, or rate limiting.

Basic infrastructure posture

The ECS deployment uses a simplified networking setup suitable for early portfolio validation. A production-grade deployment would use private subnets, tighter ingress, VPC endpoints or NAT, stronger secrets handling, and more restrictive IAM policies.

Minimal observability

CloudWatch logs are available, but structured request logs, request IDs, latency metrics, alerting, tracing, and dashboards are not implemented yet.

No CI/CD yet

Build, test, container push, and deployment are still manually executed. Automated GitHub Actions or another CI/CD workflow is not yet in place.

## Next Engineering Steps
The next engineering phase should strengthen reproducibility, testing, and governance before adding advanced models.

Next steps:

Add config-driven experiments so training behavior is controlled by YAML rather than hardcoded values.
Add stronger unit tests for data generation and model input shapes.
Add API contract tests for deployed and local inference behavior.
Add CI/CD so tests run automatically on every commit.
Add model card and data card templates.
Add basic drift monitoring using synthetic reference and current splits.
Add threshold tuning so alert behavior is explainable.
Move from synthetic-only data toward a public cyber dataset sample.
Begin baseline comparison across logistic regression, tree-based models, and neural baselines.