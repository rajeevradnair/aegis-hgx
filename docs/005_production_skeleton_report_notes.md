AEGIS-HGX Production Skeleton Report Notes

Objective
Create the first production skeleton report for AEGIS-HGX.

Report File
reports/0_production_skeleton_report.md

Purpose
The report proves that AEGIS-HGX has a working platform foundation before advanced graph, temporal, and research-grade modeling begins.

The report does not claim real-world cyber detection quality yet. It proves the production skeleton exists.

Current System Evidence

Data
- Synthetic cyber event generator exists.
- Local dataset path:
  data/processed/synthetic_events.csv
- Bronze S3 dataset path:
  s3://aegis-hgx/bronze/synthetic/events/synthetic_events.csv

Training
- Logistic regression baseline training pipeline exists.
- Model artifact path:
  artifacts/models/logistic_baseline.joblib
- S3 model artifact path:
  s3://aegis-hgx/artifacts/models/logistic_baseline.joblib

Metrics
- Metrics report path:
  reports/baseline_logistic_metrics.json
- Expected metrics:
  test_rows
  positive_rows
  accuracy
  precision
  recall
  f1
  roc_auc
  pr_auc
  confusion_matrix

Experiment Tracking
- MLflow local tracking exists.
- Local paths:
  mlflow/mlflow.db
  mlflow/mlruns/
- MLflow currently validates local experiment tracking.
- Future cloud artifact storage can use:
  s3://aegis-hgx/experiments/mlflow-artifacts/

Inference API
- FastAPI serving application exists.
- Endpoint contracts:
  /api/v1/baseline_logistic/health
  /api/v1/baseline_logistic/predict
- The API validates requests before model inference.
- Invalid input returns validation errors before reaching the model.

Containerization
- Training and inference are containerized.
- Files:
  Dockerfile.training
  Dockerfile.inference
  compose.yaml

Cloud Data Lake
- Existing S3 bucket:
  s3://aegis-hgx
- Main prefixes:
  bronze/
  silver/
  gold/
  metadata/
  configs/
  artifacts/
  experiments/

AWS Deployment
- Inference service deployed to ECS Fargate.
- Container image stored in ECR.
- Application exposed through an Application Load Balancer.
- Model artifact loaded from S3 using the ECS task role.
- Runtime logs available in CloudWatch Logs.

Key Limitations
- Synthetic data only.
- Logistic regression only.
- No public cyber dataset yet.
- MLflow tracking is still local.
- ECS endpoint uses public HTTP for early validation.
- No CI/CD yet.
- Minimal observability.
- No drift monitoring yet.
- No model card or data card yet.
- No graph or temporal model yet.

Recommended Next Steps
1. Add config-driven experiments.
2. Add stronger unit tests for data generation.
3. Add model input shape tests.
4. Add API tests.
5. Add CI/CD.
6. Add model card and data card templates.
7. Add basic drift monitoring.
8. Add threshold tuning.
9. Ingest a public cyber dataset sample.
10. Start baseline comparison across model families.

Verification Commands

Check report:
ls reports/production_skeleton_report.md

Check local artifacts:
ls data/processed/synthetic_events.csv
ls artifacts/models/logistic_baseline.joblib
cat reports/baseline_logistic_metrics.json

Check S3:
aws s3 ls s3://aegis-hgx/
aws s3 ls s3://aegis-hgx/bronze/synthetic/events/
aws s3 ls s3://aegis-hgx/artifacts/models/

Check ECS endpoints:
cd infra/aws/ecs
terraform output -raw inference_health_url
terraform output -raw inference_predict_url
curl $(terraform output -raw inference_health_url)

Check logs:
aws logs tail /ecs/aegis-hgx-dev-inference --region us-west-2 --since 30m

Cheat Sheet

Production skeleton:
A minimal but working foundation for data, training, tracking, serving, containers, cloud storage, and deployment.

Evidence document:
A report proving what exists, how it works, how it was tested, and what remains limited.

Baseline model:
A simple reference model used to validate the pipeline before advanced modeling.

MLflow:
Tracks training runs, parameters, metrics, and artifacts.

FastAPI:
Serves model predictions through validated HTTP endpoints.

S3 data lake:
Stores raw data, future validated data, model-ready data, metadata, configs, artifacts, and experiment artifacts.

ECS Fargate:
Runs the inference container in AWS without managing servers.

ALB:
Provides a public HTTP entry point to the ECS service.

Known limitation:
A truthful gap between the current platform foundation and future research-grade detection quality.


I wrote a production skeleton report for AEGIS-HGX that documents the current end-to-end ML platform foundation: synthetic data generation, baseline training, MLflow tracking, FastAPI inference serving, Docker containerization, S3 data lake layout, ECS Fargate deployment, verification evidence, metrics, limitations, and next engineering steps.
