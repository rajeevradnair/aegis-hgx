AEGIS-HGX Containerization Notes

Objective
Package model training and model inference into portable Docker containers.

Files Added
- Dockerfile.training
- Dockerfile.inference
- compose.yaml
- .dockerignore

Core Design
Training and inference are separate containers because they have different lifecycles.

Training:
- Runs once.
- Reads data from data/.
- Trains the logistic regression baseline.
- Writes model artifacts to artifacts/.
- Writes evaluation metrics to reports/.
- Writes MLflow tracking state to mlflow/.
- Exits after completion.

Inference:
- Starts FastAPI with Uvicorn.
- Loads the trained joblib model during startup.
- Serves /health and /predict.
- Keeps running until stopped.

Docker Compose Services
training:
- Builds from Dockerfile.training.
- Mounts data/, artifacts/, reports/, and mlflow/.
- Runs:
  python -m aegis_hgx.models.baselines.train_logistic_baseline

inference:
- Builds from Dockerfile.inference.
- Mounts artifacts/ read-only.
- Mounts configs/ read-only.
- Maps local port 8000 to container port 8000.
- Runs:
  uvicorn aegis_hgx.models.serving.app:app --host 0.0.0.0 --port 8000

Important Runtime Paths
Local project path:
- artifacts/models/logistic_baseline.joblib

Inside container:
- /app/artifacts/models/logistic_baseline.joblib

Because artifacts/ is mounted, the inference container can load the model trained by the training container.

Verification Commands

Build images:
docker compose build

Run training:
docker compose run --rm training

Verify training outputs:
ls artifacts/models/logistic_baseline.joblib
cat reports/baseline_logistic_metrics.json
ls mlflow

Run inference:
docker compose up inference

Test health endpoint:
curl http://127.0.0.1:8000/health

Expected health response shape:
{
  "status": "healthy",
  "model_loaded": true,
  "model_path": "artifacts/models/logistic_baseline.joblib"
}

Test prediction endpoint:
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_014",
    "host_id": "host_003",
    "process_name": "encoded_powershell",
    "event_type": "privilege_change",
    "source_ip": "10.0.0.12",
    "destination_ip": "203.0.113.18",
    "bytes_in": 500,
    "bytes_out": 95000,
    "event_hour": 2,
    "is_business_hour": false
  }'

Expected prediction response shape:
{
  "prediction": 0 or 1,
  "classification": "normal" or "suspicious",
  "suspicious_probability": number between 0.0 and 1.0
}

Common Issues

Issue:
MLflow save_model fails because artifacts/models/logistic_baseline_mlflow already exists.

Fix:
rm -rf artifacts/models/logistic_baseline_mlflow
docker compose run --rm training

Better code fix:
Delete the existing exported MLflow model folder before calling mlflow.sklearn.save_model.

Issue:
New model does not appear under mlflow/mlruns.

Likely causes:
- Training crashed before mlflow.log_model or mlflow.log_artifacts.
- mlflow.sklearn.save_model only wrote a local model folder, not a tracked MLflow artifact.
- Existing mlflow.db points to an old artifact_location path.

Clean reset:
rm -rf mlflow
mkdir -p mlflow
docker compose run --rm training

Then inspect:
sqlite3 mlflow/mlflow.db "select experiment_id, name, artifact_location from experiments;"


Recommended check:
docker compose build
docker compose run --rm training
docker compose up inference
python -m pytest tests/test_inference_api.py -v
make test
make lint

