from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import joblib
import pandas as pd
from fastapi import FastAPI, Request

import os
from urllib.parse import urlparse
import boto3

from aegis_hgx.utils.config import load_yaml
from aegis_hgx.models.serving.schemas import HealthResponse, PredictionRequest, PredictionResponse

CONFIG_PATH = Path("configs/baseline_logistic.yaml")

def download_model_from_s3(model_uri: str) -> Path:
    parsed_uri = urlparse(model_uri)

    if parsed_uri.scheme != "s3":
        raise ValueError(f"Expected S3 URI, received: {model_uri}")

    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip("/")

    if not bucket or not key:
        raise ValueError(f"Invalid S3 model URI: {model_uri}")

    #local fs location where we first download the model file from s3
    #works on my mac, local docker container and ecs container
    local_model_path = Path("/tmp/aegis_hgx_model.joblib")

    s3_client = boto3.client("s3")
    s3_client.download_file(
        Bucket=bucket,
        Key=key,
        Filename=str(local_model_path),
    )

    return local_model_path


def resolve_model_path(config_model_path: Path) -> Path:
    model_uri = os.getenv("MODEL_URI")

    if model_uri is None:
        return config_model_path

    if model_uri.startswith("s3://"):
        return download_model_from_s3(model_uri)

    return Path(model_uri)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_yaml(CONFIG_PATH)

    configured_model_path = Path(config["outputs"]["model_path"])
    model_path = resolve_model_path(configured_model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
    )

    app.state.model = joblib.load(model_path)
    app.state.model_path = model_path

    print(f"Model at {model_path} loaded into memory")

    yield

    app.state.model = None

# Define the FastAPI app
app = FastAPI(
    title="Aegis-HGX Inference API",
    version="0.1.0",
    lifespan=lifespan,
)

@app.get("/api/v1/baseline_logistic/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    model = request.app.state.model
    model_path = request.app.state.model_path

    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=str(model_path),
    )

@app.post("/api/v1/baseline_logistic/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    
    # fetch the model from app state
    model = request.app.state.model

    # model accepts dataframe as input
    event = pd.DataFrame([payload.model_dump()])

    # model.predict returns a numpy array with 1 row and 1 column
    prediction = int(model.predict(event)[0])
    classification = (
        "suspicious" if prediction == 1 else "normal"
    )
    # model.predict_proba returns a numpy array with 1 row and 2 columns of probabilities
    suspicious_probability = float(
        model.predict_proba(event)[0, 1]
    )

    return PredictionResponse(
        prediction=prediction,
        classification=classification,
        suspicious_probability=suspicious_probability,
    )