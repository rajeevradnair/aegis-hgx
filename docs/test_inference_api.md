AEGIS-HGX INFERENCE API
=======================

Purpose
-------

The FastAPI service exposes the trained logistic-regression pipeline through HTTP.
It accepts one cyber event and returns a class prediction and suspicious-event
probability.


Components
----------

src/aegis_hgx/models/serving/schemas.py
  Defines and validates request, health, and prediction response contracts.

src/aegis_hgx/models/serving/app.py
  Loads the trained pipeline once at startup and exposes /health and /predict.

tests/test_inference_api.py
  Tests startup health, valid prediction, and invalid-input rejection.


Request Flow
------------

JSON request
  -> PredictionRequest validation
  -> one-row pandas DataFrame
  -> fitted preprocessing pipeline
  -> logistic-regression prediction
  -> PredictionResponse
  -> JSON response


Run Locally
-----------

Install serving dependencies:

python -m pip install -e ".[ml,serving]"

Ensure the trained model exists:

ls artifacts/models/logistic_baseline.joblib

Start the API:

uvicorn aegis_hgx.models.serving.app:app --reload

API documentation:

http://127.0.0.1:8000/docs

Health endpoint:

curl http://127.0.0.1:8000/api/v1/baseline_logistic/health


Validation
----------

Run API tests:

python -m pytest tests/test_inference_api.py -v

Run the complete project checks:

make test
make lint

Expected API test result:

3 passed


Important Behavior
------------------

- The complete sklearn pipeline is loaded once during application startup.
- The request excludes label because label is what the model predicts.
- Invalid input is rejected with HTTP 422 before model inference.
- predict() returns one class per input row.
- predict_proba() returns one probability per class for every input row.
- The endpoint returns the probability associated with suspicious class 1.


Current Limitations
-------------------

- The service loads a local Joblib model rather than resolving a promoted model
  from the MLflow Model Registry.
- The endpoint processes one event per request.
- Authentication, request logging, latency metrics, and deployment packaging are
  not yet implemented.


Commit
------

git add pyproject.toml src tests
git commit -m "Add FastAPI endpoint for cyber event inference"


Cheat Sheet
-----------

Training:
  Learns and saves model state.

Inference:
  Applies saved model state to new events.

Pydantic:
  Validates API input and output contracts.

FastAPI lifespan:
  Loads the model once at startup and cleans up at shutdown.

TestClient:
  Calls the ASGI application in-process without starting Uvicorn.

/api/v1/baseline_logistic/health:
  Confirms that the service is running and the model is loaded.

/api/v1/baseline_logistic/predict:
  Validates an event, creates a DataFrame, invokes the model, and returns the
  prediction and suspicious probability.

