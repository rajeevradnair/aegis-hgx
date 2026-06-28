from fastapi.testclient import TestClient

from aegis_hgx.models.serving.app import app

def valid_event_payload() -> dict[str, object]:
    return {
        "user_id": "user_014",
        "host_id": "host_003",
        "process_name": "encoded_powershell",
        "event_type": "privilege_change",
        "source_ip": "10.0.0.12",
        "destination_ip": "203.0.113.18",
        "bytes_in": 500,
        "bytes_out": 95000,
        "event_hour": 2,
        "is_business_hour": False,
    }

def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/baseline_logistic/health")
                               
    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["model_path"].endswith(
        "logistic_baseline.joblib"
    )

def test_predict_accepts_valid_event() -> None:
    payload = valid_event_payload()

    with TestClient(app) as client:
        response = client.post("/api/v1/baseline_logistic/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]
    assert body["classification"] in ["normal", "suspicious"]
    assert 0.0 <= body["suspicious_probability"] <= 1.0

    expected_classification = (
        "suspicious"
        if body["prediction"] == 1
        else "normal"
    )

    assert body["classification"] == expected_classification

def test_predict_response_schema_is_stable() -> None:
    payload = valid_event_payload()

    with TestClient(app) as client:
        response = client.post("/api/v1/baseline_logistic/predict", json=payload)

    assert response.status_code == 200

    body = response.json()

    print(body, set(body.keys()))

    assert set(body.keys()) == {
        "classification",
        "prediction",
        "suspicious_probability",
    }

def test_predict_rejects_wrong_field_type() -> None:
    payload = valid_event_payload()
    payload["bytes_out"] = "very high"

    with TestClient(app) as client:
        response = client.post("/api/v1/baseline_logistic/predict", json=payload)

    assert response.status_code == 422

    error_fields = {
        error["loc"][-1]
        for error in response.json()["detail"]
    }

    assert "bytes_out" in error_fields

def test_predict_rejects_missing_required_field() -> None:
    payload = valid_event_payload()
    payload.pop("bytes_out")

    k= {"":""}

    with TestClient(app) as client:
        response = client.post("/api/v1/baseline_logistic/predict", json=payload)

    assert response.status_code == 422

    error_fields = {
        error["loc"][-1]
        for error in response.json()["detail"]
    }

    assert "bytes_out" in error_fields

def test_predict_rejects_invalid_event() -> None:
    payload = {
        "user_id": "user_014",
        "host_id": "host_003",
        "process_name": "chrome",
        "event_type": "login_success",
        "source_ip": "10.0.0.12",
        "destination_ip": "10.0.0.15",
        "bytes_in": 500,
        "bytes_out": -1,
        "event_hour": 29,
        "is_business_hour": True,
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/baseline_logistic/predict", json=payload)

    assert response.status_code == 422


    error_fields = {
        error["loc"][-1]
        for error in response.json()["detail"]
    }

    assert "bytes_out" in error_fields
    assert "event_hour" in error_fields