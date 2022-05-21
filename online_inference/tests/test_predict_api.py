import os

from fastapi.testclient import TestClient

from ..heart_cleveland_app.predict_api import (
    app,
    load_model_startup,
    HeartClevelandModel,
)

PATH_TO_MODEL = "data/model.pkl"

client = TestClient(app)
load_model_startup(PATH_TO_MODEL)


def test_read_root_page():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "This is heart cleveland prediction model api"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    request = HeartClevelandModel(
        data=[[69, 1, 0, 160, 234, 1, 2, 131, 0, 0.1, 1, 1, 0],
              [65, 1, 0, 138, 282, 1, 2, 174, 0, 1.4, 1, 1, 0]],
        features=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    )
    response = client.get(
        f"/predict/",
        json={"data": request.data, "features": request.features},
    )

    assert response.status_code == 200
    assert response.json() == [{"condition": 0}, {"condition": 1}]
