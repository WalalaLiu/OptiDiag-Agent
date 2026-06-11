import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from optidiag.api.main import app


def _png_bytes() -> bytes:
    image = np.zeros((64, 64), dtype=np.uint8)
    image[30:34, :] = 255
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_analyze_file_upload():
    client = TestClient(app)
    response = client.post(
        "/analyze",
        files={"file": ("test.png", _png_bytes(), "image/png")},
        data={"experiment_type": "diffraction"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "issues" in data
    assert "metrics" in data
    assert "diagnosis" in data
    assert "suggestions" in data
    assert data["model_available"] is False
    assert data["fallback"] == "rule_based"
