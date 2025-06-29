import os
import sys
from fastapi.testclient import TestClient

# Add root directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fast_api.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to AI MedScan Inference API 🚑"}

def test_predict_endpoint():
    """Test the /predict POST endpoint with an actual image file"""
    test_image_path = "data/raw/test/person102_bacteria_487.jpeg"
    assert os.path.exists(test_image_path), f"Test image not found: {test_image_path}"

    with open(test_image_path, "rb") as image_file:
        response = client.post(
            "/predict",
            files={"file": ("filename.jpg", image_file, "image/jpeg")}
        )

    assert response.status_code == 200

    json_data = response.json()
    assert "prediction" in json_data
    assert "confidence" in json_data
    assert isinstance(json_data["prediction"], str)
    assert isinstance(json_data["confidence"], float)

# ✅ Create output file for DVC
output_file = "tests/fastapi_test_output.txt"
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w") as f:
    f.write(" All FastAPI tests passed successfully.\n")
