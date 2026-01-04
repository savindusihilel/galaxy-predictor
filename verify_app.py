from fastapi.testclient import TestClient
import os
import sys

# Ensure api is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import app


def run_tests():
    with TestClient(app) as client:
        # Test Static
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Galaxy" in response.text
        print("  Static file serving (Index) - PASSED")

        # Test Prediction
        payload = {
            "u": 20.7468,
            "g": 19.5216,
            "r": 18.8356,
            "i": 18.4295,
            "z": 18.158,
            "redshift": 0.0394
        }
        
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"Status {response.status_code}: {response.text}"
        data = response.json()
        
        # Check structure
        required_keys = ["mass_log_mean", "quenching_prob_mean", "quenching_posterior"]
        for k in required_keys:
            assert k in data, f"Missing key: {k}"
            
        print(f"  Prediction Endpoint - PASSED")
        print(f"   Mass: {data['mass_log_mean']:.2f}")
        print(f"   SFR: {data['sfr_log_mean']:.2f}")
        print(f"   Q_prob: {data['quenching_prob_mean']:.2f}")

if __name__ == "__main__":
    try:
        run_tests()
        print("\nAll verifications passed!")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        exit(1)
