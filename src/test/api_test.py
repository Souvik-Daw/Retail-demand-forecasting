import requests
import json

# Define the API URL
# Note: We added prefix="/api/v1" in app.py
URL = "http://localhost:8000/api/v1/predict"

# ---------------------------------------------------------
# TEST 1: XGBoost (Regression)
# ---------------------------------------------------------
print("\n--- Testing XGBoost ---")

# Dummy data: 2 rows, 5 features each (Replace with your actual feature count)
xgb_payload = {
    "model_type": "xgboost",
    "data": [
        [100.5, 20.0, 3.0, 0.0, 50.1],  # Row 1
        [105.2, 22.0, 1.0, 1.0, 55.2]   # Row 2
    ]
}

try:
    response = requests.post(URL, json=xgb_payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection Failed: {e}")

# ---------------------------------------------------------
# TEST 2: LSTM (Time Series)
# ---------------------------------------------------------
print("\n--- Testing LSTM ---")

# Same data structure; the backend handles the 3D reshaping
lstm_payload = {
    "model_type": "lstm",
    "data": [
        [100.5, 20.0, 3.0, 0.0, 50.1],
        [105.2, 22.0, 1.0, 1.0, 55.2]
    ]
}

try:
    response = requests.post(URL, json=lstm_payload)
    
    if response.status_code == 200:
        print("Success!")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection Failed: {e}")