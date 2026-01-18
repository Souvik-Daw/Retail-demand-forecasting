import requests
import json
import requests
import json
import pandas as pd
import numpy as np

# Define the API URL
URL = "http://localhost:8000/api/v1/predict"

# ---------------------------------------------------------
# Feature Calculation Function (Simulating Client Logic)
# ---------------------------------------------------------
def prepare_client_features(raw_price_history):
    """
    Takes raw price history and returns a List of Lists (Values Only)
    """
    # 1. Convert to DataFrame
    df = pd.DataFrame(raw_price_history, columns=['sales'])
    
    # 2. Calculate Lags
    df['lag_7'] = df['sales'].shift(7)
    df['lag_28'] = df['sales'].shift(28)
    
    # 3. Calculate Rolling Stats
    df['rolling_mean_7'] = df['sales'].rolling(window=7).mean()
    df['rolling_std_7'] = df['sales'].rolling(window=7).std()
    
    # 4. Add Calendar Features
    df['day_of_week'] = 1
    df['month'] = 5
    df['is_weekend'] = 0
    
    # 5. Drop NaN and take last row
    feature_row = df.dropna().tail(1)
    
    # 6. Select ONLY the feature columns in the EXACT ORDER required
    feature_cols = [
        'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 
        'day_of_week', 'month', 'is_weekend'
    ]
    
    return feature_row[feature_cols].values.tolist()

# ---------------------------------------------------------
# TEST 1: XGBoost (Sending Pre-Calculated Features)
# ---------------------------------------------------------
print("\n--- Testing XGBoost (With Pre-Calculated Features) ---")

# 1. dummy history (35 days) for feature calculation
dummy_history = [100 + i + (i%5) for i in range(40)]

# 2. Calculate features LOCALLY (Client side)
payload_data = prepare_client_features(dummy_history)

print(f"Generated Payload: {json.dumps(payload_data, indent=2)}")

xgb_payload = {
    "model_type": "xgboost",
    "data": payload_data  
}

try:
    response = requests.post(URL, json=xgb_payload)
    
    if response.status_code == 200:
        print("\nSuccess! Prediction received:")
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nError {response.status_code}: {response.text}")

except Exception as e:
    print(f"Connection Failed: {e}")

# ---------------------------------------------------------
# TEST 2: LSTM (Time Series)
# ---------------------------------------------------------
print("\n--- Testing LSTM ---")

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