import pandas as pd
import numpy as np
import joblib
import os
import sys
import json
from src.utils.logger import logger
from src.utils.exception import CustomException

class Preprocessor:
    def __init__(self):
        # Paths based on your structure: root/artifacts/filename.pkl
        # We use 'artifacts' assuming the app runs from the project root
        self.xgb_scaler_path = os.path.join("artifacts", "preprocessor_xgboost.pkl")
        self.nn_scaler_path = os.path.join("artifacts", "preprocessor_nn.pkl")
        
        self.nn_scaler = None
        self.xgb_scaler = None

        # Load NN Preprocessor (Critical for LSTM)
        if os.path.exists(self.nn_scaler_path):
            self.nn_scaler = joblib.load(self.nn_scaler_path)
            logger.info(f"Loaded NN Preprocessor from {self.nn_scaler_path}")
        else:
            logger.warning(f"NN Preprocessor not found at {self.nn_scaler_path}")

        # Load XGB Preprocessor (Optional, usually XGB handles raw data well, but good to have)
        if os.path.exists(self.xgb_scaler_path):
            self.xgb_scaler = joblib.load(self.xgb_scaler_path)
            logger.info(f"Loaded XGBoost Preprocessor from {self.xgb_scaler_path}")

    def preprocess_xgboost(self, input_data: list) -> str:
        """
        Converts list of dicts/lists to CSV string (text/csv).
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame(input_data)
            
            # If you applied scaling during training for XGBoost, apply it here using self.xgb_scaler
            if self.xgb_scaler:
                df = pd.DataFrame(self.xgb_scaler.transform(df), columns=df.columns)

            # Ensure no headers are sent, just comma-separated values
            csv_data = df.to_csv(header=False, index=False)
            
            return csv_data.strip()
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_lstm(self, input_data: list) -> str:
        """
        Scales and Reshapes data to 3D JSON (application/json).
        """
        try:
            # 1. Convert to Numpy
            data_array = np.array(input_data)
            
            # 2. Scale using the specific NN preprocessor
            if self.nn_scaler:
                data_array = self.nn_scaler.transform(data_array)
            
            # 3. Reshape to [Samples, Timesteps, Features]
            # [Samples, Features] -> [Samples, 1, Features]
            if data_array.ndim == 2:
                data_array = data_array.reshape((data_array.shape[0], 1, data_array.shape[1]))
            
            # 4. Convert to JSON format for TF Serving
            payload = json.dumps({"instances": data_array.tolist()})
            
            return payload
        except Exception as e:
            raise CustomException(e, sys)