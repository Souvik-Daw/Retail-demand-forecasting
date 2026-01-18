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

        self.xgb_scaler_path = os.path.join("artifacts", "preprocessor_xgboost.pkl")
        self.nn_scaler_path = os.path.join("artifacts", "preprocessor_nn.pkl")
        
        self.nn_scaler = None
        self.xgb_scaler = None

        if os.path.exists(self.nn_scaler_path):
            self.nn_scaler = joblib.load(self.nn_scaler_path)
            logger.info(f"Loaded NN Preprocessor from {self.nn_scaler_path}")
        else:
            logger.warning(f"NN Preprocessor not found at {self.nn_scaler_path}")

        if os.path.exists(self.xgb_scaler_path):
            self.xgb_scaler = joblib.load(self.xgb_scaler_path)
            logger.info(f"Loaded XGBoost Preprocessor from {self.xgb_scaler_path}")
        
    def preprocess_xgboost(self, input_data: list) -> str:
        try:
  
            if input_data and isinstance(input_data[0], list):
                df = pd.DataFrame(input_data)
                required_cols = [
                    'lag_7', 'lag_28', 'rolling_mean_7', 'rolling_std_7', 
                    'day_of_week', 'month', 'is_weekend'
                ]
                if len(df.columns) == len(required_cols):
                    df.columns = required_cols
                
                if self.xgb_scaler:
                    try:
                        scaled_values = self.xgb_scaler.transform(df)
                        df = pd.DataFrame(scaled_values, columns=df.columns)
                    except Exception as e:
                        print(f"[WARNING] Scaling failed (Dimension mismatch?): {e}")

                return df.to_csv(header=False, index=False).strip()
            
            df = pd.DataFrame(input_data)
            if df.empty:
                raise ValueError("Preprocessing resulted in empty data.")
                
            return df.to_csv(header=False, index=False).strip()

        except Exception as e:
            print(f"Preprocessing Error: {e}")
            raise CustomException(e, sys)

    def preprocess_lstm(self, input_data: list) -> str:

        try:
            data_array = np.array(input_data)
            
            if self.nn_scaler:
                data_array = self.nn_scaler.transform(data_array)
            
            if data_array.ndim == 2:
                data_array = data_array.reshape((data_array.shape[0], 1, data_array.shape[1]))
            
            payload = json.dumps({"instances": data_array.tolist()})
            
            return payload
        except Exception as e:
            raise CustomException(e, sys)