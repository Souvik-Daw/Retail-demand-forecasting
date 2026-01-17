import json
import sys
import numpy as np
from src.utils.exception import CustomException

class Postprocessor:
    def postprocess_xgboost(self, raw_response: bytes) -> list:
        try:
            response_str = raw_response.decode("utf-8")
            # XGBoost returns CSV string: "100.5, 200.1"
            predictions = [float(x) for x in response_str.replace('\n', ',').split(',') if x.strip()]
            return predictions
        except Exception as e:
            raise CustomException(e, sys)

    def postprocess_lstm(self, raw_response: bytes) -> list:
        try:
            # TF Serving returns JSON: {"predictions": [[50.5], [60.1]]}
            response_json = json.loads(raw_response.decode("utf-8"))
            
            if "predictions" in response_json:
                predictions = np.array(response_json["predictions"]).flatten().tolist()
                return predictions
            else:
                return []
        except Exception as e:
            raise CustomException(e, sys)