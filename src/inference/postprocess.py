import json
import sys
import numpy as np
from src.utils.exception import CustomException

class Postprocessor:
    def postprocess_xgboost(self, raw_response) -> list:
        try:
            
            if isinstance(raw_response, list):
                return np.array(raw_response).flatten().tolist()

            if isinstance(raw_response, bytes):
                response_str = raw_response.decode("utf-8")
            else:
                response_str = str(raw_response)

            clean_str = response_str.replace('[', '').replace(']', '').replace('\n', ',')

            predictions = [float(x) for x in clean_str.split(',') if x.strip()]
            
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

    def postprocess_lstm(self, raw_response: bytes) -> list:
        try:
            response_json = json.loads(raw_response.decode("utf-8"))
            
            if "predictions" in response_json:
                predictions = np.array(response_json["predictions"]).flatten().tolist()
                return predictions
            else:
                return []
        except Exception as e:
            raise CustomException(e, sys)