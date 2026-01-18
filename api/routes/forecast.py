from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import os
import sys
from src.inference.predictor import ModelPredictor
from src.inference.preprocess import Preprocessor
from src.inference.postprocess import Postprocessor
from src.utils.logger import logger

router = APIRouter()

predictor = ModelPredictor(region_name="us-east-1")
preprocessor = Preprocessor()
postprocessor = Postprocessor()

class ForecastRequest(BaseModel):
    model_type: str 
    data: List[List[float]] 

XGB_ENDPOINT = os.getenv("XGB_ENDPOINT_NAME", "retail-xgb-endpoint-2023-...") 
LSTM_ENDPOINT = os.getenv("LSTM_ENDPOINT_NAME", "retail-lstm-endpoint-2023-...")

@router.post("/predict")
async def get_forecast(request: ForecastRequest):
    try:
        model_type = request.model_type.lower()
        logger.info(f"API Request received for {model_type}")
        
        if model_type == "xgboost":
            # 1. Preprocess 
            payload = preprocessor.preprocess_xgboost(request.data)
            
            # 2. Predict
            raw_response = predictor.predict(XGB_ENDPOINT, payload, "text/csv")
            
            # 3. Postprocess
            forecast = postprocessor.postprocess_xgboost(raw_response)

        elif model_type == "lstm":
            # 1. Preprocess 
            payload = preprocessor.preprocess_lstm(request.data)
            
            # 2. Predict
            raw_response = predictor.predict(LSTM_ENDPOINT, payload, "application/json")
            
            # 3. Postprocess
            forecast = postprocessor.postprocess_lstm(raw_response)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid model_type. Use 'xgboost' or 'lstm'")

        return {
            "status": "success",
            "model": model_type,
            "forecast": forecast
        }

    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))