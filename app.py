import sys
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.utils.logger import logger
from api.routes import forecast 


load_dotenv()

app = FastAPI(
    title="Retail Forecasting API",
    description="API for forecasting sales using XGBoost and LSTM SageMaker Endpoints",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints will be available at: http://localhost:8000/api/v1/predict
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecasting"])

@app.get("/", tags=["Health Check"])
async def root():
    logger.info("Health check endpoint triggered")
    return {
        "status": "active",
        "message": "Retail Forecasting API is running successfully."
    }

if __name__ == "__main__":
    logger.info("Starting API Server...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

