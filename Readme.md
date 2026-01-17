Retail Demand Forecasting System (End-to-End MLOps)
A production-grade Machine Learning system designed to forecast retail sales using a hybrid approach of XGBoost (Gradient Boosting) and LSTM (Deep Learning). This project demonstrates a full MLOps lifecycle: from feature engineering pipelines to scalable deployment on AWS SageMaker, wrapped in a high-performance FastAPI service.

Project Structure
```
Retail_Forecasting/
├── api/                        # API Interface Layer
│   └── routes/                 # Endpoint logic (connects API to Inference)
├── app.py                      # FastAPI Entry Point (Gateway)
├── artifacts/                  # Serialized objects (Scalers, Encoders)
├── data/                       # Local data storage (Raw, Processed)
├── logs/                       # Application & Error logs
├── notebooks/                  # EDA and SageMaker Prototyping
├── src/                        # Core Application Logic
│   ├── data_processing/        # ETL Pipelines (Clean, Load, Split)
│   ├── features/               # Feature Engineering (Lags, Rolling Windows)
│   ├── inference/              # Inference Engine (SageMaker Connection)
│   ├── monitoring/             # Drift Detection & Performance Monitoring
│   ├── pipeline/               # Training Pipeline Orchestration
│   ├── training/               # Model Training Scripts (entry points for AWS)
│   └── utils/                  # Shared Utilities (Logger, Exception Handling)
├── requirements.txt            # Project Dependencies
└── .env                        # Environment Variables (Endpoint Names)
```

Architecture & Flow
1. Training Pipeline (Offline)
The training process is decoupled from the application logic to ensure reproducibility.
Data Ingestion: Raw sales data is cleaned and validated.
Feature Engineering:
XGBoost: Creation of Lag features, Rolling Means, and Calendar features.
LSTM: Sequential scaling and reshaping into time-step windows.
S3 Upload: Processed datasets (train.csv, test.csv) are uploaded to AWS S3.
SageMaker Training:
src/training/train_xgboost.py runs on an ml.m5.xlarge instance.
src/training/train_lstm.py runs on a TensorFlow container.
Artifact Storage: Trained model artifacts (model.tar.gz) are saved back to S3.

2. Inference Pipeline (Online)
Real-time predictions are served via a REST API.
Client Request: User sends data to POST /api/v1/predict.
Preprocessing (src/inference/preprocess.py):
Loads local artifacts (preprocessor_nn.pkl, preprocessor_xgboost.pkl).
XGBoost: Converts data to CSV format (text/csv).
LSTM: Scales and reshapes data into 3D JSON tensors (application/json).
Model Invocation (src/inference/predictor.py):
The API invokes the specific AWS SageMaker Endpoint via boto3.
Post-processing (src/inference/postprocess.py):
Converts raw AWS bytes back into a clean list of predicted sales figures.

Data url -> https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
Schema
----------------------------------------------------------------
id | date       | store_nbr | family       | sales | onpromotion
----------------------------------------------------------------

XGboost
X = [
  lag_1, lag_7, lag_14,
  roll_7_mean, roll_14_mean,
  day_of_week, month,year,
  onpromotion, store_nbr, family
]
y = sales

Custom NN (LSTM)
X = [
  day_of_week, month,year,
  onpromotion, store_nbr, family
]
y = sales