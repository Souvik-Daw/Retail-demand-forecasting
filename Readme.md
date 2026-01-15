A production-grade demand forecasting system that predicts future product sales 
and detects demand anomalies using both classical ML (XGBoost) and deep learning 
models deployed on AWS SageMaker.

retail-demand-forecasting/
├── data/
│	├── raw/        # inital csv
│	├── processed/  #final csvs
├── notebooks/
├── src/
│   ├── data/
│	│	├── load.py        # load raw data
│	│	├── clean.py       # cleaning logic
│	│	├── split.py       # time-aware train/val/test split
│	│	├── validate.py    # schema & sanity checks
│   ├── features/
│	│	├── lag_features.py       # t-1, t-7, t-14
│	│	├── rolling_features.py   # rolling mean/std
│	│	├── calendar_features.py  # day, week, month
│	│	├── holiday_features.py   # optional
│	│	└── build_features.py     # orchestration
│   ├── training/
│   │   ├── train_xgboost.py
│   │   ├── train_nn.py
│	│	├── evaluate.py
│   ├── scripts/
│   │   ├── xgboost.py
│   │   ├── nn.py
│   ├── inference/
│	│	├── predictor.py        # prediction logic
│	│	├── preprocess.py       # input → features
│	│	├── postprocess.py      # predictions → response
│	│	└── handler.py          # SageMaker entry point
│   ├── monitoring/
│	│	├── drift_detection.py
│	│	├── retrain_trigger.py	
│   ├── test/
│	│	├── test_case.py	
├── pipelines/
│		└── sagemaker_pipeline.py
├── api/
│	├── app.py           # FastAPI
│	└── routes/
│		├── forecast.py
│		└── model_info.py
├── infra/
│	├── sagemaker_endpoint.py	
├── UI/
└── README.md

TRAINING:
data → features → training → endpoint

INFERENCE:
UI → API → inference → API

MONITORING:
predictions → drift → retrain → pipeline

Data
id | date       | store_nbr | family       | sales | onpromotion
----------------------------------------------------------------