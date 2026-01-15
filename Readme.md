A production-grade demand forecasting system that predicts future product sales 
and detects demand anomalies using both classical ML (XGBoost) and deep learning 
models deployed on AWS SageMaker.

## Project Structure
```
retail-demand-forecasting/
├── data/
│   ├── raw/
│   ├── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── training/
│   ├── inference/
│   ├── monitoring/
│   └── test/
│   └── utils/
├── pipelines/
├── api/
├── infra/
├── UI/
└── README.md
```


TRAINING:
data → features → training → endpoint
INFERENCE:
UI → API → inference → API
MONITORING:
predictions → drift → retrain → pipeline


Data url -> https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data


Schema
----------------------------------------------------------------
id | date       | store_nbr | family       | sales | onpromotion
----------------------------------------------------------------