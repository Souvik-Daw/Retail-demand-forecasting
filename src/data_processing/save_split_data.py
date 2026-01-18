import os
import sys
import pandas as pd
import numpy as np
import joblib
from src.data_processing.transform import get_xgboost_preprocessor, get_nn_preprocessor
from src.utils.logger import logger
from src.utils.exception import CustomException

def split_data_time_series(df, test_days=16):
    """
    Splits data based on time. The last 'test_days' become the test set.
    """
    df = df.sort_values(by=['date', 'store_nbr', 'family'])
    
    max_date = df['date'].max()
    cutoff_date = max_date - pd.Timedelta(days=test_days)
    
    train_df = df[df['date'] <= cutoff_date].copy()
    test_df = df[df['date'] > cutoff_date].copy()
    
    logger.info(f"Time Split - Cutoff: {cutoff_date}")
    logger.info(f"Train Shape: {train_df.shape}, Test Shape: {test_df.shape}")
    
    return train_df, test_df

def process_and_save_xgboost(df: pd.DataFrame):
    """
    1. Splits data (Train/Test).
    2. Fits XGBoost Preprocessor on Train.
    3. Transforms both.
    4. Saves to data/post/xgboost/ WITH HEADER NAMES.
    """
    logger.info("Starting XGBoost Data Processing...")
    try:
        # 1. Split Data
        train_df, test_df = split_data_time_series(df)

        # 2. Separate X and y
        target_col = 'sales'
        X_train = train_df.drop(columns=[target_col, 'id', 'date'], errors='ignore')
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col, 'id', 'date'], errors='ignore')
        y_test = test_df[target_col]

        # 3. Get & Fit Preprocessor
        preprocessor = get_xgboost_preprocessor()
        logger.info("Fitting XGBoost Preprocessor on Train data...")
        
        # Fit on Train ONLY to avoid leakage
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # --- NEW: Get Feature Names ---
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Extracted {len(feature_names)} feature names for XGBoost.")

        # 4. Save Artifacts
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(preprocessor, "artifacts/preprocessor_xgboost.pkl")

        # Save Data (CSV)
        save_dir = os.path.join("data", "post", "xgboost")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with columns names
        pd.DataFrame(X_train_processed, columns=feature_names).to_csv(os.path.join(save_dir, "train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(save_dir, "train_target.csv"), index=False)
        
        pd.DataFrame(X_test_processed, columns=feature_names).to_csv(os.path.join(save_dir, "test.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(save_dir, "test_target.csv"), index=False)
        
        logger.info(f"XGBoost data saved to {save_dir}")

    except Exception as e:
        raise CustomException(e, sys)

def process_and_save_nn(df: pd.DataFrame):
    """
    1. Splits data (Train/Test).
    2. Fits NN Preprocessor (StandardScaler) on Train.
    3. Transforms both.
    4. Saves to data/post/nn/ WITH HEADER NAMES.
    """
    logger.info("Starting Neural Network Data Processing...")
    try:
        # 1. Split Data
        train_df, test_df = split_data_time_series(df)

        # 2. Separate X and y
        target_col = 'sales'
        X_train = train_df.drop(columns=[target_col, 'id', 'date'], errors='ignore')
        y_train = train_df[target_col]
        
        X_test = test_df.drop(columns=[target_col, 'id', 'date'], errors='ignore')
        y_test = test_df[target_col]

        # 3. Get & Fit Preprocessor
        preprocessor = get_nn_preprocessor()
        logger.info("Fitting NN Preprocessor (Scaling) on Train data...")
        
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # --- NEW: Get Feature Names ---
        feature_names = preprocessor.get_feature_names_out()
        logger.info(f"Extracted {len(feature_names)} feature names for Neural Network.")

        # 4. Save Artifacts
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(preprocessor, "artifacts/preprocessor_nn.pkl")

        # Save Data
        save_dir = os.path.join("data", "post", "nn")
        os.makedirs(save_dir, exist_ok=True)
        
        # Save with columns names
        pd.DataFrame(X_train_processed, columns=feature_names).to_csv(os.path.join(save_dir, "train.csv"), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(save_dir, "train_target.csv"), index=False)
        
        pd.DataFrame(X_test_processed, columns=feature_names).to_csv(os.path.join(save_dir, "test.csv"), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(save_dir, "test_target.csv"), index=False)
        
        logger.info(f"Neural Network data saved to {save_dir}")

    except Exception as e:
        raise CustomException(e, sys)