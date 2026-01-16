import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates Lag and Rolling Window features.
    Sorts data -> GroupBy Store/Family -> Shift.
    """
    try:
        logger.info("Generating Lag & Rolling features...")
        df = df.copy()
        
        # CRITICAL: Sort is required before shifting
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        # Forecast Horizon (e.g., 16 days ahead)
        HORIZON = 16 
        target_col = 'sales'
        
        # Group by Entity (Store + Family)
        grouped = df.groupby(['store_nbr', 'family'])[target_col]

        # 1. Simple Lags
        # "What were sales 17 days ago?" (Horizon 16 + Lag 1)
        for lag in [1, 7, 14]:
            df[f'lag_{lag}'] = grouped.shift(HORIZON + lag)
            
        # 2. Rolling Means
        # "What was the average sales of the last 7 days (as of 16 days ago)?"
        df['roll_7_mean'] = grouped.transform(lambda x: x.shift(HORIZON).rolling(7).mean())
        df['roll_14_mean'] = grouped.transform(lambda x: x.shift(HORIZON).rolling(14).mean())

        # 3. Drop Rows with NaNs created by Lags
        # (The first ~30 days will be empty due to shifts)
        initial_rows = df.shape[0]
        df = df.dropna()
        dropped = initial_rows - df.shape[0]
        logger.info(f"Feature Engineering complete. Dropped {dropped} rows due to lags.")

        return df
        
    except Exception as e:
        raise CustomException(e, sys)