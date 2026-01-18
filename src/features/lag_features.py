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
        
        df = df.sort_values(['store_nbr', 'family', 'date'])
        
        HORIZON = 16 
        target_col = 'sales'
        
        grouped = df.groupby(['store_nbr', 'family'])[target_col]

        for lag in [1, 7, 14]:
            df[f'lag_{lag}'] = grouped.shift(HORIZON + lag)
            
        df['roll_7_mean'] = grouped.transform(lambda x: x.shift(HORIZON).rolling(7).mean())
        df['roll_14_mean'] = grouped.transform(lambda x: x.shift(HORIZON).rolling(14).mean())

        initial_rows = df.shape[0]
        df = df.dropna()
        dropped = initial_rows - df.shape[0]
        logger.info(f"Feature Engineering complete. Dropped {dropped} rows due to lags.")

        return df
        
    except Exception as e:
        raise CustomException(e, sys)