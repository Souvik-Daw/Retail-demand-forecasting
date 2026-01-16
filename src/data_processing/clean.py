import pandas as pd
import numpy as np
import sys
from src.utils.logger import logger
from src.utils.exception import CustomException

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw dataframe for time-series forecasting.
    
    Steps:
    1. Normalizes string columns (strip/upper).
    2. Removes duplicates.
    3. Handles negative sales (clips to 0).
    4. Fills missing values (0 for sales).
    5. Sorts data by Store -> Family -> Date.
    """
    logger.info("Starting robust data cleaning process...")
    try:
        # --- 1. STRING NORMALIZATION ---
        # Fixes issues like " FOOD" vs "FOOD"
        if 'family' in df.columns:
            if df['family'].dtype == 'object':
                df['family'] = df['family'].str.strip().str.upper()

        # --- 2. REMOVE DUPLICATES ---
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        if df.shape[0] != initial_rows:
            logger.info(f"Removed {initial_rows - df.shape[0]} duplicate rows.")
        
        # --- 4. FILL NULLS ---
        # Sales: NaN -> 0.0
        df['sales'] = df['sales'].fillna(0.0)
        # Promotion: NaN -> 0 (Assume no promo if missing)
        df['onpromotion'] = df['onpromotion'].fillna(0)

        # --- 5. NEGATIVE SALES HANDLING ---
        # Returns (-50) should be 0 demand, not negative.
        min_sales = df['sales'].min()
        if min_sales < 0:
            logger.warning(f"Found negative sales ({min_sales}). Clipping to 0.")
            df['sales'] = df['sales'].clip(lower=0.0)

        # --- 6. TYPE CONVERSION ---
        # Now safe to cast types
        df['store_nbr'] = df['store_nbr'].astype('int32')
        df['onpromotion'] = df['onpromotion'].astype('int32')
        df['sales'] = df['sales'].astype('float32')
        df['family'] = df['family'].astype('category') 
        
        # Drop ID if present
        if 'id' in df.columns:
            df = df.drop(columns=['id'])

        # --- 7. FINAL SORTING (CRITICAL) ---
        # Essential for Lag features later
        df = df.sort_values(by=['store_nbr', 'family', 'date'])

        logger.info(f"Cleaning complete. Final Shape: {df.shape}")
        return df

    except Exception as e:
        raise CustomException(e, sys)