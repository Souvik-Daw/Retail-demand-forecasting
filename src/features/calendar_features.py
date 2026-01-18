import pandas as pd
import numpy as np
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts Year, Month, Day, DayOfWeek from 'date'.
    """
    try:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
            
        df['year'] = df['date'].dt.year.astype('int16')
        df['month'] = df['date'].dt.month.astype('int8')
        df['day_of_week'] = df['date'].dt.dayofweek.astype('int8') # 0=Mon, 6=Sun
        
        return df
    except Exception as e:
        raise CustomException(e, sys)