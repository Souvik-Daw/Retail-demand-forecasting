import pandas as pd
import os
import sys
from src.utils.logger import logger
from src.utils.exception import CustomException

def load_data(file_path: str) -> pd.DataFrame:
    logger.info(f"Initiating data load from: {file_path}")
    
    try:
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist. Check your path.")

        df = pd.read_csv(file_path)
        
        original_mem = df.memory_usage().sum() / 1024**2
        logger.info(f"Initial Memory Usage: {original_mem:.2f} MB")
        
        return df

    except Exception as e:
        raise CustomException(e, sys)