import pandas as pd
import numpy as np
from src.data_processing.clean import clean_data
import pytest

def test_cleaning_logic():
    
    data = {
        'date': ['2023-01-04', '2023-01-01', '2023-01-01', '2023-01-03'], 
        'store_nbr': [1, 1, 1, 1],
        'family': ['FOOD', ' FOOD ', ' FOOD ', 'FOOD'], 
        'sales': [-50.0, 10.0, 10.0, 30.0],
        'onpromotion': [0, 0, 0, 1]
    }
    
    df_raw = pd.DataFrame(data)
    
    # 2. RUN CLEANING
    try:
        df_clean = clean_data(df_raw)
        print(f"\n[CLEAN DATA]\n{df_clean}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise e

if __name__ == "__main__":
    test_cleaning_logic()