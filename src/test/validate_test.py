import pandas as pd
import pytest
from src.data_processing.validate import validate_data
from src.utils.exception import CustomException

def test_validation_rejects_negative_sales():
    
    data = {
        'date': ['2023-01-01'],
        'store_nbr': [1],
        'family': ['FOOD'],
        'sales': [-100.0], 
        'onpromotion': [0]
    }
    df = pd.DataFrame(data)

    with pytest.raises(CustomException):
        validate_data(df)
    print("Test Passed: Negative sales were caught!")

if __name__ == "__main__":
    test_validation_rejects_negative_sales()