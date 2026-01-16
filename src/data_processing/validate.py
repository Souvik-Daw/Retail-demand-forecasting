import os
import pandera as pa
from pandera.typing import Series
import pandas as pd
from src.utils.logger import logger
from src.utils.exception import CustomException
import sys
from pandera import Column, Check

# --- 1. Define the Schema ---
SalesSchema = pa.DataFrameSchema({
    "date": Column(pa.DateTime, coerce=True),
    "store_nbr": Column(pa.Int, Check.greater_than_or_equal_to(1), coerce=True),
    "family": Column(pa.String, coerce=True),
    "sales": Column(pa.Float, Check.greater_than_or_equal_to(0.0), coerce=True),
    "onpromotion": Column(pa.Int, Check.greater_than_or_equal_to(0), coerce=True),
})

# --- 2. Validation Function ---
def validate_data(df: pd.DataFrame):
    """
    Validates the dataframe against the SalesSchema.
    Returns the dataframe if successful (potentially with coerced types).
    Raises CustomException if validation fails.
    """
    logger.info("Validating data schema...")
    try:
        validated_df = SalesSchema.validate(df, lazy=True)
        
        logger.info("Schema validation passed successfully.")
        return validated_df

    except pa.errors.SchemaErrors as err:
        
        logger.error("Schema validation failed!")
        
        failure_cases = err.failure_cases
        logger.error(f"Validation Errors Summary:\n{failure_cases}")
        
        raise CustomException(err, sys)
        
    except Exception as e:
        raise CustomException(e, sys)
    


