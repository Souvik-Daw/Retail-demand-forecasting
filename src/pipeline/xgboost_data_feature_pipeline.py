import sys
import os

# Ensure the root directory is in the path to handle imports correctly
sys.path.append(os.getcwd())

from src.data_processing.load import load_data
from src.data_processing.validate import validate_data
from src.data_processing.clean import clean_data
from src.features.calendar_features import add_date_features
from src.features.lag_features import add_lag_features
from src.data_processing.save_split_data import process_and_save_xgboost
from src.utils.logger import logger
from src.utils.exception import CustomException

def main():
    """
    Orchestrates the End-to-End Data Pipeline for XGBoost:
    1. Load -> 2. Validate -> 3. Clean -> 4. Feature Engineering -> 5. Transform & Split -> 6. Save
    """
    try:
        logger.info(">>>>>>>> STARTING XGBOOST PIPELINE <<<<<<<<")

        # --- STEP 1: LOAD ---
        # Path should be relative to where you run the script from (usually root)
        raw_data_path = os.path.join("data", "raw", "sales_data.csv")
        logger.info(f"Step 1: Loading data from {raw_data_path}...")
        df = load_data(raw_data_path)

        # --- STEP 2: VALIDATE ---
        # Fail fast if schemas don't match
        logger.info("Step 2: Validating schema...")
        validate_data(df)

        # --- STEP 3: CLEAN ---
        # Handle implicit missingness (grid), negatives, and sorting
        logger.info("Step 3: Cleaning data (Grid expansion & sorting)...")
        df = clean_data(df)

        # --- STEP 4: CALENDAR FEATURES ---
        # Extract Year, Month, Day, DayOfWeek
        logger.info("Step 4: generating Calendar Features...")
        df = add_date_features(df)

        # --- STEP 5: LAG FEATURES ---
        # Generate Shifts and Rolling means (Business Logic)
        logger.info("Step 5: Generating Lag & Rolling Features...")
        df = add_lag_features(df)

        # --- STEP 6: TRANSFORM, SPLIT & SAVE ---
        # Fits the XGBoost Preprocessor, Splits by Time, and saves artifacts
        logger.info("Step 6: Splitting, Transforming, and Saving...")
        process_and_save_xgboost(df)

        logger.info(">>>>>>>> PIPELINE COMPLETED SUCCESSFULLY <<<<<<<<")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()