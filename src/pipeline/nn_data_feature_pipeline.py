import sys
import os

sys.path.append(os.getcwd())

from src.data_processing.load import load_data
from src.data_processing.validate import validate_data
from src.data_processing.clean import clean_data
from src.features.calendar_features import add_date_features
from src.data_processing.save_split_data import process_and_save_nn
from src.utils.logger import logger
from src.utils.exception import CustomException

def main():
    try:
        logger.info(">>>>>>>> STARTING NEURAL NETWORK PIPELINE <<<<<<<<")

        # --- STEP 1: LOAD ---
        raw_data_path = os.path.join("data", "raw", "sales_data.csv")
        logger.info(f"Step 1: Loading data from {raw_data_path}...")
        df = load_data(raw_data_path)

        # --- STEP 2: VALIDATE ---
        logger.info("Step 2: Validating schema...")
        validate_data(df)

        # --- STEP 3: CLEAN ---
        logger.info("Step 3: Cleaning data...")
        df = clean_data(df)

        # --- STEP 4: CALENDAR FEATURES ---
        logger.info("Step 4: Generating Calendar Features...")
        df = add_date_features(df)

        # --- STEP 5: LAG FEATURES ---
        # SKIPPED 
        
        # --- STEP 6: TRANSFORM, SPLIT & SAVE ---
        logger.info("Step 6: Splitting, Transforming (Scaling), and Saving...")
        process_and_save_nn(df)

        logger.info(">>>>>>>> PIPELINE COMPLETED SUCCESSFULLY <<<<<<<<")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()