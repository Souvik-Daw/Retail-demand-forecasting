import sys
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from src.utils.logger import logger
from src.utils.exception import CustomException

class DriftDetector:
    def __init__(self, threshold: float = 0.05):
        """
        Initializes the Drift Detector.
        :param threshold: The p-value threshold. If p < threshold, drift is detected.
        """
        self.threshold = threshold
        self.drift_report = {}

    def load_data(self, path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise CustomException(e, sys)

    def detect_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """
        Compares numeric columns in reference (train) vs current (new/test) data.
        Returns a report of columns that have drifted.
        """
        try:
            logger.info("Initiating Drift Detection...")
            drifted_columns = []
            
            # 1. Identify Columns
            numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
            common_cols = [c for c in numeric_cols if c in current_df.columns]
            
            # Exclude 
            exclude_cols = ['sales', 'id', 'year'] # Year often drifts naturally, handled by MinMax
            features_to_check = [c for c in common_cols if c not in exclude_cols]

            logger.info(f"Checking drift for features: {features_to_check}")

            for col in features_to_check:
                
                ref_data = reference_df[col].dropna()
                cur_data = current_df[col].dropna()

                if len(ref_data) == 0 or len(cur_data) == 0:
                    continue

                test_stat, p_value = ks_2samp(ref_data, cur_data)

                # 3. Check Threshold
                drift_detected = p_value < self.threshold
                
                self.drift_report[col] = {
                    "p_value": float(p_value),
                    "drift_detected": drift_detected
                }

                if drift_detected:
                    drifted_columns.append(col)
                    logger.warning(f"DRIFT DETECTED in feature '{col}' (p={p_value:.5f})")
                else:
                    logger.info(f"Feature '{col}' is stable (p={p_value:.5f})")

            status = "FAILED" if drifted_columns else "PASSED"
            logger.info(f"Drift Detection Completed. Status: {status}")
            
            return {
                "status": status,
                "drifted_features": drifted_columns,
                "detailed_report": self.drift_report
            }

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    
    try:
        detector = DriftDetector()
        
        ref_path = os.path.join("data", "post", "xgboost", "train.csv")
        cur_path = os.path.join("data", "post", "xgboost", "test.csv")
        
        if os.path.exists(ref_path) and os.path.exists(cur_path):
            ref = detector.load_data(ref_path)
            cur = detector.load_data(cur_path)
            report = detector.detect_drift(ref, cur)
            print(report)
        else:
            logger.error("Data files not found for manual test.")
    except Exception as e:
        logger.error(f"Manual execution failed: {e}")