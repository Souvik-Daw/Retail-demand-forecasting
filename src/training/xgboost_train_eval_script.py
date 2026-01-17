import argparse
import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def model_fn(model_dir):
    """
    Deserializes the model from the model_dir.
    SageMaker calls this function to load the model for inference.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")

if __name__ == "__main__":
    print("[Info] Extracting arguments")
    parser = argparse.ArgumentParser()

    # --- XGBoost Hyperparameters ---
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--random_state", type=int, default=42)

    # --- Data Directories (SageMaker Env Vars or Local Defaults) ---
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "artifacts"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/post/xgboost"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "data/post/xgboost"))
    
    # --- Filenames (Matching your Pipeline Output) ---
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--train-target-file", type=str, default="train_target.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--test-target-file", type=str, default="test_target.csv")

    args, _ = parser.parse_known_args()

    print(f"XGBoost Version: {xgb.__version__}")
    print(f"Joblib Version: {joblib.__version__}")

    print("\n[INFO] Reading data...")
    
    # Load Training Data
    X_train = pd.read_csv(os.path.join(args.train, args.train_file))
    y_train = pd.read_csv(os.path.join(args.train, args.train_target_file))
    
    # Load Testing Data
    X_test = pd.read_csv(os.path.join(args.test, args.test_file))
    y_test = pd.read_csv(os.path.join(args.test, args.test_target_file))

    # Flatten y (targets) to 1D array if they are read as DataFrames
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    print("\n---- DATA SHAPES ----")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    print("\n[INFO] Training XGBoost Regressor...")
    
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        random_state=args.random_state,
        n_jobs=-1,  # Use all cores
        objective='reg:squarederror' # Regression Objective
    )
    
    model.fit(X_train, y_train)
    print("Training Complete.")

    # --- Saving Model ---
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved at {model_path}")

    # --- Evaluation ---
    print("\n---- EVALUATION ON TEST SET ----")
    y_pred = model.predict(X_test)

    # Calculate Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE):    {mae:.4f}")
    print(f"R2 Score:                     {r2:.4f}")
    
    # Optional: Baseline Comparison (Mean Predictor)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, [y_train.mean()] * len(y_test)))
    print(f"Baseline (Mean) RMSE:         {baseline_rmse:.4f}")