import argparse
import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------
# 1. MODEL LOADING (Server-Side Logic)
# ---------------------------------------------------------
def model_fn(model_dir):
    """
    Load the model as a native XGBoost Booster.
    This works perfectly with SageMaker's default predict_fn.
    """
    model_file = "xgboost-model"
    booster = xgb.Booster()
    booster.load_model(os.path.join(model_dir, model_file))
    return booster

if __name__ == "__main__":
    print("[Info] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--subsample", type=float, default=0.8)
    
    # Locations
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    
    # Filenames
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--train-target-file", type=str, default="train_target.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--test-target-file", type=str, default="test_target.csv")

    args, _ = parser.parse_known_args()

    # ---------------------------------------------------------
    # 2. DATA LOADING
    # ---------------------------------------------------------
    print(f"[INFO] Loading data from {args.train}...")
    
    # Read CSVs
    X_train = pd.read_csv(os.path.join(args.train, args.train_file))
    y_train = pd.read_csv(os.path.join(args.train, args.train_target_file))
    X_test = pd.read_csv(os.path.join(args.test, args.test_file))
    y_test = pd.read_csv(os.path.join(args.test, args.test_target_file))

    # Reshape targets
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    print(f"Training Shape: {X_train.shape}") # Expecting (rows, 43)

    # ---------------------------------------------------------
    # 3. TRAINING (Scikit-Learn API)
    # ---------------------------------------------------------
    print("[INFO] Training...")
    
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 4. SAVING (THE FIX)
    # ---------------------------------------------------------
    # extract the underlying booster
    booster = model.get_booster()
    
    # Save as 'xgboost-model' 
    save_path = os.path.join(args.model_dir, "xgboost-model")
    booster.save_model(save_path)
    
    print(f"[INFO] Model successfully saved to: {save_path}")

    # ---------------------------------------------------------
    # 5. EVALUATION
    # ---------------------------------------------------------
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")