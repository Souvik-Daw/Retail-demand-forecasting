import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == "__main__":

    print(f"[Info] TensorFlow Version: {tf.__version__}")

    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)

    # Data Directories (Defaults for Local vs SageMaker)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "artifacts"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "data/post/nn"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST", "data/post/nn"))

    # Filenames 
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--train-target-file", type=str, default="train_target.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--test-target-file", type=str, default="test_target.csv")

    args, _ = parser.parse_known_args()

    # 2. Data Loading & Preprocessing
    print("[INFO] Reading data...")
    
    # Load Training Data
    X_train = pd.read_csv(os.path.join(args.train, args.train_file))
    y_train = pd.read_csv(os.path.join(args.train, args.train_target_file))
    
    # Load Testing Data
    X_test = pd.read_csv(os.path.join(args.test, args.test_file))
    y_test = pd.read_csv(os.path.join(args.test, args.test_target_file))

    # Convert to Numpy Arrays
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values
    y_test = y_test.values

    # --- RESHAPE INPUT ---
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    input_dim = X_train.shape[2] # Number of features
    print(f"[INFO] Input Shape: {X_train.shape} (Samples, Timesteps, Features)")

    # 3. Build Custom LSTM Model
    print("[INFO] Building Model Architecture")
    model = tf.keras.Sequential([        
        # LSTM Layer 1
        tf.keras.layers.LSTM(64, input_shape=(1, input_dim), return_sequences=True),
        tf.keras.layers.Dropout(0.2), # Prevent Overfitting
        # LSTM Layer 2
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        # Dense Hidden Layer
        tf.keras.layers.Dense(32, activation='relu'),
        # Output Layer
        tf.keras.layers.Dense(1) 
    ])

    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='mean_squared_error', 
        metrics=['mean_absolute_error']
    )

    model.summary()

    # 4. Train the Model
    print("[INFO] Training Model ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2
    )

    # 5. Evaluation
    print("\n[INFO] Evaluating on Test Data")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate Sklearn Metrics for detailed reporting
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R2:   {r2:.4f}")

    # 6. Save Model
    save_path = os.path.join(args.model_dir, '00000001')
    model.save(save_path)
    print(f"Model saved to {save_path}")