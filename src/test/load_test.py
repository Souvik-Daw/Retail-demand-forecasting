import os
from src.data_processing.load import load_data

DATA_PATH = os.path.join("data", "raw", "sales_data.csv") 

if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        df = load_data(DATA_PATH)
        print("\nHead of Data:")
        print(df.head())
    else:
        print(f"Please check path: {DATA_PATH} exists before testing.")