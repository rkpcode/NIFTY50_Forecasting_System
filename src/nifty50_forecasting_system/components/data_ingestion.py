import os
import sys
from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.nifty50_forecasting_system.utils import read_yfinance_data

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion process")
        try:
            # Read the dataset from Yfinance
            df = read_yfinance_data()
            logging.info("Dataset read successfully")

            # Reset index to make Date a column
            df.reset_index(inplace=True)

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Adding Year column
            df['Year'] = df['Date'].dt.year

            # Split Data based on Year (Temporal Split)
            train_df = df[df['Year'] < 2025].copy()
            test_df  = df[df['Year'] == 2025].copy()

            # Define columns to keep (Features)
            # We exclude 'Close' from features if we treat it as target, 
            # but usually for saving train/test sets we might want to keep it clearly.
            # The original code separated X and y and then concatenated them.
            # Let's define the columns we expect to exist.
            expected_cols = ['Date', 'Open', 'High', 'Low', 'Volume', 'Close']
            
            # Check for missing columns
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                # Fallback: maybe columns are different?
                logging.warning(f"Missing columns: {missing_cols}. Available: {df.columns}")
                # If Close is missing, we can't proceed.
                if 'Close' in missing_cols:
                     raise CustomException(f"Required column 'Close' is missing. Available: {df.columns}", sys)

            # Select features and target
            # We will save the full dataframe (features + target) into train/test csvs
            # The separation X/y in the original code was:
            # X_train = train_df[feature_cols]
            # y_train = train_df['Close']
            # train_set = pd.concat([X_train, y_train], axis=1)
            # This results in 'Close' being in the file. 
            # If 'Close' was in feature_cols, it would be duplicated.
            
            feature_cols = [c for c in expected_cols if c != 'Close' and c in df.columns]
            
            X_train = train_df[feature_cols].copy()
            y_train = train_df['Close']
            X_test  = test_df[feature_cols].copy()
            y_test  = test_df['Close']
            
            train_set = pd.concat([X_train, y_train], axis=1)
            test_set = pd.concat([X_test, y_test], axis=1)

            logging.info("Data split into training and testing sets")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)
