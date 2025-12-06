import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from dataclasses import dataclass
from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add common technical indicators and returns a new DataFrame.
        - SMA_7, SMA_21
        - EMA_12, EMA_26
        - MACD (EMA12 - EMA26)
        - RSI_14 (Wilder smoothing)
        - daily_return, log_return
        """
        try:
            df = df.copy()
            if "Close" not in df.columns:
                raise ValueError("Column 'Close' required for indicators")

            # Simple Moving Averages
            df["SMA_7"] = df["Close"].rolling(window=7, min_periods=1).mean()
            df["SMA_21"] = df["Close"].rolling(window=21, min_periods=1).mean()

            # Exponential Moving Averages
            df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
            df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

            # MACD
            df["MACD"] = df["EMA_12"] - df["EMA_26"]

            # RSI using Wilder smoothing (alpha = 1/period)
            delta = df["Close"].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            roll_up = up.ewm(alpha=1/14, adjust=False).mean()
            roll_down = down.ewm(alpha=1/14, adjust=False).mean()
            rs = roll_up / (roll_down.replace(0, np.nan))
            df["RSI_14"] = 100 - (100 / (1 + rs))
            df["RSI_14"] = df["RSI_14"].fillna(method="bfill").fillna(50.0)

            # Returns
            df["daily_return"] = df["Close"].pct_change().fillna(0.0)
            df["log_return"] = np.log1p(df["daily_return"].replace([np.inf, -np.inf], 0.0))

            # Fill any remaining NaNs reasonably
            df = df.fillna(method="ffill").fillna(method="bfill")

            logging.info(f"Added technical indicators; result shape: {df.shape}")
            return df

        except Exception as e:
            logging.exception("Error in add_technical_indicators")
            raise CustomException(e, sys)


    def create_lag_features(self, df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
        """
        For each column in cols, create lag features with given lags.
        Example: Close_lag1, Close_lag2...
        """
        try:
            df = df.copy()
            for col in cols:
                if col not in df.columns:
                    raise ValueError(f"Column '{col}' not present for lag creation")
                for lag in lags:
                    df[f"{col}_lag{lag}"] = df[col].shift(lag)
            # After creating lags, fill edge NaNs by backward-fill then forward-fill
            df = df.fillna(method="bfill").fillna(method="ffill")
            logging.info(f"Created lag features for {cols} with lags={lags}")
            return df
        except Exception as e:
            logging.exception("Error in create_lag_features")
            raise CustomException(e, sys)


    def create_sequences(
        self,
        scaled_array: np.ndarray,
        in_len: int = 60,
        out_len: int = 1,
        target_index: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert scaled 2D array (n_rows, n_features) into sequence windows:
        - X: (n_samples, in_len, n_features)
        - y: (n_samples, out_len)
        target_index selects which column in scaled_array is target (e.g., Close index)
        """
        try:
            X, y = [], []
            n_rows, n_features = scaled_array.shape
            if n_rows < in_len + out_len:
                raise ValueError("Not enough rows to create one window (increase data or reduce in_len)")
            for i in range(in_len, n_rows - out_len + 1):
                X.append(scaled_array[i - in_len:i, :])
                y.append(scaled_array[i:i + out_len, target_index])
            X_arr = np.array(X)
            y_arr = np.array(y)
            logging.info(f"Created sequences X={X_arr.shape} y={y_arr.shape}")
            return X_arr, y_arr
        except Exception as e:
            logging.exception("Error in create_sequences")
            raise CustomException(e, sys)


    def time_series_split(self, df: pd.DataFrame,
                          split_date: Optional[pd.Timestamp] = None,
                          split_fraction: float = 0.8
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Temporal train-test split.
        Either provide split_date (pandas Timestamp/string) or split_fraction (0-1).
        Returns (train_df, test_df)
        """
        try:
            df = df.copy().sort_values("date")
            if split_date is not None:
                split_date = pd.to_datetime(split_date)
                train_df = df[df["date"] < split_date].copy()
                test_df = df[df["date"] >= split_date].copy()
                logging.info(f"Split by date {split_date} -> train:{train_df.shape} test:{test_df.shape}")
            else:
                if not (0.0 < split_fraction < 1.0):
                    raise ValueError("split_fraction must be between 0 and 1")
                n = int(len(df) * split_fraction)
                train_df = df.iloc[:n].copy()
                test_df = df.iloc[n:].copy()
                logging.info(f"Split by fraction {split_fraction} -> train:{train_df.shape} test:{test_df.shape}")
            return train_df, test_df
        except Exception as e:
            logging.exception("Error in time_series_split")
            raise CustomException(e, sys)


    def initiate_data_transformation(
        self,
        train_path: str,
        test_path: Optional[str] = None,
        feature_cols: List[str] = ["Close", "Open", "High", "Low", "Volume"],
        lag_cols: Optional[List[str]] = ["Close"],
        lags: Optional[List[int]] = [1, 2, 3, 5],
        in_len: int = 60,
        out_len: int = 1,
        split_date: Optional[str] = None,
        split_fraction: float = 0.8
    ) -> Dict[str, object]:
        """
        Main method to initiate data transformation.
        Reads data, applies pipeline, saves scaler, returns processed data.
        """
        try:
            logging.info("Initiating Data Transformation")
            
            # Read Data
            df = pd.read_csv(train_path)
            logging.info(f"Read data from {train_path}, shape: {df.shape}")

            # Ensure date column exists and is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            elif "Date" in df.columns:
                df["date"] = pd.to_datetime(df["Date"])
            else:
                logging.warning("No 'date' column found. Checking if index is datetime.")
            
            # 1) Add indicators
            df2 = self.add_technical_indicators(df)

            # 2) Add lags
            if lag_cols and lags:
                df2 = self.create_lag_features(df2, lag_cols, lags)

            # 3) Select final features
            final_features = feature_cols.copy()
            
            # Append technical indicators
            indicators = ["SMA_7", "SMA_21", "EMA_12", "EMA_26", "MACD", "RSI_14", "daily_return", "log_return"]
            for ind in indicators:
                if ind not in final_features and ind in df2.columns:
                    final_features.append(ind)

            if lag_cols and lags:
                for c in lag_cols:
                    for lag in lags:
                        feat_name = f"{c}_lag{lag}"
                        if feat_name not in final_features:
                            final_features.append(feat_name)

            # Filter to keep only what's in df2
            final_features = [f for f in final_features if f in df2.columns]
            
            if "Close" not in final_features:
                final_features.insert(0, "Close")

            logging.info(f"Final features used: {final_features}")

            # 4) Split raw df temporally BEFORE scaling/windowing to avoid leakage
            train_df, test_df = self.time_series_split(df2, split_date=split_date, split_fraction=split_fraction)

            # 5) Scale using train stats only
            scaler = MinMaxScaler()
            train_arr = train_df[final_features].astype(float).values
            test_arr = test_df[final_features].astype(float).values
            
            scaler.fit(train_arr)
            X_train_scaled = scaler.transform(train_arr)
            X_test_scaled = scaler.transform(test_arr)

            # Save scaler
            scaler_path = self.data_transformation_config.preprocessor_obj_file_path
            Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(scaler, scaler_path)
            logging.info(f"Saved scaler at {scaler_path}")

            # 6) Create sequences
            target_idx = final_features.index("Close")
            X_train, y_train = self.create_sequences(X_train_scaled, in_len=in_len, out_len=out_len, target_index=target_idx)
            X_test, y_test = self.create_sequences(X_test_scaled, in_len=in_len, out_len=out_len, target_index=target_idx)

            result = {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "scaler": scaler,
                "feature_cols_used": final_features,
                "df_transformed": df2
            }
            logging.info(f"Transform pipeline complete. Train samples: {X_train.shape[0]} Test samples: {X_test.shape[0]}")
            
            return result

        except Exception as e:
            logging.exception("Error in initiate_data_transformation")
            raise CustomException(e, sys)