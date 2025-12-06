
import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging
from src.nifty50_forecasting_system.components.data_transformation import DataTransformation

class PredictionPipeline:
    def __init__(self):
        try:
            # Hardcoded paths matching training artifacts
            self.model_path = os.path.join("artifacts", "model.h5")
            self.scaler_path = os.path.join("artifacts", "preprocessor.pkl")
            
            logging.info(f"Loading model from {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            
            logging.info(f"Loading scaler from {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            self.data_transformation = DataTransformation()
            
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features_df: pd.DataFrame, in_len: int = 60):
        """
        Args:
            features_df: DataFrame containing at least 'Close' and other required columns.
                         Ideally enough rows to calculate indicators (e.g., > 26 for MACD).
            in_len: Input sequence length for the model.
        Returns:
            Formatted prediction result.
        """
        try:
            logging.info("Starting prediction request")
            
            # 1. Feature Engineering (Reuse DataTransformation logic)
            # We need to calculate indicators first
            df_processed = self.data_transformation.add_technical_indicators(features_df)
            
            # Note: create_lag_features might fail if we don't have enough history before the target window
            # For inference, if we just want the *next* step prediction based on the *last* window,
            # we need to ensure we have the most recent data populated.
            
            # Reconstruct the feature list used in training
            # This must match exactly what was defined in DataTransformation.initiate_data_transformation
            # For robustness, we should ideally save the feature names list in the artifacts.
            # For now, I will manually reconstruct the standard list used.
            
            required_features = ["Close", "Open", "High", "Low", "Volume",
                                 "SMA_7", "SMA_21", "EMA_12", "EMA_26", "MACD", 
                                 "RSI_14", "daily_return", "log_return"]
            
            # Filtering existing columns
            final_features = [f for f in required_features if f in df_processed.columns]
            
            if "Close" not in final_features:
                 raise ValueError("Close price missing from features")

            # Check if we have enough data for the sequence
            if len(df_processed) < in_len:
                raise ValueError(f"Not enough data points. Need at least {in_len} rows, got {len(df_processed)}")
            
            # Select the last 'in_len' rows for prediction
            input_df = df_processed.iloc[-in_len:][final_features]
            
            # Scale
            input_arr = input_df.astype(float).values
            input_scaled = self.scaler.transform(input_arr)
            
            # Reshape for LSTM (1, in_len, n_features)
            bias = input_scaled.shape[0] - in_len 
            # If bias > 0 (shouldn't be if we sliced above), handle it. 
            # Logic: We sliced exactly but if transformation weirdly added rows? No.
            
            X_input = input_scaled.reshape(1, in_len, len(final_features))
            
            # Predict
            pred_scaled = self.model.predict(X_input)
            
            # Inverse Transform
            # The scaler was fitted on ALL features. pred_scaled is only for target (Close).
            # We need to create a dummy array to inverse transform.
            
            # Assumption: "Close" is at index 0 (as ensured in DataTransformation)
            dummy_arr = np.zeros((1, len(final_features)))
            dummy_arr[0, 0] = pred_scaled[0, 0]
            
            pred_inv = self.scaler.inverse_transform(dummy_arr)
            predicted_price = pred_inv[0, 0]
            
            logging.info(f"Prediction result: {predicted_price}")
            return predicted_price

        except Exception as e:
            raise CustomException(e, sys)
