
import os
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import tensorflow as tf
import yfinance as yf
from datetime import timedelta
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
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            
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
            
            # 2. Add Lag Features (replicating DataTransformation defaults)
            lag_cols = ["Close"]
            lags = [1, 2, 3, 5]
            df_processed = self.data_transformation.create_lag_features(df_processed, lag_cols, lags)

            # Reconstruct the feature list used in training
            final_features = self.get_final_feature_names(df_processed, lag_cols, lags)
            
            logging.info(f"Using features for prediction: {final_features}")
            
            logging.info(f"Using features for prediction: {final_features}")

            if "Close" not in final_features:
                 raise ValueError("Close price missing from features")

            # Check if we have enough data for the sequence
            # We need enough history for lags + window
            # But create_lag_features handles NaNs (backfill/ffill).
            # We just need enough rows for the window.
            
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

    def get_final_feature_names(self, df_processed: pd.DataFrame, lag_cols=["Close"], lags=[1, 2, 3, 5]):
        """
        Helper to reconstruct the exact feature order used in training.
        """
        # Base features
        feature_cols = ["Close", "Open", "High", "Low", "Volume"]
        
        # Indicators
        indicators = ["SMA_7", "SMA_21", "EMA_12", "EMA_26", "MACD", "RSI_14", "daily_return", "log_return"]
        
        final_features = feature_cols + indicators
        
        # Lags
        for c in lag_cols:
            for lag in lags:
                final_features.append(f"{c}_lag{lag}")
        
        # Filter to keep only what's in df_processed
        final_features = [f for f in final_features if f in df_processed.columns]
        
        # Ensure Close is first
        if "Close" in final_features and final_features[0] != "Close":
            final_features.remove("Close")
            final_features.insert(0, "Close")
            
        return final_features

    def fetch_live_data(self, period="2y"):
        try:
            import time
            ticker = "^NSEI" 
            
            # Retry logic for cloud environments
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logging.info(f"Fetching live data from yfinance (attempt {attempt + 1}/{max_retries})...")
                    
                    # Create Ticker object with headers (helps with cloud restrictions)
                    stock = yf.Ticker(ticker)
                    df = stock.history(period=period)
                    
                    if not df.empty:
                        break
                        
                    logging.warning(f"Attempt {attempt + 1} returned empty data, retrying...")
                    time.sleep(2)  # Wait before retry
                    
                except Exception as fetch_error:
                    logging.warning(f"Attempt {attempt + 1} failed: {fetch_error}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2)
            
            if df.empty:
                raise ValueError("Failed to fetch data from yfinance after all retries. Please try again later or check your internet connection.")

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            
            # yfinance history returns 'Date' in index, now in column after reset
            if "Date" not in df.columns:
                # Check if index was datetime
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    df['Date'] = df.index
                    df = df.reset_index(drop=True)
                    
            required_raw = ["Date", "Open", "High", "Low", "Close", "Volume"]
            
            # Basic validation
            missing = [c for c in required_raw if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns from live data: {missing}. Available columns: {list(df.columns)}")

            logging.info(f"Successfully fetched {len(df)} rows of live data")
            return df[required_raw].copy()
            
        except Exception as e:
            logging.error(f"Failed to fetch live data: {str(e)}")
            raise CustomException(e, sys)

    def refit_and_update(self, epochs: int = 5):
        """
        Fetches latest data, updates the model (fine-tuning), and overwrites the artifact.
        """
        try:
            logging.info("Starting online learning (refit)...")
            
            # 1. Fetch Data
            df = self.fetch_live_data(period="2y") # 2 years gives good context for LSTM
            
            # 2. Transform Features
            df_processed = self.data_transformation.add_technical_indicators(df)
            lag_cols = ["Close"]
            lags = [1, 2, 3, 5]
            df_processed = self.data_transformation.create_lag_features(df_processed, lag_cols, lags)
            
            # 3. Prepare X, y
            final_features = self.get_final_feature_names(df_processed, lag_cols, lags)
            logging.info(f"Refit features: {final_features}")
            
            # Filter rows with NaNs created by lags/indicators
            df_processed = df_processed.dropna()
            
            if len(df_processed) < 60:
                raise ValueError("Not enough data after processing to refit model.")

            # Scale
            input_arr = df_processed[final_features].astype(float).values
            # We use the EXISTING scaler. We do NOT fit_transform, because we want to maintain 
            # the original scaling logic the model learnt. 
            # Note: If market shifts drastically outside original bounds, this might clip or skew.
            # ideally we'd partial_fit, but MinMaxScaler doesn't support partial_fit easily 
            # without keeping min/max state manually. We'll assume stability or use transform.
            input_scaled = self.scaler.transform(input_arr)
            
            # Create Sequences
            target_idx = final_features.index("Close")
            X_train, y_train = self.data_transformation.create_sequences(
                input_scaled, in_len=60, out_len=1, target_index=target_idx
            )
            
            # 4. Fine-tune Model
            # Model loaded with compile=False, so we must compile.
            self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Use small learning rate if possible? Keras 'fit' uses existing optimizer state.
            # We'll just run a few epochs.
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)
            logging.info("Model fine-tuning complete.")
            
            # 5. Save Updated Model
            self.model.save(self.model_path)
            logging.info(f"Updated model saved to {self.model_path}")
            
        except Exception as e:
            logging.exception("Error in refit_and_update")
            raise CustomException(e, sys)

    def predict_next_n_days(self, steps: int = 7, retrain: bool = False):
        """
        Fetches live data and predicts the next n days recursively.
        Optionally retrains the model on latest data before predicting.
        Falls back to using existing model if live data fetch fails (for cloud deployment).
        """
        try:
            if retrain:
                logging.info("Retraining requested, attempting to fetch live data...")
                try:
                    self.refit_and_update(epochs=5)
                except Exception as retrain_error:
                    logging.warning(f"Retraining failed (likely due to yfinance restrictions in cloud): {retrain_error}")
                    logging.info("Continuing with existing model without retraining...")
            
            logging.info(f"Starting live prediction for next {steps} days")
            
            # 1. Fetch Live Data (with fallback)
            try:
                df = self.fetch_live_data(period="1y")
                logging.info("Successfully fetched live data")
            except Exception as fetch_error:
                logging.warning(f"Failed to fetch live data: {fetch_error}")
                logging.info("Using demo/cached data as fallback...")
                
                # Fallback: Try to load test data from artifacts
                fallback_paths = [
                    "artifacts/test.csv",
                    "artifacts/train.csv",
                    "artifacts/dummy_train.csv"
                ]
                
                df = None
                for path in fallback_paths:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        logging.info(f"Loaded fallback data from {path}")
                        # Ensure Date column exists
                        if 'date' in df.columns:
                            df['Date'] = df['date']
                        if 'Date' not in df.columns:
                            # Create synthetic dates
                            from datetime import datetime
                            end_date = datetime.now()
                            df['Date'] = pd.date_range(end=end_date, periods=len(df), freq='D')
                        break
                
                if df is None:
                    raise ValueError("Could not fetch live data and no fallback data available. Please ensure you have internet connection or add demo data to artifacts/")
            
            predictions = []
            
            # Current state DataFrame to append predictions to
            current_df = df.copy()
            
            for i in range(steps):
                # Predict next close
                pred_price = self.predict(current_df)
                
                # Determine next date
                last_date = pd.to_datetime(current_df['Date'].iloc[-1])
                next_date = last_date + timedelta(days=1)
                
                if next_date.weekday() == 5: # Saturday
                    next_date += timedelta(days=2) # Skip to Monday
                elif next_date.weekday() == 6: # Sunday
                    next_date += timedelta(days=1) # Skip to Monday

                # Create next row
                # Heuristic: Open/High/Low = Predicted Close. Volume = Last Volume.
                next_row = {
                    "Date": next_date,
                    "Open": pred_price,
                    "High": pred_price,
                    "Low": pred_price,
                    "Close": pred_price,
                    "Volume": current_df['Volume'].iloc[-1]
                }
                
                # Add to history for next iteration
                new_row_df = pd.DataFrame([next_row])
                current_df = pd.concat([current_df, new_row_df], ignore_index=True)
                
                predictions.append({
                    "Date": next_date.strftime("%Y-%m-%d"),
                    "Price": float(pred_price)
                })
                
            return predictions

        except Exception as e:
            raise CustomException(e, sys)
