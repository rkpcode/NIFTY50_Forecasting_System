
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass

from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_model(self, input_shape, output_length):
        """
        Builds an LSTM model adaptable to input shape and output length.
        """
        try:
            model = Sequential()
            # Input layer handled implicitly or explicitly
            model.add(Input(shape=input_shape))
            
            # LSTM layers
            # Return sequences=True for stacking LSTMs if needed, or if we want to capture temporal features
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(output_length))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_data, test_data):
        try:
            logging.info("Initiating Model Training")
            
            # Unpack data
            # Assuming train_data/test_data are tuples (X, y) or dicts
            # Based on data_transformation.py, they are passed as a dictionary 'result' usually or as arguments
            # Let's handle the dictionary case if passed directly, or tuples
            
            X_train, y_train = train_data
            X_test, y_test = test_data
            
            logging.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_length = y_train.shape[1]
            
            model = self.build_model(input_shape, output_length)
            logging.info("Model architecture built successfully")
            
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
            )
            
            logging.info("Model training completed")
            
            # Evaluate
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            
            logging.info(f"Model Evaluation Metrics - RMSE: {rmse}, MAE: {mae}")
            
            # Save model
            model_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)
            logging.info(f"Model saved at {model_path}")
            
            return {
                "rmse": rmse,
                "mae": mae,
                "model_path": model_path
            }
            
        except Exception as e:
            logging.exception("Error in initiate_model_trainer")
            raise CustomException(e, sys)
