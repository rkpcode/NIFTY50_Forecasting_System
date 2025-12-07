
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass

from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging

@dataclass
class ModelTrainerConfig:
    # Save both models separately
    multivariate_model_path = os.path.join("artifacts", "model_multivariate.h5")
    seq2seq_model_path = os.path.join("artifacts", "model_seq2seq.h5")
    
    # Backward compatibility
    trained_model_file_path = os.path.join("artifacts", "model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def build_multivariate_model(self, input_shape, output_length):
        """
        Builds a Multivariate LSTM model.
        Uses all features (Close, Open, High, Low, Volume, indicators, lags)
        RMSE: ~1083 (from comparison report)
        """
        try:
            model = Sequential()
            model.add(Input(shape=input_shape))
            
            # LSTM layers
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.2))
            
            # Output layer
            model.add(Dense(output_length))
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            logging.info(f"Built Multivariate LSTM: Input{input_shape} -> Output({output_length})")
            return model
        except Exception as e:
            raise CustomException(e, sys)

    def build_seq2seq_model(self, input_shape, output_length):
        """
        Builds a Seq2Seq LSTM model for multi-step forecasting.
        Best performer from comparison report: RMSE 228 (Step 1)
        
        Architecture:
        - Encoder: 2 LSTM layers (128, 64 units)
        - Decoder: RepeatVector + LSTM (64 units) + TimeDistributed Dense
        """
        try:
            # Encoder
            encoder_inputs = Input(shape=input_shape)
            encoder = LSTM(128, return_sequences=True, name='encoder_lstm_1')(encoder_inputs)
            encoder = Dropout(0.2)(encoder)
            encoder_output = LSTM(64, return_sequences=False, name='encoder_lstm_2')(encoder)
            encoder_output = Dropout(0.2)(encoder_output)
            
            # Decoder
            decoder = RepeatVector(output_length, name='repeat_vector')(encoder_output)
            decoder = LSTM(64, return_sequences=True, name='decoder_lstm')(decoder)
            decoder = Dropout(0.2)(decoder)
            
            # Output: TimeDistributed to predict each timestep
            outputs = TimeDistributed(Dense(1), name='output_dense')(decoder)
            
            model = Model(inputs=encoder_inputs, outputs=outputs)
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            logging.info(f"Built Seq2Seq LSTM: Input{input_shape} -> Output({output_length}, 1)")
            return model
        except Exception as e:
            raise CustomException(e, sys)
    
    # Keep old method for backward compatibility
    def build_model(self, input_shape, output_length):
        """Legacy method - calls build_multivariate_model"""
        return self.build_multivariate_model(input_shape, output_length)

    def initiate_model_trainer(self, train_data, test_data, train_both_models=True):
        """
        Train both LSTM models and save them separately.
        
        Args:
            train_data: tuple (X_train, y_train)
            test_data: tuple (X_test, y_test)
            train_both_models: If True, trains both Multivariate and Seq2Seq models
        
        Returns:
            dict with metrics for both models
        """
        try:
            logging.info("=" * 80)
            logging.info("Initiating Dual Model Training (Multivariate + Seq2Seq LSTM)")
            logging.info("=" * 80)
            
            X_train, y_train = train_data
            X_test, y_test = test_data
            
            logging.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
            logging.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
            
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_length = y_train.shape[1]
            
            results = {}
            
            # ===== TRAIN MULTIVARIATE MODEL =====
            logging.info("\n" + "=" * 80)
            logging.info("Training Multivariate LSTM (All Features)")
            logging.info("=" * 80)
            
            model_multi = self.build_multivariate_model(input_shape, output_length)
            
            early_stopping_multi = EarlyStopping(
                monitor='val_loss', 
                patience=10, 
                restore_best_weights=True,
                verbose=1
            )
            
            checkpoint_multi = ModelCheckpoint(
                self.model_trainer_config.multivariate_model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
            
            history_multi = model_multi.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping_multi, checkpoint_multi],
                verbose=1
            )
            
            # Evaluate Multivariate
            predictions_multi = model_multi.predict(X_test)
            mse_multi = mean_squared_error(y_test, predictions_multi)
            rmse_multi = np.sqrt(mse_multi)
            mae_multi = mean_absolute_error(y_test, predictions_multi)
            
            logging.info(f"Multivariate LSTM - RMSE: {rmse_multi:.2f}, MAE: {mae_multi:.2f}")
            
            results['multivariate'] = {
                'rmse': rmse_multi,
                'mae': mae_multi,
                'model_path': self.model_trainer_config.multivariate_model_path
            }
            
            # Save multivariate model
            os.makedirs(os.path.dirname(self.model_trainer_config.multivariate_model_path), exist_ok=True)
            model_multi.save(self.model_trainer_config.multivariate_model_path)
            logging.info(f"✓ Multivariate model saved to: {self.model_trainer_config.multivariate_model_path}")
            
            if train_both_models:
                # ===== TRAIN SEQ2SEQ MODEL =====
                logging.info("\n" + "=" * 80)
                logging.info("Training Seq2Seq LSTM (Best Performer)")
                logging.info("=" * 80)
                
                model_seq2seq = self.build_seq2seq_model(input_shape, output_length)
                
                # Reshape y for Seq2Seq if needed (add feature dimension)
                if len(y_train.shape) == 2:
                    y_train_seq = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)
                    y_test_seq = y_test.reshape(y_test.shape[0], y_test.shape[1], 1)
                else:
                    y_train_seq = y_train
                    y_test_seq = y_test
                
                early_stopping_seq = EarlyStopping(
                    monitor='val_loss', 
                    patience=12,  # More patience for seq2seq
                    restore_best_weights=True,
                    verbose=1
                )
                
                checkpoint_seq = ModelCheckpoint(
                    self.model_trainer_config.seq2seq_model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
                
                history_seq = model_seq2seq.fit(
                    X_train, y_train_seq,
                    validation_data=(X_test, y_test_seq),
                    epochs=60,  # More epochs for seq2seq
                    batch_size=32,
                    callbacks=[early_stopping_seq, checkpoint_seq],
                    verbose=1
                )
                
                # Evaluate Seq2Seq
                predictions_seq = model_seq2seq.predict(X_test)
                # Reshape back to 2D for metrics
                predictions_seq_2d = predictions_seq.reshape(predictions_seq.shape[0], predictions_seq.shape[1])
                
                mse_seq = mean_squared_error(y_test, predictions_seq_2d)
                rmse_seq = np.sqrt(mse_seq)
                mae_seq = mean_absolute_error(y_test, predictions_seq_2d)
                
                logging.info(f"Seq2Seq LSTM - RMSE: {rmse_seq:.2f}, MAE: {mae_seq:.2f}")
                
                results['seq2seq'] = {
                    'rmse': rmse_seq,
                    'mae': mae_seq,
                    'model_path': self.model_trainer_config.seq2seq_model_path
                }
                
                # Save seq2seq model
                os.makedirs(os.path.dirname(self.model_trainer_config.seq2seq_model_path), exist_ok=True)
                model_seq2seq.save(self.model_trainer_config.seq2seq_model_path)
                logging.info(f"✓ Seq2Seq model saved to: {self.model_trainer_config.seq2seq_model_path}")
            
            # ===== SUMMARY =====
            logging.info("\n" + "=" * 80)
            logging.info("TRAINING COMPLETE - Model Comparison Summary")
            logging.info("=" * 80)
            logging.info(f"Multivariate LSTM: RMSE={rmse_multi:.2f}, MAE={mae_multi:.2f}")
            if train_both_models:
                logging.info(f"Seq2Seq LSTM:      RMSE={rmse_seq:.2f}, MAE={mae_seq:.2f}")
                improvement = ((rmse_multi - rmse_seq) / rmse_multi) * 100
                logging.info(f"Seq2Seq Improvement: {improvement:.1f}% better RMSE")
            logging.info("=" * 80)
            
            return results
            
        except Exception as e:
            logging.exception("Error in initiate_model_trainer")
            raise CustomException(e, sys)
