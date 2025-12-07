import os
import sys

# Ensure import from src is possible if run as script
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging
from src.nifty50_forecasting_system.components.data_ingestion import DataIngestion
from src.nifty50_forecasting_system.components.data_transformation import DataTransformation
from src.nifty50_forecasting_system.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    try:
        logging.info(">>>> Training Pipeline Started <<<<<")

        # Data Ingestion
        logging.info("Step 1 : Data Ingestion")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        print(f"Data Ingestion  Completed. Train path : {train_data_path}, Test path : {test_data_path}")

        # Data Transformation
        logging.info("Step 2 : Data Transformation")
        data_transformation = DataTransformation()
        transformation_result = data_transformation.initiate_data_transformation(train_path=train_data_path, test_path=test_data_path)
        print(f"Data Transformation Completed. X_train shape: {transformation_result['X_train'].shape}")

        # Model Training
        logging.info("Step 3 : Model Training")
        model_trainer = ModelTrainer()
        
        train_data = (transformation_result['X_train'], transformation_result['y_train'])
        test_data = (transformation_result['X_test'], transformation_result['y_test'])
        
        model_metrics = model_trainer.initiate_model_trainer(train_data=train_data, test_data=test_data)
        print(f"Model Training Completed. Metrics: {model_metrics}")

    except Exception as e:
        raise CustomException(e, sys)