import os
import sys
from src.nifty50_forecasting_system.exception import CustomException
from src.nifty50_forecasting_system.logger import logging
import pandas as pd
from dataclasses import dataclass
import yfinance as yf
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV


def read_yfinance_data():
    logging.info("Reading Yfinace database")
    try:
        df = yf.download("^NSEI", start="1999-01-01", end="2025-12-04")
        print (df.head())
        return df
    except Exception as e:
        logging.info("Exception occurred in read_yfinance_data")
        raise CustomException(e,sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)