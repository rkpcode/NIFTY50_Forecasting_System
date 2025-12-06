
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Ensure src path is available
sys.path.append(os.getcwd())

from src.nifty50_forecasting_system.pipelines.prediction_pipeline import PredictionPipeline

st.set_page_config(page_title="NIFTY 50 Forecasting", layout="wide")

st.title("📈 NIFTY 50 Stock Price Forecasting")
st.markdown("This dashboard uses an **LSTM Deep Learning Model** to forecast the next day's Close price based on historical data.")

# Initialize pipeline
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

try:
    pipeline = load_pipeline()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar for input
st.sidebar.header("Input Data")

data_source = st.sidebar.radio("Select Data Source", ("Upload CSV", "Use Dummy Data"))

df = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV (must contain Open, High, Low, Close, Volume)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
elif data_source == "Use Dummy Data":
    st.sidebar.info("Using auto-generated dummy data for demonstration.")
    dummy_path = "artifacts/dummy_train.csv"
    if os.path.exists(dummy_path):
        df = pd.read_csv(dummy_path)
    else:
        st.warning("Dummy data not found. Please upload a file.")

if df is not None:
    st.subheader("Historical Data Preview")
    st.dataframe(df.tail(10))
    
    st.subheader("Price History")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        st.line_chart(df.set_index("Date")["Close"])
    else:
        st.line_chart(df["Close"])

    if st.button("Forecast Next Day Price"):
        with st.spinner("Calculating Technical Indicators & Forecasting..."):
            try:
                # Ensure date column doesn't break logic if passed features don't need it or need it
                # PredictionPipeline expects DataFrame. data_transformation adds Date logic if present.
                
                prediction = pipeline.predict(df)
                
                st.metric(label="Predicted Close Price", value=f"₹ {prediction:.2f}")
                
            except Exception as e:
                st.error(f"Prediction Failed: {e}")
else:
    st.info("Please upload a CSV file or ensure dummy data exists to proceed.")
