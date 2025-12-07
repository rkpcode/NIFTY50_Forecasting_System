---
title: NIFTY 50 Forecasting System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 📈 NIFTY 50 Forecasting System

AI-powered forecasting system for predicting NIFTY 50 stock prices for the next 7 days.

## Features

- **Live Data**: Fetches real-time NIFTY 50 data using `yfinance`
- **7-Day Forecast**: Predicts next 7 days of stock prices
```
---
title: NIFTY 50 Forecasting System
emoji: 📈
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
license: mit
---

# 📈 NIFTY 50 Forecasting System

AI-powered forecasting system for predicting NIFTY 50 stock prices for the next 7 days.

## Features

- **Live Data**: Fetches real-time NIFTY 50 data using `yfinance`
- **7-Day Forecast**: Predicts next 7 days of stock prices
- **Online Learning**: Optional retraining on latest market data
- **Clean UI**: Weather card-style display of predictions

## Tech Stack

- **Frontend**: Streamlit
- **ML Framework**: TensorFlow + Keras (LSTM)
- **Model Architecture**: Hybrid Approach (User Selectable)
  - **Seq2Seq LSTM** (Recommended): 
    - Encoder-Decoder architecture
    - RMSE: **228.00** (Top Performer)
    - specialized for 7-day forecasts
  - **Multivariate LSTM**: 
    - Standard stacked LSTM
    - Uses 13+ technical indicators
    - RMSE: 1083.22
- **Data Source**: Yahoo Finance (yfinance)

## Features
- **Hybrid Model Selection**: Switch between Seq2Seq and Multivariate models
- **Real-time Forecasting**: Fetches live data for instant predictions
- **Interactive UI**: "Weather Card" style visualization with trend indicators
- **Online Learning**: Optional retraining on the absolute latest data

## Disclaimer

⚠️ This is an educational AI model. Do not use for actual trading decisions.
```
