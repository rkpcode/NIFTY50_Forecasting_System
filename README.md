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
- **ML Framework**: TensorFlow + Scikit-learn
- **Data Source**: Yahoo Finance (yfinance)
- **Model**: Ensemble ML models (XGBoost, Random Forest, etc.)

## Disclaimer

⚠️ This is an educational AI model. Do not use for actual trading decisions.
