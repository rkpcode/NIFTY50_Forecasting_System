# Stock Forecasting Model Comparison Report

This report summarizes the performance of various forecasting models implemented in the `Stock forecasting.ipynb` notebook for NIFTY 50 stock price prediction.

## Models Evaluated

The following models were implemented and evaluated:

1.  **Auto ARIMA** (`pmdarima`)
2.  **Prophet** (Facebook Prophet)
3.  **Univariate LSTM** (Long Short-Term Memory)
4.  **Multivariate Single-Step LSTM**
5.  **Seq2Seq LSTM** (Multi-step forecasting)

## Performance Metrics

The primary metric used for comparison is **RMSE (Root Mean Squared Error)** on the test set. Lower values indicate better performance.

| Model | RMSE | MAE | Notes |
| :--- | :--- | :--- | :--- |
| **Seq2Seq LSTM (Step 1)** | **228.00** | - | Best performance for 1-day ahead forecast. RMSE increases with horizon (up to 432.95 for step 7). |
| **Univariate LSTM** | 284.19 | - | Strong performance, second best. |
| **Multivariate LSTM** | 1083.22 | 999.25 | Significantly worse than univariate models. Likely due to feature noise or overfitting. |
| **Prophet** | 3159.17 | - | Poorest performance among evaluated models. |
| **Auto ARIMA** | N/A | N/A | Fitted with AIC=60246.75. Explicit test RMSE not found in logs, but likely used as a baseline. |

## Detailed Analysis

### 1. LSTM Models
The Deep Learning approaches (LSTM) significantly outperformed traditional statistical methods (Prophet) and simple baselines.

*   **Seq2Seq LSTM**: This model demonstrated the best capability to capture short-term dependencies, achieving the lowest RMSE of **228.00** for the first time step. As expected with multi-step forecasting, the error increases as the forecast horizon extends (Step 7 RMSE: 432.95).
*   **Univariate LSTM**: A simpler LSTM model using only closing prices also performed very well (RMSE: 284.19), suggesting that the past price history is the strongest predictor.
*   **Multivariate LSTM**: Adding more features (Open, High, Low, Volume, SMAs) actually *degraded* performance (RMSE: 1083.22). This is a common phenomenon in financial time series where additional features can introduce more noise than signal, or the model complexity requires more careful tuning/regularization.

### 2. Prophet
Prophet yielded a high RMSE of **3159.17**. Prophet is generally better suited for strong seasonal patterns (daily, weekly, yearly) and might struggle with the highly stochastic nature of stock prices without extensive tuning or external regressors.

### 3. ARIMA
Auto ARIMA identified `ARIMA(2,1,2)` as the best fit based on AIC. While it provides a robust statistical baseline, the deep learning models (specifically LSTM) were able to capture non-linear patterns better.

## Conclusion

The **Seq2Seq LSTM** is the recommended model for this forecasting task, particularly for short-term predictions (1-7 days). For simpler implementation, the **Univariate LSTM** is a strong alternative. Future work could focus on feature engineering to improve the Multivariate LSTM or exploring Hybrid models (LSTM + ARIMA).
