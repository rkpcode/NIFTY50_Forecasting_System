
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import sys
import os

# Ensure src path is available
sys.path.append(os.getcwd())

from src.nifty50_forecasting_system.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

# Initialize pipeline once
pipeline = PredictionPipeline()

@app.route('/')
def home():
    return render_template_string("""
    <h1>NIFTY 50 Stock Forecasting API</h1>
    <p>Use /predict endpoint (POST) with JSON data containing OHLCV history (at least 60 records).</p>
    <p>Example JSON format: <code>[{"Date": "...", "Open": ..., "High": ..., "Low": ..., "Close": ..., "Volume": ...}, ...]</code></p>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Convert JSON to DataFrame
        df = pd.DataFrame(json_data)
        
        # Ensure we have required columns
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(col in df.columns for col in required_cols):
             return jsonify({'error': f'Missing columns. Required: {required_cols}'}), 400
             
        # Make prediction
        prediction = pipeline.predict(df)
        
        return jsonify({
            'symbol': 'NIFTY 50',
            'prediction': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
