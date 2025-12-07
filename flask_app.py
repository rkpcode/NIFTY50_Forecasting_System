
from flask import Flask, request, jsonify, render_template
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
    return render_template('index.html')

@app.route('/predict_dummy', methods=['POST'])
def predict_dummy():
    try:
        # Load dummy data logic
        dummy_path = "artifacts/test.csv"
        if not os.path.exists(dummy_path):
             return jsonify({'error': 'Dummy data not found on server.'}), 404
        
        df = pd.read_csv(dummy_path)
        prediction = pipeline.predict(df)
        
        return jsonify({
            'prediction': float(prediction),
            'source': 'dummy_data'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_upload', methods=['POST'])
def predict_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        try:
            df = pd.read_csv(file)
        except Exception:
            return jsonify({'error': 'Invalid CSV file'}), 400
            
        # Ensure we have required columns (PredictionPipeline checks this too, but good distinct check)
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
             return jsonify({'error': f'Missing columns: {missing}'}), 400
             
        # Make prediction
        prediction = pipeline.predict(df)
        
        return jsonify({
            'prediction': float(prediction),
            'source': 'upload'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_live', methods=['POST'])
def predict_live():
    try:
        # Check if retraining is requested
        data = request.get_json(silent=True)
        retrain = False
        if data and 'retrain' in data:
            retrain = bool(data['retrain'])
            
        # Predict next 7 days using live data (and retrain if requested)
        predictions = pipeline.predict_next_n_days(steps=7, retrain=retrain)
        
        return jsonify({
            'predictions': predictions,
            'source': 'live_7_days',
            'retrained': retrain
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Keep original API endpoint for compatibility
@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if not json_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        df = pd.DataFrame(json_data)
        prediction = pipeline.predict(df)
        
        return jsonify({
            'symbol': 'NIFTY 50',
            'prediction': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
