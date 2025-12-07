import requests
import time
import json

def test_api():
    url = "http://127.0.0.1:5000/predict_live"
    payload = {"retrain": True}
    
    print("Sending request to /predict_live with retrain=True...")
    start = time.time()
    try:
        response = requests.post(url, json=payload)
        duration = time.time() - start
        
        print(f"Request took {duration:.2f} seconds")
        
        if response.status_code == 200:
            print("Success!")
            data = response.json()
            print(json.dumps(data, indent=2))
            
            if data.get("retrained"):
                print("✅ Retraining confirmed.")
            else:
                print("❌ Retraining flag missing or false.")
                
            predictions = data.get("predictions", [])
            if len(predictions) == 7:
                 print("✅ Received 7-day forecast.")
            else:
                 print(f"❌ Expected 7 predictions, got {len(predictions)}")
                 
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Wait a bit for server to start if running immediately after spawn
    time.sleep(5) 
    test_api()
