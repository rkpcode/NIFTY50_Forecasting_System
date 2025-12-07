import streamlit as st
import pandas as pd
import requests
import time

st.set_page_config(page_title="NIFTY 50 Wise", layout="wide", page_icon="📈")

# Custom CSS for "Weather Card" look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .date-text {
        color: #888;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .price-text {
        font-size: 1.8em;
        font-weight: bold;
        color: #fff;
    }
    .trend-up {
        color: #00ff7f;
    }
    .trend-down {
        color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 NIFTY 50 Forecasting System")
st.markdown("### Next 7 Days Prediction (AI Powered)")

# Controls
col1, col2 = st.columns([1, 4])
with col1:
    retrain = st.checkbox("Online Learning (Retrain on latest data)", value=True, help="If checked, the model will fine-tune itself on the absolute latest live market data before predicting.")
with col2:
    if st.button("Get 7-Day Forecast", type="primary"):
        with st.spinner("Fetching Live Data & Forecasting... (This may take a moment if Online Learning is active)"):
            try:
                # Call Flask API
                api_url = "http://127.0.0.1:5000/predict_live"
                payload = {"retrain": retrain}
                
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    predictions = data.get("predictions", [])
                    
                    if not predictions:
                        st.warning("No predictions returned.")
                    else:
                        st.subheader("7-Day Forecast")
                        
                        # Display cards
                        cols = st.columns(7)
                        
                        previous_price = None # To calculate trend 
                        
                        for i, day_pred in enumerate(predictions):
                            date = day_pred["Date"]
                            price = day_pred["Price"]
                            
                            # Determine trend if we have previous price (from index 0 we rely on ... wait, index 0 is tomorrow. 
                            # We can't know today's close easily unless passed. 
                            # We'll compare day i to day i-1. For i=0, just show neutral or no trend icon).
                            
                            trend_icon = ""
                            trend_class = ""
                            
                            if i > 0:
                                if price > previous_price:
                                    trend_icon = "🔼"
                                    trend_class = "trend-up"
                                elif price < previous_price:
                                    trend_icon = "🔽"
                                    trend_class = "trend-down"
                                else:
                                    trend_icon = "➖"
                            
                            previous_price = price
                            
                            with cols[i]:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="date-text">{date}</div>
                                    <div class="price-text" title="{price}">₹{int(price)}</div>
                                    <div class="{trend_class}">{trend_icon}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Show raw data below
                        with st.expander("View Detailed Data"):
                            st.dataframe(pd.DataFrame(predictions))
                            
                        if data.get("retrained"):
                            st.success("✅ Model successfully updated with latest market data!")
                            
                else:
                    st.error(f"API Error: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend API. Is 'python app.py' running?")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("*Disclaimer: This is an AI generic model. Do not use for actual trading.*")
