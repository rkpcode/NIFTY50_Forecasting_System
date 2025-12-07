import streamlit as st
import pandas as pd
import sys
import os
import traceback

# Ensure src path is available
sys.path.append(os.getcwd())

from src.nifty50_forecasting_system.pipelines.prediction_pipeline import PredictionPipeline

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

# Initialize pipeline with caching
@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

try:
    pipeline = load_pipeline()
    # Check what models loaded
    available_models = list(pipeline.models.keys()) if hasattr(pipeline, 'models') else []
    if available_models:
         st.success(f"✅ Models loaded: {', '.join([m.upper() for m in available_models])}")
    else:
         st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.video("https://www.youtube.com/watch?v=dQw4w9WgXcQ") # Easter egg handling? No, let's stick to standard error.
    st.stop()

# ===== SIDEBAR: Model Selection =====
st.sidebar.title("⚙️ Model Settings")
st.sidebar.markdown("---")

# Get available models
available_models = list(pipeline.models.keys()) if hasattr(pipeline, 'models') else ["legacy"]

# Model selector
if len(available_models) > 1:
    model_choices = {
        "seq2seq": "🏆 Seq2Seq LSTM (Best - RMSE: 228)",
        "multivariate": "📊 Multivariate LSTM (All Features - RMSE: 1083)"
    }
    
    # Create display names for available models
    display_options = [model_choices.get(m, m.upper()) for m in available_models]
    
    # Default to seq2seq if available
    default_idx = 0
    if 'seq2seq' in available_models:
        default_idx = available_models.index('seq2seq')
    
    selected_display = st.sidebar.selectbox(
        "Select Forecasting Model",
        display_options,
        index=default_idx,
        help="Choose between different LSTM architectures"
    )
    
    # Map back to internal model name
    # Find key by value in the list constructed
    selected_internal_index = display_options.index(selected_display)
    model_type = available_models[selected_internal_index]
else:
    # Use whatever model is active (likely legacy or single model loaded)
    model_type = available_models[0] if available_models else "multivariate"
    st.sidebar.info(f"Using: {model_type.upper()} LSTM")

# Model comparison info
with st.sidebar.expander("📊 Model Comparison"):
    st.markdown("""
    ### Seq2Seq LSTM ⭐ (Recommended)
    - **RMSE:** 228.00 (Step 1)
    - **Architecture:** Encoder-Decoder
    - **Best for:** 1-7 day forecasts
    - **Approach:** Univariate (price-based)
    
    ### Multivariate LSTM
    - **RMSE:** 1083.22  
    - **Architecture:** Stacked LSTM
    - **Features:** 13+ indicators
    - **Note:** More complex but noisier
    
    > 💡 **Seq2Seq performs ~80% better!**
    """)

st.sidebar.markdown("---")
st.sidebar.caption("Model comparison from `model_comparison_report.md`")

# ===== MAIN CONTROLS =====
col1, col2 = st.columns([1, 4])
with col1:
    retrain = st.checkbox("Online Learning (Retrain on latest data)", value=False, 
                         help="If checked, the model will fine-tune itself on the absolute latest live market data before predicting. This takes ~30 seconds.")

# Show selected model info
st.info(f"🤖 Active Model: **{model_type.upper()} LSTM**")

with col2:
    if st.button("Get 7-Day Forecast", type="primary"):
        with st.spinner(f"🔄 Running {model_type.upper()} forecast... (Please wait 30-60 seconds)"):
            try:
                # Call prediction pipeline with selected model
                if hasattr(pipeline, 'predict_next_n_days'):
                    # Check if method accepts model_type
                    # We updated it, so it should.
                    predictions = pipeline.predict_next_n_days(
                        steps=7, 
                        retrain=retrain,
                        model_type=model_type
                    )
                else:
                    st.error("Pipeline method predict_next_n_days not found.")
                    predictions = []
                
                if not predictions:
                    st.warning("No predictions returned.")
                else:
                    st.subheader("7-Day Forecast")
                    
                    # Display cards
                    cols = st.columns(7)
                    
                    previous_price = None
                    
                    for i, day_pred in enumerate(predictions):
                        date = day_pred["Date"]
                        price = day_pred["Price"]
                        
                        trend_icon = ""
                        trend_class = ""
                        
                        if i > 0 and previous_price is not None:
                            if price > previous_price:
                                trend_icon = "🔼"
                                trend_class = "trend-up"
                            elif price < previous_price:
                                trend_icon = "🔽"
                                trend_class = "trend-down"
                            else:
                                trend_icon = "➖"
                        
                        previous_price = price
                        
                        # Handle potential index out of bounds if cols < 7? No, streamlits creates N columns.
                        if i < len(cols):
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
                        
                    if retrain:
                        st.success("✅ Model successfully updated with latest market data!")
                        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

st.markdown("---")
st.markdown("*Disclaimer: This is an AI model for educational purposes. Do not use for actual trading.*")
