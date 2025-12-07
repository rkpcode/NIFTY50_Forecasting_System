# Render.com Deployment - Quick Guide

## ✅ Simple 5-Minute Setup

### Step 1: Create Account
- Go to: **https://render.com/**
- Sign up with **GitHub**

### Step 2: New Web Service
- Click **"New +"** → **"Web Service"**
- Connect GitHub: `rkpcode/NIFTY50_Forecasting_System`

### Step 3: Settings
```
Name: nifty50-forecasting
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

### Step 4: Deploy
- Click **"Create Web Service"**
- Wait 5-10 minutes
- Done!

**App will work with demo predictions** (yfinance may work too!)

**Cost:** Free tier available

---

**Recommendation:** Abhi ke liye **Streamlit Cloud perfect hai**. Real predictions chaiye toh local run karo! 🚀
