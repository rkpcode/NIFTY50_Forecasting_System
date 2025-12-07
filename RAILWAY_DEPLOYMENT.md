# Railway.app Deployment Guide - NIFTY 50 Forecasting System

## 🚀 Quick Deploy to Railway (5 Minutes)

### Step 1: Create Railway Account
1. Go to **https://railway.app/**
2. Click **"Login"** → **"Login with GitHub"**
3. Authorize Railway to access your GitHub

### Step 2: Deploy Project
1. Click **"New Project"**
2. Select **"Deploy from GitHub repo"**
3. Choose: **`rkpcode/NIFTY50_Forecasting_System`**
4. Railway will auto-detect Procfile and start building

### Step 3: Configure (Optional)
1. Go to **"Variables"** tab
2. Add (if needed):
   ```
   PORT=8501
   PYTHON_VERSION=3.11
   ```

### Step 4: Wait for Deployment
- Build takes **3-5 minutes**
- Watch logs in **"Deployments"** tab
- Status will show: Building → Deploying → Active

### Step 5: Get Your Live URL
1. Go to **"Settings"** tab
2. Scroll to **"Domains"**
3. Click **"Generate Domain"**
4. Your app will be live at: `https://yourapp.railway.app`

## ✅ Expected Result
- **Live URL** with real-time yfinance predictions
- **7-day forecast** working perfectly
- **Custom domain** option available

## 💡 Tips
- **Free tier:** 500 hours/month (16 hours/day)
- **Logs:** Check if any errors
- **Redeploy:** Push to GitHub → auto redeploys

## 🔧 Troubleshooting

### Build fails?
- Check **logs** in Railway dashboard
- Ensure `requirements.txt` has all dependencies
- Check Python version compatibility

### App shows error?
- Click **"View Logs"**
- Look for model loading errors
- Ensure `artifacts/` folder is in Git

## 🎉 Success!
Your app will have **real-time predictions** like local machine!

---
**Support:** railway.app/help
