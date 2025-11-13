# ChronoFit MVAA - AI-Powered Workout Recommendation System

Personalized workout recommendation engine powered by machine learning, featuring continuous learning from user feedback and cloud-based MongoDB storage.

## 🎯 Features

- 🏋️ AI-powered personalized workout recommendations based on recovery metrics
- 📊 Post-workout feedback loop for continuous learning
- ☁️ Cloud-based MongoDB storage for persistent data across sessions
- 🧠 Automatic model retraining every 3 user feedbacks
- 🥗 Intelligent multi-source nutrient lookup (USDA + Indian APIs + local database)
- 📱 Professional minimal UI optimized for Streamlit Cloud
- 🔄 Background threading for non-blocking model retraining

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git
- GitHub account
- MongoDB Atlas account (free M0 tier)
- USDA API key (free)

### Local Development

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/ChronoFit.git
cd ChronoFit

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create secrets file
mkdir .streamlit
echo 'MONGODB_URI = "your_mongodb_connection_string"
USDA_API_KEY = "your_usda_api_key"' > .streamlit/secrets.toml

# Run app
streamlit run chronofit_app.py
```

Visit: http://localhost:8501

## 📦 Deployment to Streamlit Cloud

### Step 1: Push to GitHub
```bash
cd ChronoFit
git add .
git commit -m "Update: Fixed notebook evaluation metrics"
git push
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Click "New app"
3. Select your repository (sairam-aleti/ChronoFit), branch (main), and file (chronofit_app.py)
4. Click "Deploy!"

### Step 3: Configure Secrets in Streamlit Cloud
1. Once deployed, click app settings (gear icon)
2. Go to "Secrets"
3. Add your environment variables:

```toml
MONGODB_URI = "<your-mongodb-connection-string>"
USDA_API_KEY = "<your-usda-api-key>"
```

4. Save - app will restart with secrets loaded

**App will be live at:** https://chronofit.streamlit.app

## 🏗️ Architecture

```
User Input (Age, Weight, Sleep, RHR, etc.)
        ↓
  Preprocessing Pipeline
        ↓
  ML Model (RandomForest)
        ↓
  Workout Recommendations (Duration, Intensity, Exercises)
        ↓
  Post-Workout Feedback (User rates satisfaction)
        ↓
  MongoDB Storage (Cloud)
        ↓
  Background Retraining (Every 3 feedbacks)
        ↓
  Updated Model Files
```

## 🧠 Continuous Learning

1. User enters fitness metrics and receives personalized workout recommendation
2. After workout, user provides feedback (completion %, intensity, recovery feeling, etc.)
3. Feedback automatically saved to MongoDB
4. **Background thread** checks if 3+ new feedbacks accumulated
5. If threshold met: Model retrains on 20k synthetic data + weighted user feedback
6. Updated models save to cloud storage
7. Next user session loads updated models → improved recommendations

## 📂 Files Overview

| File | Purpose |
|------|---------|
| `chronofit_app.py` | Main Streamlit web application |
| `mongodb_handler.py` | MongoDB connection & data operations |
| `mvva_data_generator.ipynb` | Synthetic data generation & retraining notebook |
| `mvva_model_v2.joblib` | Trained RandomForestRegressor for duration/intensity |
| `mvva_preprocessor_v2.joblib` | Scikit-learn preprocessing pipeline |
| `mvva_goal_classifier_v2.joblib` | Goal classification model |
| `mvva_goal_encoder_v2.joblib` | Goal encoding model |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python version for Streamlit Cloud |

## 🔑 Required Setup

### MongoDB Atlas (Free M0 Cluster)
1. Create account: https://www.mongodb.com/cloud/atlas
2. Create M0 cluster
3. Create database user in "Database Access"
4. Configure "Network Access" → Add `0.0.0.0/0` (allow all IPs for Streamlit Cloud)
5. Get connection string from "Connect" button
6. Replace `<USERNAME>` and `<PASSWORD>` in connection string

### USDA Food Data Central API (Free)
1. Sign up: https://api.nal.usda.gov/signup
2. Copy your API key
3. Add to `.streamlit/secrets.toml` as `USDA_API_KEY`

## ⚙️ Local Configuration

Create `.streamlit/secrets.toml`:
```toml
MONGODB_URI = "<your-mongodb-connection-string>"
USDA_API_KEY = "<your-api-key>"
```

**Note:** This file is in `.gitignore` and never committed to GitHub for security.

## 📊 Model Training Data

- **Base**: 20,000 synthetic workout records
- **Age Distribution**: Bimodal (28±6 years and 48±8 years)
- **Features**: Age, weight, sex, sleep, RHR, soreness, stress, nutrition
- **Targets**: Workout duration (min), intensity (1-10 scale)
- **User Feedback**: 2x weighted during retraining for rapid adaptation
- **Retraining Trigger**: Every 3 new user feedbacks

## 🔄 Deployment Workflow

```
Local Development
    ↓
Test Locally (streamlit run)
    ↓
Commit to Git (git add . && git commit -m "message")
    ↓
Push to GitHub (git push)
    ↓
Streamlit Cloud auto-deploys
    ↓
App live at chronofit.streamlit.app
```

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Models fail to load | Ensure all `.joblib` files committed to GitHub |
| MongoDB connection error | Check `MONGODB_URI` in Streamlit Cloud secrets |
| App crashes on startup | Check Python version matches `runtime.txt` |
| Retraining takes too long | Background threading prevents UI blocking - check Streamlit logs |
| Nutrient lookup returns 0 | Multiple data sources tried (USDA → Indian API → local database) |

## 📈 Performance Notes

- **First load**: 10-15 seconds (models load + UI renders)
- **Feedback retraining**: Happens in background thread (~2-3 seconds)
- **Model update frequency**: Every 3 feedbacks (adaptive)
- **Streamlit Cloud**: Free tier sufficient for MVP (3 concurrent sessions)

## 🔐 Security

- MongoDB connection string stored only in Streamlit Cloud secrets
- `.streamlit/secrets.toml` never committed to Git
- All APIs use HTTPS
- No user data stored locally on client
- Background threads isolated from main UI thread

## 📝 License

MIT

## 🎓 Learning Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [MongoDB Atlas Setup](https://docs.atlas.mongodb.com)
- [scikit-learn RandomForest](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [USDA Food Data Central API](https://fdc.nal.usda.gov/api-guide.html)

---

**Get Started:** Follow the Quick Start section to run locally, then deploy to Streamlit Cloud using the deployment steps above.
