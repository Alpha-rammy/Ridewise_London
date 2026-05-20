import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import uvicorn

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

# =========================================================
# LOAD MODEL + ARTIFACTS
# =========================================================

model = joblib.load(
    os.path.join(MODEL_DIR, "churn_model.pkl")
)

FEATURE_COLS = joblib.load(
    os.path.join(MODEL_DIR, "feature_cols.pkl")
)

optimal_threshold = joblib.load(
    os.path.join(MODEL_DIR, "optimal_threshold.pkl")
)

print("Model and artifacts loaded successfully!")

# Print expected features
print(FEATURE_COLS)

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Ridewise Churn Prediction API",
    description="Predict customer churn using trained XGBoost model",
    version="0.1"
)

# =========================================================
# INPUT SCHEMA
# IMPORTANT:
# Replace these example fields with ALL features printed
# by FEATURE_COLS in your terminal
# =========================================================

class RidewiseFeatures(BaseModel):

    days_since_last_trip_x: float
    days_since_last_session: float
    trips_last_7d: float
    trips_last_30d_x: float
    trips_last_60d: float
    trips_last_90d: float
    monetary_total_x: float
    monetary_avg: float
    trips_lifetime: float
    monetary_last_30d: float
    days_since_last_trip_y: float
    trips_last_30d_y: float
    monetary_total_y: float
    rfm_recency_score: float
    rfm_frequency_score: float
    rfm_monetary_score: float
    rfm_combined_score: float
    avg_trip_duration: float
    avg_fare: float
    avg_surge: float
    tip_rate: float
    peak_hour_rate: float
    weekend_ratio: float
    rainy_ride_ratio: float
    preferred_payment_encoded: float
    avg_driver_rating_received: float
    session_last_30d: float
    session_last_60d: float
    avg_time_on_app: float
    avg_pages_visited: float
    session_conversion_rate: float
    engagement_score: float
    account_age_days: float
    activity_trend_30d: float
    unique_active_days: float
    was_referred: float
    avg_rating_given: float
    loyalty_encoded: float
    city_encoded: float
    
    

# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def welcome_root():

    return {
        "message": "Welcome to Ridewise Churn Prediction API"
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(rider: RidewiseFeatures):

    # Convert JSON input to dataframe
    data = pd.DataFrame([rider.model_dump()])

    # Ensure correct column order
    data = data[FEATURE_COLS]

    # Predict probability
    churn_probability = model.predict_proba(data)[:, 1][0]

    # Convert probability to class prediction
    churn_prediction = int(
        churn_probability >= optimal_threshold
    )

    return {

        "churn_prediction": churn_prediction,

        "churn_probability": round(
            float(churn_probability), 4
        ),

        "threshold_used": round(
            float(optimal_threshold), 4
        )
    }


# =========================================================
# RUN SERVER
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )