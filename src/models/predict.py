import os
import pickle
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict_churn(input_data: dict):

    df = pd.DataFrame([input_data])

    # force exact feature order
    expected_features = [
    "days_since_last_trip_x",
    "days_since_last_session",
    "trips_last_7d",
    "trips_last_30d_x",
    "trips_last_60d",
    "trips_last_90d",
    "monetary_total_x",
    "monetary_avg",
    "trips_lifetime",
    "monetary_last_30d",
    "days_since_last_trip_y",
    "trips_last_30d_y",
    "monetary_total_y",
    "rfm_recency_score",
    "rfm_frequency_score",
    "rfm_monetary_score",
    "rfm_combined_score",
    "avg_trip_duration",
    "avg_fare",
    "avg_surge",
    "tip_rate",
    "peak_hour_rate",
    "weekend_ratio",
    "rainy_ride_ratio",
    "preferred_payment_encoded",
    "avg_driver_rating_received",
    "session_last_30d",
    "session_last_60d",
    "avg_time_on_app",
    "avg_pages_visited",
    "session_conversion_rate",
    "engagement_score",
    "account_age_days",
    "activity_trend_30d",
    "unique_active_days",
    "was_referred",
    "avg_rating_given",
    "loyalty_encoded",
    "city_encoded"
]

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_features]

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability)
    }