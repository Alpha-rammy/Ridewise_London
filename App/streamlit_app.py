import streamlit as st
import pandas as pd
import sys
import os

st.set_page_config(
    page_title="Ridewise Churn Prediction",
    page_icon="🚗",
    layout="centered"
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.models.predict import predict_churn

st.title("🚗 Ridewise Churn Prediction App")
st.write("Enter rider details to predict churn risk.")

days_since_last_trip_x = st.number_input("Days Since Last Trip X", value=7.0)
days_since_last_session = st.number_input("Days Since Last Session", value=5.0)
trips_last_7d = st.number_input("Trips Last 7 Days", value=2.0)
trips_last_30d_x = st.number_input("Trips Last 30 Days X", value=8.0)
trips_last_60d = st.number_input("Trips Last 60 Days", value=15.0)
trips_last_90d = st.number_input("Trips Last 90 Days", value=22.0)
monetary_total_x = st.number_input("Monetary Total X", value=250.0)
monetary_avg = st.number_input("Monetary Average", value=18.0)
trips_lifetime = st.number_input("Trips Lifetime", value=40.0)
monetary_last_30d = st.number_input("Monetary Last 30 Days", value=120.0)
days_since_last_trip_y = st.number_input("Days Since Last Trip Y", value=7.0)
trips_last_30d_y = st.number_input("Trips Last 30 Days Y", value=8.0)
monetary_total_y = st.number_input("Monetary Total Y", value=250.0)
rfm_recency_score = st.number_input("RFM Recency Score", value=3.0)
rfm_frequency_score = st.number_input("RFM Frequency Score", value=3.0)
rfm_monetary_score = st.number_input("RFM Monetary Score", value=3.0)
rfm_combined_score = st.number_input("RFM Combined Score", value=333.0)
avg_trip_duration = st.number_input("Average Trip Duration", value=20.0)
avg_fare = st.number_input("Average Fare", value=15.0)
avg_surge = st.number_input("Average Surge", value=1.0)
tip_rate = st.number_input("Tip Rate", value=0.2)
peak_hour_rate = st.number_input("Peak Hour Rate", value=0.4)
weekend_ratio = st.number_input("Weekend Ratio", value=0.2)
rainy_ride_ratio = st.number_input("Rainy Ride Ratio", value=0.1)
preferred_payment_encoded = st.number_input("Preferred Payment Encoded", value=1.0)
avg_driver_rating_received = st.number_input("Average Driver Rating Received", value=4.7)
session_last_30d = st.number_input("Sessions Last 30 Days", value=10.0)
session_last_60d = st.number_input("Sessions Last 60 Days", value=20.0)
avg_time_on_app = st.number_input("Average Time on App", value=8.0)
avg_pages_visited = st.number_input("Average Pages Visited", value=5.0)
session_conversion_rate = st.number_input("Session Conversion Rate", value=0.4)
engagement_score = st.number_input("Engagement Score", value=50.0)
account_age_days = st.number_input("Account Age Days", value=365.0)
activity_trend_30d = st.number_input("Activity Trend 30 Days", value=0.0)
unique_active_days = st.number_input("Unique Active Days", value=20.0)
was_referred = st.number_input("Was Referred", value=1.0)
avg_rating_given = st.number_input("Average Rating Given", value=4.6)
loyalty_encoded = st.number_input("Loyalty Encoded", value=1.0)
city_encoded = st.number_input("City Encoded", value=1.0)

input_data = {
    "days_since_last_trip_x": days_since_last_trip_x,
    "days_since_last_session": days_since_last_session,
    "trips_last_7d": trips_last_7d,
    "trips_last_30d_x": trips_last_30d_x,
    "trips_last_60d": trips_last_60d,
    "trips_last_90d": trips_last_90d,
    "monetary_total_x": monetary_total_x,
    "monetary_avg": monetary_avg,
    "trips_lifetime": trips_lifetime,
    "monetary_last_30d": monetary_last_30d,
    "days_since_last_trip_y": days_since_last_trip_y,
    "trips_last_30d_y": trips_last_30d_y,
    "monetary_total_y": monetary_total_y,
    "rfm_recency_score": rfm_recency_score,
    "rfm_frequency_score": rfm_frequency_score,
    "rfm_monetary_score": rfm_monetary_score,
    "rfm_combined_score": rfm_combined_score,
    "avg_trip_duration": avg_trip_duration,
    "avg_fare": avg_fare,
    "avg_surge": avg_surge,
    "tip_rate": tip_rate,
    "peak_hour_rate": peak_hour_rate,
    "weekend_ratio": weekend_ratio,
    "rainy_ride_ratio": rainy_ride_ratio,
    "preferred_payment_encoded": preferred_payment_encoded,
    "avg_driver_rating_received": avg_driver_rating_received,
    "session_last_30d": session_last_30d,
    "session_last_60d": session_last_60d,
    "avg_time_on_app": avg_time_on_app,
    "avg_pages_visited": avg_pages_visited,
    "session_conversion_rate": session_conversion_rate,
    "engagement_score": engagement_score,
    "account_age_days": account_age_days,
    "activity_trend_30d": activity_trend_30d,
    "unique_active_days": unique_active_days,
    "was_referred": was_referred,
    "avg_rating_given": avg_rating_given,
    "loyalty_encoded": loyalty_encoded,
    "city_encoded": city_encoded
}

if st.button("Predict Churn"):
    result = predict_churn(input_data)

    probability = result["churn_probability"]
    prediction = result["churn_prediction"]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"High churn risk: {probability:.2%}")
    else:
        st.success(f"Low churn risk: {probability:.2%}")

    st.json(result)