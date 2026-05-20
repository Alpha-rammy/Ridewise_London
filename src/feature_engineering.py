from pathlib import Path
import pandas as pd
import numpy as np

import warnings
import os

warnings.filterwarnings("ignore")


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PROCESSED = BASE_DIR / "data" / "processed"


# =====================================================
# CONSTANTS
# =====================================================

PEAK_HOURS = list(range(7, 10)) + list(range(17, 20))

PAYMENT_MAP = {
    "Card": 0,
    "Cash": 1,
    "Wallet": 2,
    "Corporate": 3
}

LOYALTY_MAP = {
    "Bronze": 0,
    "Silver": 1,
    "Gold": 2,
    "Platinum": 3
}

CITY_MAP = {
    "Nairobi": 0,
    "Lagos": 1,
    "Cairo": 2
}


# =====================================================
# LOAD DATASETS
# =====================================================

def load_datasets():

    riders = pd.read_csv(
        os.path.join(DATA_PROCESSED, "riders_clean.csv"),
        parse_dates=["signup_date"]
    )

    drivers = pd.read_csv(
        os.path.join(DATA_PROCESSED, "drivers_clean.csv")
    )

    trips = pd.read_csv(
        os.path.join(DATA_PROCESSED, "trips_clean.csv"),
        parse_dates=["pickup_time", "dropoff_time"]
    )

    sessions = pd.read_csv(
        os.path.join(DATA_PROCESSED, "sessions_clean.csv"),
        parse_dates=["session_time"]
    )

    return riders, drivers, trips, sessions


# =====================================================
# CREATE ELIGIBLE RIDERS
# =====================================================

def create_eligible_riders(riders, trips, reference_date):

    riders_with_trips = set(trips["user_id"])

    riders["account_age_days"] = (
        reference_date - riders["signup_date"]
    ).dt.days

    eligible = riders[
        riders["user_id"].isin(riders_with_trips)
        &
        (riders["account_age_days"] >= 60)
    ][["user_id", "churned"]].copy()

    return riders, eligible


# =====================================================
# CREATE RECENCY FEATURES
# =====================================================

def create_recency_features(eligible, trips, sessions, reference_date):

    last_trip = (
        trips.groupby("user_id")["pickup_time"]
        .max()
        .reset_index(name="last_trip")
    )

    last_session = (
        sessions.groupby("rider_id")["session_time"]
        .max()
        .reset_index()
        .rename(
            columns={
                "session_time": "last_session",
                "rider_id": "user_id"
            }
        )
    )

    recency = (
        eligible[["user_id"]]
        .merge(last_trip, on="user_id", how="left")
        .merge(last_session, on="user_id", how="left")
    )

    recency["days_since_last_trip"] = (
        reference_date - recency["last_trip"]
    ).dt.days.fillna(999)

    recency["days_since_last_session"] = (
        reference_date - recency["last_session"]
    ).dt.days.fillna(999)

    return recency[[
        "user_id",
        "days_since_last_trip",
        "days_since_last_session"
    ]]


# =====================================================
# CREATE FREQUENCY FEATURES
# =====================================================

def create_frequency_features(eligible, trips, reference_date):

    frequency = eligible[["user_id"]].copy()

    for days in [7, 30, 60, 90]:

        cutoff = reference_date - pd.Timedelta(days=days)

        counts = (
            trips[trips["pickup_time"] >= cutoff]
            .groupby("user_id")
            .size()
            .reset_index(name=f"trips_last_{days}d")
        )

        frequency = frequency.merge(
            counts,
            on="user_id",
            how="left"
        )

    freq_cols = [
        f"trips_last_{d}d"
        for d in [7, 30, 60, 90]
    ]

    frequency[freq_cols] = (
        frequency[freq_cols]
        .fillna(0)
        .astype(int)
    )

    return frequency


# =====================================================
# CREATE MONETARY FEATURES
# =====================================================

def create_monetary_features(trips, reference_date):

    monetary = (
        trips.groupby("user_id")
        .agg(
            monetary_total=("total_revenue", "sum"),
            monetary_avg=("total_revenue", "mean"),
            trips_lifetime=("trip_id", "count")
        )
        .round(2)
        .reset_index()
    )

    cutoff_30 = reference_date - pd.Timedelta(days=30)

    monetary_30 = (
        trips[trips["pickup_time"] >= cutoff_30]
        .groupby("user_id")["total_revenue"]
        .sum()
        .reset_index(name="monetary_last_30d")
    )

    monetary = monetary.merge(
        monetary_30,
        on="user_id",
        how="left"
    )

    monetary["monetary_last_30d"] = (
        monetary["monetary_last_30d"]
        .fillna(0)
    )

    return monetary


# =====================================================
# CREATE RFM SCORES
# =====================================================

def create_rfm_scores(recency, frequency, monetary):

    rfm = (
        recency[["user_id", "days_since_last_trip"]]
        .merge(
            frequency[["user_id", "trips_last_30d"]],
            on="user_id",
            how="left"
        )
        .merge(
            monetary[["user_id", "monetary_total"]],
            on="user_id",
            how="left"
        )
        .fillna({
            "trips_last_30d": 0,
            "monetary_total": 0
        })
    )

    rfm["rfm_recency_score"] = pd.qcut(
        rfm["days_since_last_trip"].rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1]
    ).astype(int)

    rfm["rfm_frequency_score"] = pd.qcut(
        rfm["trips_last_30d"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["rfm_monetary_score"] = pd.qcut(
        rfm["monetary_total"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5]
    ).astype(int)

    rfm["rfm_combined_score"] = (
        rfm["rfm_recency_score"]
        + rfm["rfm_frequency_score"]
        + rfm["rfm_monetary_score"]
    ) / 3

    rfm["rfm_combined_score"] = (
        rfm["rfm_combined_score"]
        .round()
    )

    return rfm


# =====================================================
# CREATE BEHAVIORAL FEATURES
# =====================================================

def create_behavioral_features(trips, drivers):

    trips["is_peak"] = (
        trips["hour_of_day"]
        .isin(PEAK_HOURS)
        .astype(int)
    )

    trips["is_weekend"] = (
        trips["day_of_week"]
        .isin(["Saturday", "Sunday"])
        .astype(int)
    )

    trips["is_rainy"] = (
        (trips["weather"] == "Rainy")
        .astype(int)
    )

    behavioral = (
        trips.groupby("user_id")
        .agg(
            avg_trip_duration=("trip_duration", "mean"),
            avg_fare=("fare", "mean"),
            avg_surge=("surge_multiplier", "mean"),
            tip_rate=("tip", lambda x: (x > 0).mean()),
            peak_hour_rate=("is_peak", "mean"),
            weekend_ratio=("is_weekend", "mean"),
            rainy_ride_ratio=("is_rainy", "mean")
        )
        .round(2)
        .reset_index()
    )

    payment_mode = (
        trips.groupby("user_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )

    payment_mode["preferred_payment_encoded"] = (
        payment_mode["payment_type"]
        .map(PAYMENT_MAP)
        .fillna(0)
        .astype(int)
    )

    behavioral = behavioral.merge(
        payment_mode[[
            "user_id",
            "preferred_payment_encoded"
        ]],
        on="user_id",
        how="left"
    )

    trips_rated = trips.merge(
        drivers[["driver_id", "rating"]],
        on="driver_id",
        how="left"
    )

    driver_quality = (
        trips_rated.groupby("user_id")["rating"]
        .mean()
        .reset_index(name="avg_driver_rating_received")
        .round(3)
    )

    behavioral = behavioral.merge(
        driver_quality,
        on="user_id",
        how="left"
    )

    behavioral["avg_driver_rating_received"] = (
        behavioral["avg_driver_rating_received"]
        .fillna(
            behavioral[
                "avg_driver_rating_received"
            ].median()
        )
    )

    return behavioral


# =====================================================
# CREATE SESSION FEATURES
# =====================================================

def create_session_features(sessions, reference_date):

    sess_freq = pd.DataFrame({
        "user_id": sessions["rider_id"].unique()
    })

    for days in [30, 60]:

        cutoff = reference_date - pd.Timedelta(days=days)

        counts = (
            sessions[
                sessions["session_time"] >= cutoff
            ]
            .groupby("rider_id")
            .size()
            .reset_index(name=f"session_last_{days}d")
            .rename(columns={"rider_id": "user_id"})
        )

        sess_freq = sess_freq.merge(
            counts,
            on="user_id",
            how="left"
        )

    sess_freq[[
        "session_last_30d",
        "session_last_60d"
    ]] = (
        sess_freq[[
            "session_last_30d",
            "session_last_60d"
        ]]
        .fillna(0)
        .astype(int)
    )

    engagement = (
        sessions.groupby("rider_id")
        .agg(
            avg_time_on_app=("time_on_app", "mean"),
            avg_pages_visited=("pages_visited", "mean"),
            session_conversion_rate=("converted", "mean")
        )
        .round(2)
        .reset_index()
        .rename(columns={"rider_id": "user_id"})
    )

    engagement["engagement_score"] = (
        engagement["avg_time_on_app"]
        * engagement["avg_pages_visited"]
    ).round(2)

    session_features = sess_freq.merge(
        engagement,
        on="user_id",
        how="left"
    )

    fill_cols = [
        "avg_time_on_app",
        "avg_pages_visited",
        "session_conversion_rate",
        "engagement_score"
    ]

    session_features[fill_cols] = (
        session_features[fill_cols]
        .fillna(0)
    )

    return session_features


# =====================================================
# CREATE TEMPORAL FEATURES
# =====================================================

def create_temporal_features(
    riders,
    eligible,
    trips,
    reference_date
):

    temporal = riders[
        riders["user_id"].isin(
            eligible["user_id"]
        )
    ][[
        "user_id",
        "account_age_days"
    ]].copy()

    cutoff_30 = reference_date - pd.Timedelta(days=30)
    cutoff_60 = reference_date - pd.Timedelta(days=60)

    recent = (
        trips[
            trips["pickup_time"] >= cutoff_30
        ]
        .groupby("user_id")
        .size()
        .reset_index(name="_r30")
    )

    prior = (
        trips[
            (trips["pickup_time"] >= cutoff_60)
            &
            (trips["pickup_time"] < cutoff_30)
        ]
        .groupby("user_id")
        .size()
        .reset_index(name="_p30")
    )

    temporal = (
        temporal
        .merge(recent, on="user_id", how="left")
        .merge(prior, on="user_id", how="left")
    )

    temporal[["_r30", "_p30"]] = (
        temporal[["_r30", "_p30"]]
        .fillna(0)
    )

    temporal["activity_trend_30d"] = (
        temporal["_r30"] /
        (temporal["_p30"] + 1)
    ).round(2)

    temporal = temporal.drop(
        columns=["_r30", "_p30"]
    )

    trips["trip_date"] = (
        trips["pickup_time"].dt.date
    )

    unique_days = (
        trips.groupby("user_id")["trip_date"]
        .nunique()
        .reset_index(name="unique_active_days")
    )

    temporal = temporal.merge(
        unique_days,
        on="user_id",
        how="left"
    )

    temporal["unique_active_days"] = (
        temporal["unique_active_days"]
        .fillna(0)
        .astype(int)
    )

    return temporal


# =====================================================
# CREATE CATEGORICAL FEATURES
# =====================================================

def create_categorical_features(
    riders,
    eligible
):

    rider_cats = riders[
        riders["user_id"].isin(
            eligible["user_id"]
        )
    ][[
        "user_id",
        "loyalty_status",
        "city",
        "was_referred",
        "avg_rating_given"
    ]].copy()

    rider_cats["loyalty_encoded"] = (
        rider_cats["loyalty_status"]
        .map(LOYALTY_MAP)
    )

    rider_cats["city_encoded"] = (
        rider_cats["city"]
        .map(CITY_MAP)
    )

    rider_cats = rider_cats.drop(
        columns=["loyalty_status", "city"]
    )

    return rider_cats


# =====================================================
# MERGE ALL FEATURES
# =====================================================

def merge_feature_sets(
    eligible,
    recency,
    frequency,
    monetary,
    rfm,
    behavioral,
    session_features,
    temporal,
    rider_cats
):

    features = (
        eligible
        .merge(recency, on="user_id", how="left")
        .merge(frequency, on="user_id", how="left")
        .merge(monetary, on="user_id", how="left")
        .merge(rfm, on="user_id", how="left")
        .merge(behavioral, on="user_id", how="left")
        .merge(session_features, on="user_id", how="left")
        .merge(temporal, on="user_id", how="left")
        .merge(rider_cats, on="user_id", how="left")
    )

    features = features.fillna(0)

    return features


# =====================================================
# SAVE FEATURES
# =====================================================

def save_features(features):

    output_path = os.path.join(
        DATA_PROCESSED,
        "features.csv"
    )

    features.to_csv(
        output_path,
        index=False
    )

    print(
        f"Features saved successfully:\n"
        f"{output_path}"
    )


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print("Loading datasets...")

    riders, drivers, trips, sessions = (
        load_datasets()
    )

    reference_date = (
        trips["pickup_time"].max()
    )

    print("Creating eligible riders...")

    riders, eligible = (
        create_eligible_riders(
            riders,
            trips,
            reference_date
        )
    )

    print("Creating recency features...")

    recency = create_recency_features(
        eligible,
        trips,
        sessions,
        reference_date
    )

    print("Creating frequency features...")

    frequency = create_frequency_features(
        eligible,
        trips,
        reference_date
    )

    print("Creating monetary features...")

    monetary = create_monetary_features(
        trips,
        reference_date
    )

    print("Creating RFM scores...")

    rfm = create_rfm_scores(
        recency,
        frequency,
        monetary
    )

    print("Creating behavioral features...")

    behavioral = create_behavioral_features(
        trips,
        drivers
    )

    print("Creating session features...")

    session_features = (
        create_session_features(
            sessions,
            reference_date
        )
    )

    print("Creating temporal features...")

    temporal = create_temporal_features(
        riders,
        eligible,
        trips,
        reference_date
    )

    print("Creating categorical features...")

    rider_cats = (
        create_categorical_features(
            riders,
            eligible
        )
    )

    print("Merging all feature groups...")

    features = merge_feature_sets(
        eligible,
        recency,
        frequency,
        monetary,
        rfm,
        behavioral,
        session_features,
        temporal,
        rider_cats
    )

    print(
        f"\nFeature Matrix Shape:"
        f" {features.shape}"
    )

    

    print("Saving feature dataset...")

    save_features(features)

    print(
        "\nFeature engineering pipeline completed successfully"
    )


if __name__ == "__main__":
    main()