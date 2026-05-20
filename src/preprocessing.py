from pathlib import Path
import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


# =====================================================
# PATH CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"


# =====================================================
# LOAD DATASETS
# =====================================================

def load_datasets():

    riders = pd.read_csv(
        os.path.join(DATA_RAW, "riders.csv"),
        parse_dates=["signup_date"]
    )

    trips = pd.read_csv(
        os.path.join(DATA_RAW, "trips.csv")
    )

    drivers = pd.read_csv(
        os.path.join(DATA_RAW, "drivers.csv"),
        parse_dates=["signup_date"]
    )

    sessions = pd.read_csv(
        os.path.join(DATA_RAW, "sessions.csv")
    )

    promotions = pd.read_csv(
        os.path.join(DATA_RAW, "promotions.csv")
    )

    return riders, trips, drivers, sessions, promotions


# =====================================================
# CLEAN RIDERS DATASET
# =====================================================

def clean_riders(riders):

    riders_clean = riders.copy()

    riders_clean["age"] = (
        riders_clean["age"]
        .round()
        .astype(int)
    )

    riders_clean["churned"] = (
        riders_clean["churn_prob"] > 0.5
    ).astype(int)

    riders_clean = riders_clean.drop(
        columns=["churn_prob"]
    )

    riders_clean["was_referred"] = (
        riders_clean["referred_by"]
        .notna()
        .astype(int)
    )

    riders_clean = riders_clean.drop(
        columns=["referred_by"]
    )

    return riders_clean


# =====================================================
# CLEAN TRIPS DATASET
# =====================================================

def clean_trips(trips):

    trips_clean = trips.copy()

    for col in ["pickup_time", "dropoff_time"]:

        trips_clean[col] = (
            pd.to_datetime(
                trips_clean[col],
                utc=True,
                errors="coerce"
            )
            .dt.tz_localize(None)
        )

    trips_clean = trips_clean.dropna(
        subset=["pickup_time", "dropoff_time"]
    )

    # Feature Engineering
    trips_clean["trip_duration"] = (
        trips_clean["dropoff_time"]
        - trips_clean["pickup_time"]
    ).dt.total_seconds() / 60

    trips_clean["total_revenue"] = (
        trips_clean["fare"]
        * trips_clean["surge_multiplier"]
        + trips_clean["tip"].fillna(0)
    )

    trips_clean["hour_of_day"] = (
        trips_clean["pickup_time"].dt.hour
    )

    trips_clean["day_of_week"] = (
        trips_clean["pickup_time"].dt.day_name()
    )

    # Data Quality
    trips_clean = trips_clean[
        trips_clean["trip_duration"] > 0
    ]

    trips_clean["tip"] = (
        trips_clean["tip"].fillna(0)
    )

    trips_clean["weather"] = (
        trips_clean["weather"]
        .fillna("unknown")
    )

    return trips_clean


# =====================================================
# CLEAN DRIVERS DATASET
# =====================================================

def clean_drivers(drivers):

    drivers_clean = drivers.copy()

    drivers_clean["rating"] = (
        drivers_clean["rating"]
        .fillna(
            drivers_clean["rating"].median()
        )
    )

    drivers_clean["acceptance_rate"] = (
        drivers_clean["acceptance_rate"]
        .fillna(
            drivers_clean["acceptance_rate"].median()
        )
    )

    return drivers_clean


# =====================================================
# CLEAN SESSIONS DATASET
# =====================================================

def clean_sessions(sessions):

    sessions_clean = sessions.copy()

    sessions_clean["session_time"] = (
        pd.to_datetime(
            sessions_clean["session_time"],
            utc=True,
            errors="coerce"
        )
        .dt.tz_localize(None)
    )

    sessions_clean = sessions_clean.dropna(
        subset=["session_time"]
    )

    return sessions_clean


# =====================================================
# REFERENTIAL INTEGRITY
# =====================================================

def validate_integrity(
    riders_clean,
    drivers_clean,
    trips_clean,
    sessions_clean
):

    valid_riders = set(
        riders_clean["user_id"]
    )

    valid_drivers = set(
        drivers_clean["driver_id"]
    )

    before_trips = len(trips_clean)
    before_sessions = len(sessions_clean)

    trips_clean = trips_clean[
        trips_clean["user_id"].isin(valid_riders)
    ]

    trips_clean = trips_clean[
        trips_clean["driver_id"].isin(valid_drivers)
    ]

    sessions_clean = sessions_clean[
        sessions_clean["rider_id"].isin(valid_riders)
    ]

    print(
        f"Trips dropped: "
        f"{before_trips - len(trips_clean)}"
    )

    print(
        f"Sessions dropped: "
        f"{before_sessions - len(sessions_clean)}"
    )

    return trips_clean, sessions_clean


# =====================================================
# SAVE DATASETS
# =====================================================

def save_processed_data(files):

    os.makedirs(DATA_PROCESSED, exist_ok=True)

    for fname, df in files.items():

        df.to_csv(
            os.path.join(DATA_PROCESSED, fname),
            index=False
        )

        print(f"{fname} saved successfully")


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print("Loading datasets...")

    riders, trips, drivers, sessions, promotions = (
        load_datasets()
    )

    print("Cleaning riders...")
    riders_clean = clean_riders(riders)

    print("Cleaning trips...")
    trips_clean = clean_trips(trips)

    print("Cleaning drivers...")
    drivers_clean = clean_drivers(drivers)

    print("Cleaning sessions...")
    sessions_clean = clean_sessions(sessions)

    print("Validating integrity...")

    trips_clean, sessions_clean = (
        validate_integrity(
            riders_clean,
            drivers_clean,
            trips_clean,
            sessions_clean
        )
    )

    files = {
        "riders_clean.csv": riders_clean,
        "trips_clean.csv": trips_clean,
        "drivers_clean.csv": drivers_clean,
        "sessions_clean.csv": sessions_clean
    }

    print("Saving processed datasets...")

    save_processed_data(files)

    print(
        "\\nPreprocessing pipeline completed successfully"
    )


if __name__ == "__main__":
    main()