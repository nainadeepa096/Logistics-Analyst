"""
clean_pipeline.py

Week 2 - Data Collection, Cleaning, and Preprocessing for Logistics Analysis

Loads the raw (intentionally messy) SwiftCart delivery dataset and runs it
through a full cleaning + preprocessing pipeline:
  1. Load and inspect
  2. Fix data types and remove duplicates
  3. Handle missing values
  4. Compute delivery duration and drop invalid rows
  5. Treat outliers using the IQR method
  6. Normalize numeric features
  7. Save the cleaned dataset

Run from the repo root:
    python week2_data_cleaning/clean_pipeline.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/swiftcart_deliveries.csv"
CLEAN_PATH = "data/swiftcart_deliveries_cleaned.csv"


def load_and_inspect(path):
    df = pd.read_csv(path)
    print("--- 1. Load and Inspect ---")
    print("Shape:", df.shape)
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nDtypes:\n", df.dtypes)
    return df


def fix_types_and_duplicates(df):
    df["order_time"] = pd.to_datetime(df["order_time"], errors="coerce")
    df["pickup_time"] = pd.to_datetime(df["pickup_time"], errors="coerce")
    df["delivery_time"] = pd.to_datetime(df["delivery_time"], errors="coerce")

    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    print(f"\n--- 2. Fix Types and Duplicates ---")
    print(f"Removed {before - len(df)} duplicate order(s). New shape: {df.shape}")
    return df


def handle_missing_values(df):
    df["agent_rating"] = df["agent_rating"].fillna(df["agent_rating"].median())
    df["weather"] = df["weather"].fillna("Unknown")
    df["traffic"] = df["traffic"].fillna(df["traffic"].mode()[0])
    print("\n--- 3. Handle Missing Values ---")
    print("Remaining missing values:\n", df.isnull().sum())
    return df


def compute_duration_and_filter(df):
    df["delivery_minutes"] = (
        df["delivery_time"] - df["pickup_time"]
    ).dt.total_seconds() / 60

    before = len(df)
    df = df[df["delivery_minutes"] > 0]
    print(f"\n--- 4. Compute Duration & Filter Invalid Rows ---")
    print(f"Dropped {before - len(df)} row(s) with zero/negative duration. New shape: {df.shape}")
    return df


def treat_outliers(df, column="distance_km"):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    n_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
    df[column] = np.clip(df[column], lower_bound, upper_bound)

    print(f"\n--- 5. Outlier Treatment ({column}) ---")
    print(f"Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]  |  Capped {n_outliers} outlier(s)")
    return df


def normalize_features(df, columns):
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    print(f"\n--- 6. Normalization ---")
    print(f"Scaled columns: {columns}")
    return df


def main():
    df = load_and_inspect(RAW_PATH)
    df = fix_types_and_duplicates(df)
    df = handle_missing_values(df)
    df = compute_duration_and_filter(df)
    df = treat_outliers(df, "distance_km")
    df = normalize_features(df, ["distance_km", "agent_rating", "delivery_minutes"])

    print("\n--- 7. Final Validation ---")
    print("Total remaining missing values:", df.isnull().sum().sum())
    print("Final shape:", df.shape)

    df.to_csv(CLEAN_PATH, index=False)
    print(f"\nSaved cleaned dataset to {CLEAN_PATH}")


if __name__ == "__main__":
    main()
