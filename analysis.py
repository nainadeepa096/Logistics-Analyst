"""
analysis.py

Week 1 - Strategic Planning and Data Exploration in Logistics

Demonstrates the two core techniques proposed in the Week 1 report against
the cleaned SwiftCart dataset (produced by week2_data_cleaning/clean_pipeline.py):

  1. Regression  - predict delivery duration from distance / traffic / weather
  2. Clustering  - group delivery drop-off points into geographic zones (K-Means)

Run from the repo root, AFTER running the Week 2 cleaning script:
    python week2_data_cleaning/clean_pipeline.py
    python week1_strategic_planning/analysis.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

CLEAN_PATH = "data/swiftcart_deliveries_cleaned.csv"


def run_regression(df):
    """Predict delivery_minutes from distance, traffic, and weather."""
    print("--- Regression: Predicting Delivery Duration ---")

    # Encode categorical columns as numbers so they can be used as features
    df = df.copy()
    for col in ["traffic", "weather"]:
        df[col + "_enc"] = LabelEncoder().fit_transform(df[col])

    X = df[["distance_km", "traffic_enc", "weather_enc"]]
    y = df["delivery_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error on test set: {mae:.3f} (scaled units)")
    print(f"Model coefficients: {dict(zip(X.columns, model.coef_))}")
    return model


def run_clustering(df, n_clusters=5):
    """Group delivery drop-off coordinates into geographic zones."""
    print("\n--- Clustering: Delivery Zone Formation ---")

    coords = df[["drop_lat", "drop_lon"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["zone"] = kmeans.fit_predict(coords)

    print(f"Formed {n_clusters} zones. Orders per zone:")
    print(df["zone"].value_counts().sort_index())
    return df


def main():
    df = pd.read_csv(CLEAN_PATH)
    run_regression(df)
    run_clustering(df)


if __name__ == "__main__":
    main()
