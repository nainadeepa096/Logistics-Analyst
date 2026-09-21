import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 400

regions = ["North", "South", "East", "West", "Central"]
weather_opts = ["Clear", "Rain", "Snow", "Fog"]
weather_probs = [0.65, 0.20, 0.08, 0.07]
weather_delay_mult = {"Clear": 1.00, "Rain": 1.18, "Snow": 1.45, "Fog": 1.12}

region = rng.choice(regions, size=N, p=[0.25, 0.20, 0.20, 0.20, 0.15])
region_base_dist = {"North": 6.5, "South": 7.8, "East": 5.2, "West": 9.1, "Central": 3.8}
distance_km = np.array([max(0.8, rng.normal(region_base_dist[r], 2.1)) for r in region])

hour_of_day = rng.integers(7, 21, size=N)  # delivery window 7am-9pm
day_of_week = rng.integers(0, 7, size=N)   # 0=Mon ... 6=Sun
weather = rng.choice(weather_opts, size=N, p=weather_probs)

def rush_hour_mult(h):
    if h in (8, 9, 17, 18):
        return 1.35
    if h in (12, 13):
        return 1.10
    return 1.0

traffic_index = np.clip(
    np.array([rush_hour_mult(h) for h in hour_of_day]) * rng.normal(0.42, 0.10, size=N),
    0.05, 0.95
)

shipment_volume = rng.poisson(3, size=N) + 1  # packages per stop, 1-ish to ~8
weight_kg = np.round(shipment_volume * rng.normal(1.8, 0.4, size=N).clip(0.4, None), 2)

base_speed_min_per_km = 3.1
delivery_minutes = (
    distance_km * base_speed_min_per_km
    * (1 + traffic_index * 0.9)
    * np.array([weather_delay_mult[w] for w in weather])
    + shipment_volume * rng.normal(1.6, 0.5, size=N).clip(0.2, None)
    + rng.normal(6, 4, size=N)
)
delivery_minutes = np.clip(delivery_minutes, 8, None)

promised_window_min = np.select(
    [distance_km <= 4, distance_km <= 8],
    [30, 45],
    default=60
)
delay_flag = (delivery_minutes > promised_window_min).astype(int)

cost_per_km = 0.85
cost_per_pkg = 0.60
surge = np.where(np.isin(hour_of_day, [8, 9, 17, 18]), 1.15, 1.0)
transportation_cost = np.round(
    (distance_km * cost_per_km + shipment_volume * cost_per_pkg) * surge
    * np.array([weather_delay_mult[w] for w in weather]) ** 0.5
    + rng.normal(0, 0.6, size=N).clip(-1, None),
    2
)

df = pd.DataFrame({
    "delivery_id": np.arange(1, N + 1),
    "region": region,
    "distance_km": np.round(distance_km, 2),
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "weather": weather,
    "traffic_index": np.round(traffic_index, 3),
    "shipment_volume": shipment_volume,
    "weight_kg": weight_kg,
    "transportation_cost": transportation_cost,
    "actual_delivery_minutes": np.round(delivery_minutes, 1),
    "promised_window_minutes": promised_window_min,
    "delay_flag": delay_flag,
})

df.to_csv("../data/deliveries_simulated.csv", index=False)
print(df.shape)
print(df.head(10).to_string())
print("\nDelay rate:", df["delay_flag"].mean())
