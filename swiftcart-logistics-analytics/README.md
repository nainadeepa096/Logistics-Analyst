# SwiftCart Logistics Analytics

Data simulation, exploratory analysis, and predictive modeling for last-mile
delivery optimization, built for a mid-sized e-commerce retailer ("SwiftCart").

Supports the Week 3 (Advanced Data Analysis and Visualization) and Week 4
(Predictive Modeling and Optimization) reports.

## Project structure

```
swiftcart-logistics-analytics/
├── src/
│   ├── generate_data.py     # Simulates the 400-record delivery dataset
│   ├── eda_and_charts.py    # Exploratory data analysis + 6 visualizations (Week 3)
│   └── modeling.py          # Regression models + evaluation + 3 charts (Week 4)
├── data/                    # Generated dataset and computed stats (created by scripts)
├── charts/                  # Generated PNG charts (created by scripts)
└── requirements.txt
```

## What each script does

- **`generate_data.py`** — builds a synthetic but realistically-structured dataset
  of 400 deliveries (distance, traffic, weather, shipment volume, cost, delivery
  time, delay flag) and writes it to `data/deliveries_simulated.csv`.
- **`eda_and_charts.py`** — computes summary statistics and a correlation matrix,
  and produces 6 charts: delivery-time histogram, cost-by-region boxplot,
  distance-vs-time scatter, correlation heatmap, delay-rate-by-day bar chart,
  and delivery-time-by-weather bar chart.
- **`modeling.py`** — trains and evaluates a Linear Regression baseline and a
  tuned Random Forest Regressor (via `GridSearchCV` + 5-fold cross-validation)
  to predict `actual_delivery_minutes`, and produces predicted-vs-actual,
  feature-importance, and model-comparison charts.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
cd src
python generate_data.py     # 1. creates data/deliveries_simulated.csv
python eda_and_charts.py    # 2. creates data/*.csv, data/eda_facts.txt, charts/01-06
python modeling.py          # 3. creates data/model_facts.txt, charts/07-09
```

## Key results

- Delay rate: 29.0% of deliveries exceed their promised window.
- Distance is the dominant driver of delivery time (r = 0.85) and cost.
- Linear Regression: MAE = 3.94 min, RMSE = 4.98 min, R² = 0.907 (test set).
- Tuned Random Forest: MAE = 4.79 min, RMSE = 6.08 min, R² = 0.861 (test set).

Full narrative writeups, chart interpretations, and optimization
recommendations are in the accompanying Week 3 and Week 4 Word reports.
