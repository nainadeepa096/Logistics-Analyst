import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white", "axes.facecolor": "white",
})
NAVY = "#1F3864"; ACCENT = "#2E5C8A"

df = pd.read_csv("../data/deliveries_simulated.csv")

feature_cols_num = ["distance_km", "traffic_index", "shipment_volume", "weight_kg", "hour_of_day"]
feature_cols_cat = ["weather", "region"]
target = "actual_delivery_minutes"

X = df[feature_cols_num + feature_cols_cat]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), feature_cols_cat)
], remainder="passthrough")

# ---------------- Linear Regression ----------------
lin_pipe = Pipeline([("pre", pre), ("model", LinearRegression())])
lin_pipe.fit(X_train, y_train)
lin_pred = lin_pipe.predict(X_test)

lin_mae = mean_absolute_error(y_test, lin_pred)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_pred))
lin_r2 = r2_score(y_test, lin_pred)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
lin_cv = -cross_val_score(lin_pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")

# ---------------- Random Forest (with small grid search) ----------------
rf_pipe = Pipeline([("pre", pre), ("model", RandomForestRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 8, None],
}
grid = GridSearchCV(rf_pipe, param_grid, cv=kf, scoring="neg_mean_absolute_error", n_jobs=1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
rf_pred = best_rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
rf_cv = -cross_val_score(best_rf, X, y, cv=kf, scoring="neg_mean_absolute_error")

print("=== Linear Regression ===")
print(f"MAE={lin_mae:.2f} RMSE={lin_rmse:.2f} R2={lin_r2:.3f}")
print(f"5-fold CV MAE: {lin_cv.mean():.2f} (+/- {lin_cv.std():.2f})")

print("\n=== Random Forest (tuned) ===")
print("Best params:", grid.best_params_)
print(f"MAE={rf_mae:.2f} RMSE={rf_rmse:.2f} R2={rf_r2:.3f}")
print(f"5-fold CV MAE: {rf_cv.mean():.2f} (+/- {rf_cv.std():.2f})")

# Coefficients for linear model (for interpretability discussion)
ohe = lin_pipe.named_steps["pre"].named_transformers_["cat"]
cat_names = list(ohe.get_feature_names_out(feature_cols_cat))
all_feature_names = cat_names + feature_cols_num
coefs = lin_pipe.named_steps["model"].coef_
coef_series = pd.Series(coefs, index=all_feature_names).sort_values(key=abs, ascending=False)
print("\nLinear coefficients (minutes per unit):")
print(coef_series)

# Feature importances for RF
rf_model = best_rf.named_steps["model"]
importances = pd.Series(rf_model.feature_importances_, index=all_feature_names).sort_values(ascending=False)
print("\nRF feature importances:")
print(importances)

with open("../data/model_facts.txt", "w") as f:
    f.write(f"n_train={len(X_train)}\nn_test={len(X_test)}\n")
    f.write(f"lin_mae={lin_mae:.2f}\nlin_rmse={lin_rmse:.2f}\nlin_r2={lin_r2:.3f}\n")
    f.write(f"lin_cv_mae_mean={lin_cv.mean():.2f}\nlin_cv_mae_std={lin_cv.std():.2f}\n")
    f.write(f"rf_mae={rf_mae:.2f}\nrf_rmse={rf_rmse:.2f}\nrf_r2={rf_r2:.3f}\n")
    f.write(f"rf_cv_mae_mean={rf_cv.mean():.2f}\nrf_cv_mae_std={rf_cv.std():.2f}\n")
    f.write(f"rf_best_params={grid.best_params_}\n")
    f.write(f"top_coef_1={coef_series.index[0]}={coef_series.iloc[0]:.2f}\n")
    f.write(f"top_coef_2={coef_series.index[1]}={coef_series.iloc[1]:.2f}\n")
    f.write(f"top_coef_3={coef_series.index[2]}={coef_series.iloc[2]:.2f}\n")
    f.write(f"top_imp_1={importances.index[0]}={importances.iloc[0]:.3f}\n")
    f.write(f"top_imp_2={importances.index[1]}={importances.iloc[1]:.3f}\n")
    f.write(f"top_imp_3={importances.index[2]}={importances.iloc[2]:.3f}\n")

# ---------------- Chart: predicted vs actual (RF) ----------------
fig, ax = plt.subplots(figsize=(6.6, 5))
ax.scatter(y_test, rf_pred, alpha=0.7, s=26, color=ACCENT, edgecolor="#333333", linewidth=0.2)
lims = [min(y_test.min(), rf_pred.min()) - 2, max(y_test.max(), rf_pred.max()) + 2]
ax.plot(lims, lims, color="#C0392B", linestyle="--", linewidth=1.6, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual delivery time (minutes)")
ax.set_ylabel("Predicted delivery time (minutes)")
ax.set_title(f"Random Forest: Predicted vs. Actual (Test Set, R² = {rf_r2:.2f})")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../charts/07_pred_vs_actual.png", dpi=170)
plt.close(fig)

# ---------------- Chart: feature importance (RF) ----------------
fig, ax = plt.subplots(figsize=(6.6, 4.6))
imp_sorted = importances.sort_values()
colors = ["#1F3864" if v == imp_sorted.max() else "#8FB8D6" for v in imp_sorted.values]
ax.barh(imp_sorted.index, imp_sorted.values, color=colors)
ax.set_xlabel("Relative importance")
ax.set_title("Random Forest Feature Importance — Predicting Delivery Time")
fig.tight_layout()
fig.savefig("../charts/08_feature_importance.png", dpi=170)
plt.close(fig)

# ---------------- Chart: MAE comparison bar ----------------
fig, ax = plt.subplots(figsize=(5.6, 4))
models = ["Linear\nRegression", "Random Forest\n(tuned)"]
maes = [lin_mae, rf_mae]
bars = ax.bar(models, maes, color=[ACCENT, NAVY], width=0.5)
for b, v in zip(bars, maes):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=10)
ax.set_ylabel("Test-set MAE (minutes)")
ax.set_title("Model Comparison — Mean Absolute Error")
fig.tight_layout()
fig.savefig("../charts/09_model_mae_compare.png", dpi=170)
plt.close(fig)

print("\nDone.")
