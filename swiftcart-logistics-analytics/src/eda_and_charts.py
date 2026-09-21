import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
NAVY = "#1F3864"
ACCENT = "#2E5C8A"
PALETTE = ["#1F3864", "#2E5C8A", "#5B8FB9", "#8FB8D6", "#C9DCEA"]

df = pd.read_csv("../data/deliveries_simulated.csv")

# ---------- Central tendency / dispersion summary ----------
num_cols = ["distance_km", "traffic_index", "shipment_volume", "weight_kg",
            "transportation_cost", "actual_delivery_minutes"]
summary = df[num_cols].agg(["mean", "median", "std", "min", "max"]).T
summary.to_csv("../data/summary_stats.csv")
print(summary)

# ---------- Correlation matrix ----------
corr = df[num_cols + ["delay_flag"]].corr(numeric_only=True)
corr.to_csv("../data/correlation_matrix.csv")
print(corr["actual_delivery_minutes"].sort_values(ascending=False))

# ---------- Group summaries ----------
by_region_cost = df.groupby("region")["transportation_cost"].mean().sort_values(ascending=False)
by_day_delay = df.groupby("day_of_week")["delay_flag"].mean()
by_weather_delay = df.groupby("weather")["actual_delivery_minutes"].mean().sort_values(ascending=False)
print(by_region_cost)
print(by_day_delay)
print(by_weather_delay)

with open("../data/eda_facts.txt", "w") as f:
    f.write(f"n_records={len(df)}\n")
    f.write(f"delay_rate={df['delay_flag'].mean():.3f}\n")
    f.write(f"mean_delivery_min={df['actual_delivery_minutes'].mean():.2f}\n")
    f.write(f"median_delivery_min={df['actual_delivery_minutes'].median():.2f}\n")
    f.write(f"std_delivery_min={df['actual_delivery_minutes'].std():.2f}\n")
    f.write(f"mean_cost={df['transportation_cost'].mean():.2f}\n")
    f.write(f"median_cost={df['transportation_cost'].median():.2f}\n")
    f.write(f"std_cost={df['transportation_cost'].std():.2f}\n")
    f.write(f"corr_distance_time={corr.loc['distance_km','actual_delivery_minutes']:.3f}\n")
    f.write(f"corr_traffic_time={corr.loc['traffic_index','actual_delivery_minutes']:.3f}\n")
    f.write(f"corr_volume_time={corr.loc['shipment_volume','actual_delivery_minutes']:.3f}\n")
    f.write(f"corr_traffic_delay={corr.loc['traffic_index','delay_flag']:.3f}\n")
    f.write(f"top_cost_region={by_region_cost.index[0]}\n")
    f.write(f"top_cost_region_val={by_region_cost.iloc[0]:.2f}\n")
    f.write(f"low_cost_region={by_region_cost.index[-1]}\n")
    f.write(f"low_cost_region_val={by_region_cost.iloc[-1]:.2f}\n")
    f.write(f"worst_weather={by_weather_delay.index[0]}\n")
    f.write(f"worst_weather_val={by_weather_delay.iloc[0]:.2f}\n")
    f.write(f"best_weather={by_weather_delay.index[-1]}\n")
    f.write(f"best_weather_val={by_weather_delay.iloc[-1]:.2f}\n")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    worst_day_idx = by_day_delay.idxmax()
    f.write(f"worst_day={days[worst_day_idx]}\n")
    f.write(f"worst_day_val={by_day_delay.max():.3f}\n")

# =====================================================================
# Chart 1: Histogram of delivery times
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 4))
ax.hist(df["actual_delivery_minutes"], bins=24, color=ACCENT, edgecolor="white")
ax.axvline(df["actual_delivery_minutes"].mean(), color="#C0392B", linestyle="--", linewidth=1.6,
           label=f"Mean = {df['actual_delivery_minutes'].mean():.1f} min")
ax.axvline(df["actual_delivery_minutes"].median(), color="#E67E22", linestyle="--", linewidth=1.6,
           label=f"Median = {df['actual_delivery_minutes'].median():.1f} min")
ax.set_xlabel("Actual delivery time (minutes)")
ax.set_ylabel("Number of deliveries")
ax.set_title("Distribution of Delivery Times")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("../charts/01_hist_delivery_time.png", dpi=170)
plt.close(fig)

# =====================================================================
# Chart 2: Boxplot of transportation cost by region
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 4))
order = by_region_cost.index.tolist()
data = [df.loc[df.region == r, "transportation_cost"].values for r in order]
bp = ax.boxplot(data, labels=order, patch_artist=True, medianprops=dict(color="#C0392B", linewidth=1.6))
for patch, color in zip(bp["boxes"], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax.set_ylabel("Transportation cost ($)")
ax.set_title("Transportation Cost by Region")
fig.tight_layout()
fig.savefig("../charts/02_box_cost_by_region.png", dpi=170)
plt.close(fig)

# =====================================================================
# Chart 3: Scatter of distance vs delivery time with trend line
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 4))
sc = ax.scatter(df["distance_km"], df["actual_delivery_minutes"], c=df["traffic_index"],
                 cmap="Blues", s=22, alpha=0.85, edgecolor="#333333", linewidth=0.2)
z = np.polyfit(df["distance_km"], df["actual_delivery_minutes"], 1)
xs = np.linspace(df["distance_km"].min(), df["distance_km"].max(), 100)
ax.plot(xs, np.polyval(z, xs), color="#C0392B", linewidth=1.8, label=f"Trend: y = {z[0]:.2f}x + {z[1]:.1f}")
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label("Traffic index")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Actual delivery time (minutes)")
ax.set_title("Distance vs. Delivery Time (colored by traffic)")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig("../charts/03_scatter_distance_time.png", dpi=170)
plt.close(fig)

# =====================================================================
# Chart 4: Correlation heatmap
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 5.2))
labels = ["Distance", "Traffic idx", "Shipment vol", "Weight", "Cost", "Delivery time", "Delay flag"]
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                 color="white" if abs(corr.values[i, j]) > 0.55 else "black", fontsize=8.5)
ax.set_title("Correlation Matrix — Key Delivery Variables")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
fig.tight_layout()
fig.savefig("../charts/04_corr_heatmap.png", dpi=170)
plt.close(fig)

# =====================================================================
# Chart 5: Bar chart — average delay rate by day of week
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 4))
days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
vals = [by_day_delay.get(i, 0) * 100 for i in range(7)]
bars = ax.bar(days, vals, color=ACCENT, edgecolor="white")
bars[int(by_day_delay.idxmax())].set_color("#C0392B")
ax.set_ylabel("Deliveries delayed (%)")
ax.set_title("On-Time Performance Risk by Day of Week")
ax.yaxis.set_major_formatter(mticker.PercentFormatter())
fig.tight_layout()
fig.savefig("../charts/05_bar_delay_by_day.png", dpi=170)
plt.close(fig)

# =====================================================================
# Chart 6: Average delivery time by weather condition
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 4))
order_w = by_weather_delay.index.tolist()
vals_w = by_weather_delay.values
bars = ax.bar(order_w, vals_w, color=PALETTE[:len(order_w)], edgecolor="white")
ax.set_ylabel("Average delivery time (minutes)")
ax.set_title("Average Delivery Time by Weather Condition")
fig.tight_layout()
fig.savefig("../charts/06_bar_weather_time.png", dpi=170)
plt.close(fig)

print("\nCharts written to ../charts/")
