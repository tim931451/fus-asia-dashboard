import pandas as pd
import matplotlib.pyplot as plt

# Load joined dataset (weather + remote aggregates)
df = pd.read_csv("weather_joined_remote_daily.csv")

# Parse date and ensure order count is numeric
df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")
df["orders_value_sum"] = pd.to_numeric(df.get("orders_value_sum"), errors="coerce").fillna(0)
df["orders_cnt"] = pd.to_numeric(df.get("orders_cnt"), errors="coerce")

# Treat missing order counts as 0 (days with no orders)
df["orders_cnt"] = df["orders_cnt"].fillna(0)

# Drop rows without a valid date
df = df.dropna(subset=["weather_date"])

# Ignore days with zero (or missing) order value sum
df = df[df["orders_value_sum"] != 0]

# Sort by date
df = df.sort_values("weather_date")

# Monthly aggregation (sum of daily order counts) + diagnostic info
max_date = df["weather_date"].max()

monthly = (
    df.set_index("weather_date")
      .resample("MS")
      .agg(
          orders_cnt=("orders_cnt", "sum"),
          contributing_days=("orders_cnt", "size"),
      )
      .reset_index()
)

# Days in each calendar month (for context)
monthly["days_in_month"] = monthly["weather_date"].dt.days_in_month

# Mark last (incomplete) month in the dataset
monthly["is_incomplete_month"] = monthly["weather_date"].dt.to_period("M") == max_date.to_period("M")

# Optional: drop the incomplete last month so the chart doesn't 'crash' at the end
monthly_plot = monthly[~monthly["is_incomplete_month"]].copy()

print("Max date in filtered daily data:", max_date)
print(monthly.tail(6).to_string(index=False))

plt.figure()

# Quarter separators (background)
if not monthly_plot.empty:
    q_start = monthly_plot["weather_date"].min().to_period("Q").start_time
    q_end = monthly_plot["weather_date"].max().to_period("Q").end_time
    for q in pd.date_range(start=q_start, end=q_end, freq="QS"):
        plt.axvline(q, linestyle="--", linewidth=1, alpha=0.25)

# Line chart for monthly order counts
plt.plot(monthly_plot["weather_date"], monthly_plot["orders_cnt"], label="Bestellungen pro Monat")

plt.title("Monatliche Bestellentwicklung")
plt.xlabel("Monat")
plt.ylabel("Anzahl Bestellungen")
plt.legend()
plt.tight_layout()
plt.show()