import pandas as pd
import matplotlib.pyplot as plt

# Load joined dataset (weather + remote aggregates)
df = pd.read_csv("weather_joined_remote_daily.csv")

# Parse date and ensure revenue is numeric
df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")
df["orders_value_sum"] = pd.to_numeric(df.get("orders_value_sum"), errors="coerce")

# Drop rows without a valid date
df = df.dropna(subset=["weather_date"])

# Keep only days where revenue is present and > 0
# (ignores zero revenue days as you requested)
df = df[df["orders_value_sum"].notna() & (df["orders_value_sum"] > 0)]

# Sort by date
df = df.sort_values("weather_date")

# Monthly aggregation (sum of daily revenue) + diagnostic info
max_date = df["weather_date"].max()

monthly = (
    df.set_index("weather_date")
      .resample("MS")
      .agg(
          revenue_sum=("orders_value_sum", "sum"),
          contributing_days=("orders_value_sum", "size"),
      )
      .reset_index()
)

monthly["days_in_month"] = monthly["weather_date"].dt.days_in_month
monthly["is_incomplete_month"] = monthly["weather_date"].dt.to_period("M") == max_date.to_period("M")

# Drop last incomplete month (optional, keeps chart readable)
monthly_plot = monthly[~monthly["is_incomplete_month"]].copy()

print("Max date in filtered daily data:", max_date)
print(monthly.tail(24).to_string(index=False))

plt.figure()

# Quarter separators (background)
if not monthly_plot.empty:
    q_start = monthly_plot["weather_date"].min().to_period("Q").start_time
    q_end = monthly_plot["weather_date"].max().to_period("Q").end_time
    for q in pd.date_range(start=q_start, end=q_end, freq="QS"):
        plt.axvline(q, linestyle="--", linewidth=1, alpha=0.25)

# Line chart for monthly revenue
plt.plot(monthly_plot["weather_date"], monthly_plot["revenue_sum"], label="Umsatz pro Monat (Summe)")

plt.title("Monatliche Umsatzentwicklung")
plt.xlabel("Monat")
plt.ylabel("Umsatz (Summe Bestellwert)")
plt.legend()
plt.tight_layout()
plt.show()