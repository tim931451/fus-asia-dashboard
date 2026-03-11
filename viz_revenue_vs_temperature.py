import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("weather_joined_remote_daily.csv")

df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")
df["orders_value_sum"] = pd.to_numeric(df.get("orders_value_sum"), errors="coerce")
df["temperature_2m_mean"] = pd.to_numeric(df.get("temperature_2m_mean"), errors="coerce")

# Filter: gültige Werte + Umsatz > 0
df = df.dropna(subset=["weather_date"])
df = df[df["orders_value_sum"].notna() & (df["orders_value_sum"] > 0)]
df = df[df["temperature_2m_mean"].notna()]

x = df["temperature_2m_mean"].to_numpy()
y = df["orders_value_sum"].to_numpy()

plt.figure()
plt.scatter(x, y, alpha=0.4)

# Trendlinie (lineare Regression)
if len(x) >= 2:
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = m * x_line + b
    plt.plot(x_line, y_line, linewidth=2, label="Trend (linear)")
    plt.legend()

plt.title("Umsatz vs. Durchschnittstemperatur (Tageswerte)")
plt.xlabel("Temperatur 2m Mittel (°C)")
plt.ylabel("Umsatz pro Tag (Summe Bestellwert)")
plt.tight_layout()
plt.show()