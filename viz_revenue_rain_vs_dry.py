import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("weather_joined_remote_daily.csv")

df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")
df["orders_value_sum"] = pd.to_numeric(df.get("orders_value_sum"), errors="coerce")
df["rain_sum"] = pd.to_numeric(df.get("rain_sum"), errors="coerce")
df["precipitation_sum"] = pd.to_numeric(df.get("precipitation_sum"), errors="coerce")

# Filter: gültiges Datum + Umsatz > 0
df = df.dropna(subset=["weather_date"])
df = df[df["orders_value_sum"].notna() & (df["orders_value_sum"] > 0)]

# Regentag definieren (bevorzugt rain_sum, fallback precipitation_sum)
rain_metric = df["rain_sum"].copy()
rain_metric = rain_metric.where(rain_metric.notna(), df["precipitation_sum"])
rain_metric = rain_metric.fillna(0)

df["is_rain_day"] = rain_metric > 0

rainy = df.loc[df["is_rain_day"], "orders_value_sum"].to_list()
dry = df.loc[~df["is_rain_day"], "orders_value_sum"].to_list()

plt.figure()
plt.boxplot([dry, rainy], labels=["Trocken", "Regen"])
plt.title("Umsatzverteilung: Trocken vs. Regen (Tageswerte)")
plt.xlabel("Wettertyp")
plt.ylabel("Umsatz pro Tag (Summe Bestellwert)")
plt.tight_layout()
plt.show()

print("Anzahl Tage (Trocken):", len(dry))
print("Anzahl Tage (Regen):", len(rainy))