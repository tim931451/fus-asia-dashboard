import pandas as pd

df = pd.read_csv("engineered_features_daily.csv")
df["date"] = pd.to_datetime(df["date"])
df["weekday"] = df["date"].dt.weekday

stats = df.groupby("weekday")["orders_cnt"].describe()
print(stats)

print("\nTop 10 Sonntage nach orders_cnt:")
print(df[df["weekday"] == 6].sort_values("orders_cnt", ascending=False).head(10)[["date","orders_cnt","is_public_holiday","is_school_holiday","is_rain"]])