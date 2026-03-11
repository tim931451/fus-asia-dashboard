import pandas as pd
from sqlalchemy import text
from db import LOCAL, REMOTE  # so wie wir es vorher aufgebaut haben


REMOTE_TABLE = "vu_fuboxeat_ab_2021"


def show_remote_columns():
    cols = pd.read_sql(text(f"SHOW COLUMNS FROM {REMOTE_TABLE};"), REMOTE)
    print("Remote-Spalten:")
    print(cols[["Field", "Type"]].to_string(index=False))
    return cols


def pick_date_column(cols_df):
    # Heuristik: typische Namen
    candidates = []
    for _, r in cols_df.iterrows():
        name = r["Field"].lower()
        typ = str(r["Type"]).lower()
        if ("date" in name or "datum" in name or "time" in name or "timestamp" in name) and (
            "date" in typ or "time" in typ or "timestamp" in typ or "datetime" in typ
        ):
            candidates.append(r["Field"])

    if not candidates:
        # fallback: any DATE/DATETIME/TIMESTAMP
        for _, r in cols_df.iterrows():
            typ = str(r["Type"]).lower()
            if "date" in typ or "time" in typ or "timestamp" in typ or "datetime" in typ:
                candidates.append(r["Field"])

    if not candidates:
        raise RuntimeError("Keine DATE/DATETIME/TIMESTAMP-Spalte gefunden. Bitte Datumsspalte manuell angeben.")

    # nimm die erste plausible
    return candidates[0]


def load_weather_local():
    weather = pd.read_sql(
        text("""
            SELECT
                weather_date,
                temperature_2m_mean,
                temperature_2m_max,
                temperature_2m_min,
                precipitation_sum,
                windspeed_10m_max,
                weathercode
            FROM weather_basel_daily
            WHERE weather_date >= '2023-01-01'
            ORDER BY weather_date
        """),
        LOCAL
    )
    weather["weather_date"] = pd.to_datetime(weather["weather_date"])
    return weather


def load_remote_daily(date_col):
    daily = pd.read_sql(
        text(f"""
            SELECT
                DATE({date_col}) AS business_date,
                COUNT(*) AS orders_cnt,
                COALESCE(SUM(gesamtbetrag), 0) AS orders_value_sum
            FROM {REMOTE_TABLE}
            WHERE {date_col} >= '2023-01-01'
            GROUP BY DATE({date_col})
            ORDER BY business_date
        """),
        REMOTE
    )
    daily["business_date"] = pd.to_datetime(daily["business_date"])
    return daily


def main():
    cols = show_remote_columns()
    date_col = pick_date_column(cols)
    print(f"\nGewählte Datumsspalte für Join/Aggregation: {date_col}")

    weather = load_weather_local()
    remote_daily = load_remote_daily(date_col)

    # Join: Wetter links, Remote rechts (damit alle Wettertage drin bleiben)
    df = weather.merge(remote_daily, left_on="weather_date", right_on="business_date", how="left")
    df.drop(columns=["business_date"], inplace=True)

    print("\nJoined Result (head):")
    print(df.head(10).to_string(index=False))

    print("\nCoverage Check (wie viele Tage hatten Remote-Daten?):")
    print(df["orders_cnt"].notna().value_counts(dropna=False))

    # Optional speichern
    df.to_csv("weather_joined_remote_daily.csv", index=False)
    print("\nGespeichert: weather_joined_remote_daily.csv")


if __name__ == "__main__":
    main()