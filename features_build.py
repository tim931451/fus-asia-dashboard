import pandas as pd
import numpy as np
from holidays import country_holidays
from typing import Set


CSV_PATH = "weather_joined_remote_daily.csv"

# Falls du Ferientage hast: Datei mit einer Spalte "date" (YYYY-MM-DD)
SCHOOL_HOLIDAYS_CSV = "school_holidays_bs.csv"  # optional


def load_school_holidays(path: str) -> Set[pd.Timestamp]:
    """
    Optional: lädt Schulferien/Feiertage aus CSV.
    Erwartet eine Spalte: date (YYYY-MM-DD)
    """
    try:
        h = pd.read_csv(path)
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        h = h.dropna(subset=["date"])
        return set(h["date"].dt.normalize())
    except FileNotFoundError:
        return set()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # --- Basic parsing ---
    df = df.copy()

    df["date"] = pd.to_datetime(df["weather_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date")

    # Target
    df["orders_cnt"] = pd.to_numeric(df.get("orders_cnt"), errors="coerce")

    # Umsatz (für Filter/Debug; Target bleibt orders_cnt)
    df["orders_value_sum"] = pd.to_numeric(df.get("orders_value_sum"), errors="coerce")

    # rain_sum aus precipitation_sum ableiten, falls nicht vorhanden
    if "rain_sum" not in df.columns and "precipitation_sum" in df.columns:
        df["rain_sum"] = df["precipitation_sum"]

    # Wetter numerisch
    weather_cols = [
        "temperature_2m_mean",
        "rain_sum",
        "windspeed_10m_max",
        "weathercode",
    ]
    for c in weather_cols:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")

    # --- Cleaning / Interpretation ---
    # Für orders_cnt: wenn missing, ist das bei euch oft "kein Remote Match" oder "0".
    # Für Modellierung nehmen wir konservativ 0, aber du kannst später auch missing separat behandeln.
    df["orders_cnt"] = df["orders_cnt"].fillna(0)

    # Wetter fehlend: füllen wir nicht aggressiv; wir lassen NaN und können später imputen.
    # (Viele Modelle können NaN nicht -> dann imputen wir gezielt.)

    # --- Calendar features ---
    df["weekday"] = df["date"].dt.weekday  # 0=Mo ... 6=So
    # Removed is_weekend, month, day_of_month, week_of_year per instructions

    # Saisonalität (als einfache, glatte Features)
    df["day_of_year"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)

    # One-hot für weekday (für lineare Modelle nützlich)
    weekday_dummies = pd.get_dummies(df["weekday"], prefix="wd", drop_first=False)
    df = pd.concat([df, weekday_dummies], axis=1)

    # --- Public holidays (Switzerland) ---
    years = sorted(df["date"].dt.year.unique().tolist())
    ch_holidays = country_holidays("CH", years=years)  # nationale Feiertage

    df["is_public_holiday"] = df["date"].isin(ch_holidays).astype(int)

    # Removed is_day_before_holiday and is_day_after_holiday per instructions

    # --- School holidays / Ferientage (Basel) via CSV plug-in ---
    school_holidays = load_school_holidays(SCHOOL_HOLIDAYS_CSV)
    df["is_school_holiday"] = df["date"].isin(school_holidays).astype(int)

    # Removed is_day_before_school_holiday and is_day_after_school_holiday per instructions

    # --- Rain feature (binary) ---
    # is_rain = 1 if daily rain_sum >= 2mm, else 0
    rain_sum_filled = df["rain_sum"].fillna(0)
    df["is_rain"] = (rain_sum_filled > 0).astype(int)

    # Temperatur kann auch in Bins helfen, aber erstmal lassen wir sie numerisch.

    # --- Lag features (OHNE Leakage) ---
    # Wichtig: Lags/rollings immer mit shift(1), damit “heute” nicht “heute” benutzt.
    # Removed lag_1, rolling_mean_28, rolling_std_7, rolling_std_28 and fillna lines per instructions
    df["lag_7"] = df["orders_cnt"].shift(7)

    df["rolling_mean_7"] = df["orders_cnt"].shift(1).rolling(7, min_periods=1).mean()

    # --- Select features ---
    feature_cols = [
        # Wetter (reduced)
        "temperature_2m_mean",
        "windspeed_10m_max",
        "is_rain",
        "weathercode",

        # Seasonality / calendar
        "sin_doy",
        "cos_doy",
        "is_public_holiday",
        "is_school_holiday",

        # Lags / rolling (reduced)
        "lag_7",
        "rolling_mean_7",
    ]

    # weekday dummies dynamisch ergänzen
    feature_cols += [c for c in df.columns if c.startswith("wd_")]

    # Finale Tabelle
    out = df[["date", "orders_cnt"] + feature_cols].copy()

    return out


def time_series_split_75_25(features: pd.DataFrame):
    """
    Time-based split: first 75% dates train, last 25% test.
    """
    features = features.sort_values("date").reset_index(drop=True)
    n = len(features)
    split_idx = int(np.floor(n * 0.75))

    train = features.iloc[:split_idx].copy()
    test = features.iloc[split_idx:].copy()

    X_train = train.drop(columns=["orders_cnt"])
    y_train = train["orders_cnt"]

    X_test = test.drop(columns=["orders_cnt"])
    y_test = test["orders_cnt"]

    return X_train, X_test, y_train, y_test, train, test


def main():
    df = pd.read_csv(CSV_PATH)
    feats = build_features(df)

    X_train, X_test, y_train, y_test, train, test = time_series_split_75_25(feats)

    print("Rows total:", len(feats))
    print("Train rows:", len(train), "| Test rows:", len(test))
    print("Train date range:", train["date"].min(), "->", train["date"].max())
    print("Test  date range:", test["date"].min(), "->", test["date"].max())

    # Optional: save engineered dataset (read-only output file)
    feats.to_csv("engineered_features_daily.csv", index=False)
    print("Saved: engineered_features_daily.csv")


if __name__ == "__main__":
    main()