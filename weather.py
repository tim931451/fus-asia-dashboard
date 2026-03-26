# weather.py
# Usage:
#   python weather.py                  # lädt Wetterdaten von Open-Meteo API (Basel, ab 2023)
#   python weather.py input.json       # konvertiert lokale JSON-Datei
#
# Schreibt weather_basel_daily.csv für die restliche Pipeline.
import json
import ssl
import sys
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd


WEATHER_CSV = "weather_basel_daily.csv"

# Basel Koordinaten
LATITUDE = 47.557117
LONGITUDE = 7.549342


def fetch_from_api() -> pd.DataFrame:
    """Holt tägliche Wetterdaten für Basel von der Open-Meteo Archive API + Forecast API."""
    # Archive API: bis vorgestern (sicherer Bereich)
    end_archive = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
    params_archive = (
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&start_date=2023-01-01&end_date={end_archive}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,rain_sum,windspeed_10m_max,weathercode"
        f"&timezone=Europe%2FZurich"
    )
    url_archive = f"https://archive-api.open-meteo.com/v1/archive?{params_archive}"

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    req = Request(url_archive, headers={"User-Agent": "weather-pipeline/1.0"})
    with urlopen(req, timeout=30, context=ctx) as resp:
        data_archive = json.loads(resp.read().decode("utf-8"))

    df_archive = _daily_to_df(data_archive)

    # Forecast API: letzte Tage + heute (past_days=7 gibt vergangene Tage)
    params_forecast = (
        f"latitude={LATITUDE}&longitude={LONGITUDE}"
        f"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
        f"precipitation_sum,rain_sum,windspeed_10m_max,weathercode"
        f"&timezone=Europe%2FZurich"
        f"&past_days=7&forecast_days=1"
    )
    url_forecast = f"https://api.open-meteo.com/v1/forecast?{params_forecast}"

    try:
        req2 = Request(url_forecast, headers={"User-Agent": "weather-pipeline/1.0"})
        with urlopen(req2, timeout=30, context=ctx) as resp2:
            data_forecast = json.loads(resp2.read().decode("utf-8"))
        df_forecast = _daily_to_df(data_forecast)

        # Kombinieren: Archive + Forecast (ohne Duplikate)
        df = pd.concat([df_archive, df_forecast], ignore_index=True)
        df = df.drop_duplicates(subset=["weather_date"], keep="last").sort_values("weather_date").reset_index(drop=True)
    except Exception:
        df = df_archive

    return df


def json_to_dataframe(json_path: Path) -> pd.DataFrame:
    """Liest eine lokale Open-Meteo JSON-Datei."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    return _daily_to_df(data)


def _daily_to_df(data: dict) -> pd.DataFrame:
    daily = data.get("daily")
    if not isinstance(daily, dict):
        raise ValueError('JSON muss einen "daily"-Key mit einem Objekt enthalten.')
    if "time" not in daily:
        raise ValueError('daily.time ist erforderlich.')

    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "weather_date"})
    df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")

    preferred = [
        "weather_date", "temperature_2m_max", "temperature_2m_min",
        "temperature_2m_mean", "precipitation_sum", "rain_sum",
        "windspeed_10m_max", "weathercode",
    ]
    cols = [c for c in preferred if c in df.columns]
    return df[cols]


def main():
    out_path = Path(WEATHER_CSV)

    if len(sys.argv) >= 2:
        # Lokale JSON-Datei konvertieren
        in_path = Path(sys.argv[1])
        if len(sys.argv) >= 3:
            out_path = Path(sys.argv[2])
        df = json_to_dataframe(in_path)
    else:
        # Von Open-Meteo API laden
        print("Lade Wetterdaten von Open-Meteo API (Basel)...")
        df = fetch_from_api()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{len(df)} Zeilen geschrieben nach {out_path}")


if __name__ == "__main__":
    main()
