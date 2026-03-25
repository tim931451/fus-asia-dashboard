"""
Trainiert XGBoost-Modelle und speichert sie als Pickle-Dateien.
  - Globales Modell (count:poisson)
  - 7 Wochentag-Modelle (reg:squarederror)

Ausführen:  python train_and_save.py
"""

import json
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

CSV_PATH = "engineered_features_daily.csv"
MODEL_DIR = "models"
WEEKDAY_NAMES = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}


def time_series_split_75_25(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(np.floor(len(df) * 0.75))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["orders_cnt"] = pd.to_numeric(df["orders_cnt"], errors="coerce").fillna(0)

    # --- Globales Modell ---
    y = df["orders_cnt"]
    X = df.drop(columns=["orders_cnt", "date"], errors="ignore")
    feature_cols_global = list(X.columns)

    train_df, test_df = time_series_split_75_25(df)
    train_idx, test_idx = train_df.index, test_df.index

    X_train, y_train = X.loc[train_idx], y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], y.loc[test_idx]

    model_global = XGBRegressor(
        objective="count:poisson",
        n_estimators=2500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=42,
        n_jobs=-1,
    )
    model_global.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    pred = np.clip(model_global.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred, squared=False)
    print(f"Global model  -> MAE: {mae:.3f}, RMSE: {rmse:.3f}")

    joblib.dump(model_global, f"{MODEL_DIR}/xgb_global.pkl")

    # --- Wochentag-Modelle ---
    df["weekday"] = df["date"].dt.weekday
    X_all_wd = df.drop(columns=["orders_cnt", "date", "weekday"], errors="ignore")
    wd_cols = [c for c in X_all_wd.columns if c.startswith("wd_")]
    if wd_cols:
        X_all_wd = X_all_wd.drop(columns=wd_cols)
    feature_cols_weekday = list(X_all_wd.columns)

    for wd in range(7):
        sub = df[df["weekday"] == wd].copy()
        orig_idx = sub.index
        sub = sub.sort_values("date").reset_index(drop=True)
        X_sub = X_all_wd.loc[orig_idx].copy().reset_index(drop=True)
        y_sub = sub["orders_cnt"]

        if len(sub) < 60:
            print(f"  {WEEKDAY_NAMES[wd]}: zu wenig Daten, uebersprungen")
            continue

        train_sub, test_sub = time_series_split_75_25(sub)
        ti, te = train_sub.index, test_sub.index

        model_wd = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            reg_alpha=0.5,
            min_child_weight=5.0,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
        )
        model_wd.fit(X_sub.loc[ti], y_sub.loc[ti],
                      eval_set=[(X_sub.loc[te], y_sub.loc[te])], verbose=False)

        p = np.clip(model_wd.predict(X_sub.loc[te]), 0, None)
        m = mean_absolute_error(y_sub.loc[te], p)
        print(f"  {WEEKDAY_NAMES[wd]} model -> MAE: {m:.3f}")

        joblib.dump(model_wd, f"{MODEL_DIR}/xgb_weekday_{wd}.pkl")

    # Feature-Spalten speichern
    meta = {
        "feature_cols_global": feature_cols_global,
        "feature_cols_weekday": feature_cols_weekday,
    }
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModelle gespeichert in {MODEL_DIR}/")


if __name__ == "__main__":
    main()
