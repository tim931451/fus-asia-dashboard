import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


CSV_PATH = "engineered_features_daily.csv"
WEEKDAY_NAMES = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}


def time_series_split_75_25(df: pd.DataFrame):
    df = df.sort_values("date")
    split_idx = int(np.floor(len(df) * 0.75))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def fit_xgb(X_train, y_train, X_test, y_test):
    model = XGBRegressor(
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
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model


def metrics(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.maximum(y_true.to_numpy(), 1)
    mape = float(np.mean(np.abs((y_true.to_numpy() - y_pred) / denom)))
    return mae, rmse, mape


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # weekday neu berechnen (nicht aus wd_* ableiten)
    df["weekday"] = df["date"].dt.weekday

    # Target
    df["orders_cnt"] = pd.to_numeric(df["orders_cnt"], errors="coerce").fillna(0)

    # Feature-Matrix: alles außer target/date/weekday
    X_all = df.drop(columns=["orders_cnt", "date", "weekday"], errors="ignore")

    # Tipp: Wenn du wd_* one-hot Spalten noch drin hast, sind die pro Subset konstant.
    # Wir entfernen sie, damit sie keinen Müll in den Modellen machen.
    wd_cols = [c for c in X_all.columns if c.startswith("wd_")]
    if wd_cols:
        X_all = X_all.drop(columns=wd_cols)

    # Check is_rain vorhanden?
    has_rain = "is_rain" in X_all.columns

    results = []

    for wd in range(7):
        # Subset for this weekday
        sub = df[df["weekday"] == wd].copy()

        # Keep original indices to slice X_all, then reset indices so they align with the split (0..n-1)
        orig_idx = sub.index
        sub = sub.sort_values("date").reset_index(drop=True)
        X_sub = X_all.loc[orig_idx].copy().reset_index(drop=True)
        y_sub = sub["orders_cnt"].copy()

        if len(sub) < 60:
            print(f"\n[{WEEKDAY_NAMES[wd]}] zu wenig Daten ({len(sub)} Zeilen) – übersprungen.")
            continue

        train_df, test_df = time_series_split_75_25(sub)
        train_idx = train_df.index
        test_idx = test_df.index

        X_train = X_sub.loc[train_idx]
        y_train = y_sub.loc[train_idx]
        X_test = X_sub.loc[test_idx]
        y_test = y_sub.loc[test_idx]

        model = fit_xgb(X_train, y_train, X_test, y_test)
        pred = model.predict(X_test)

        mae, rmse, mape = metrics(y_test, pred)

        print(f"\n=== {WEEKDAY_NAMES[wd]} (n={len(sub)}) ===")
        print("Train:", train_df['date'].min().date(), "->", train_df['date'].max().date(), f"| n={len(train_df)}")
        print("Test :", test_df['date'].min().date(), "->", test_df['date'].max().date(), f"| n={len(test_df)}")
        print("MAE :", round(mae, 3))
        print("RMSE:", round(rmse, 3))
        print("MAPE:", round(mape, 4))

        # Rain counterfactual innerhalb dieses Wochentags
        rain_mean = rain_median = rain_p05 = rain_p95 = None
        if has_rain:
            X0 = X_test.copy()
            X1 = X_test.copy()
            X0["is_rain"] = 0
            X1["is_rain"] = 1

            pred0 = np.clip(model.predict(X0), 0, None)
            pred1 = np.clip(model.predict(X1), 0, None)
            eff = pred1 - pred0

            rain_mean = float(np.mean(eff))
            rain_median = float(np.median(eff))
            rain_p05 = float(np.quantile(eff, 0.05))
            rain_p95 = float(np.quantile(eff, 0.95))

            print("Rain effect (pred rain=1 - rain=0):",
                  "mean", round(rain_mean, 3),
                  "| median", round(rain_median, 3),
                  "| 5/95", round(rain_p05, 3), "/", round(rain_p95, 3))

        # Top features
        imp = (
            pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False)
            .head(10)
        )
        print("Top features:")
        print(imp.to_string(index=False))

        results.append({
            "weekday": wd,
            "weekday_name": WEEKDAY_NAMES[wd],
            "n": len(sub),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "rain_effect_mean": rain_mean,
            "rain_effect_median": rain_median,
            "rain_effect_p05": rain_p05,
            "rain_effect_p95": rain_p95
        })

    # Summary table
    if results:
        summary = pd.DataFrame(results).sort_values("weekday")
        print("\n\n===== SUMMARY (per weekday) =====")
        print(summary[["weekday_name", "n", "mae", "rmse", "mape", "rain_effect_mean", "rain_effect_median"]].to_string(index=False))


if __name__ == "__main__":
    main()