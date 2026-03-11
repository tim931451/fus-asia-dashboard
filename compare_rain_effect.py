import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

CSV_PATH = "engineered_features_daily.csv"


def time_series_split_75_25(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(np.floor(len(df) * 0.75))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def train_and_eval(X_train, y_train, X_test, y_test, label: str):
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

    y_pred = np.clip(model.predict(X_test), 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    denom = np.maximum(y_test.to_numpy(), 1)
    mape = np.mean(np.abs((y_test.to_numpy() - y_pred) / denom))

    print(f"\n=== {label} ===")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("MAPE (approx):", round(mape, 4))

    return model, y_pred


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Target
    y = pd.to_numeric(df["orders_cnt"], errors="coerce").fillna(0)

    # Features: alles außer date und target
    X = df.drop(columns=["orders_cnt"], errors="ignore")
    X = X.drop(columns=["date"], errors="ignore")

    if "is_rain" not in X.columns:
        raise RuntimeError("Spalte 'is_rain' nicht gefunden. Bitte zuerst features_build.py laufen lassen.")

    # Split
    train_df, test_df = time_series_split_75_25(df)
    train_idx = train_df.index
    test_idx = test_df.index

    X_train_full = X.loc[train_idx].copy()
    y_train = y.loc[train_idx]
    X_test_full = X.loc[test_idx].copy()
    y_test = y.loc[test_idx]

    # --- Modell A: ohne Rain ---
    X_train_no_rain = X_train_full.drop(columns=["is_rain"])
    X_test_no_rain = X_test_full.drop(columns=["is_rain"])
    model_a, pred_a = train_and_eval(X_train_no_rain, y_train, X_test_no_rain, y_test, "Model A (ohne is_rain)")

    # --- Modell B: mit Rain ---
    model_b, pred_b = train_and_eval(X_train_full, y_train, X_test_full, y_test, "Model B (mit is_rain)")

    # --- Direkter Modellvergleich ---
    delta_mae = mean_absolute_error(y_test, pred_b) - mean_absolute_error(y_test, pred_a)
    delta_rmse = mean_squared_error(y_test, pred_b, squared=False) - mean_squared_error(y_test, pred_a, squared=False)
    print("\n=== Vergleich (B - A) ===")
    print("ΔMAE :", round(delta_mae, 3), "(negativ = besser mit Rain)")
    print("ΔRMSE:", round(delta_rmse, 3), "(negativ = besser mit Rain)")

    # --- Modellierter Rain-Effekt (Counterfactual on test set) ---
    # Wir nehmen Modell B und setzen is_rain einmal auf 0 und einmal auf 1.
    X0 = X_test_full.copy()
    X1 = X_test_full.copy()
    X0["is_rain"] = 0
    X1["is_rain"] = 1

    pred0 = np.clip(model_b.predict(X0), 0, None)
    pred1 = np.clip(model_b.predict(X1), 0, None)

    rain_effect = pred1 - pred0  # >0 bedeutet: Modell erwartet mehr Bestellungen bei Regen

    print("\n=== Rain-Effekt (Model B, Testset) ===")
    print("Mean effect:", round(float(np.mean(rain_effect)), 3))
    print("Median effect:", round(float(np.median(rain_effect)), 3))
    print("5% / 95%:", round(float(np.quantile(rain_effect, 0.05)), 3), "/", round(float(np.quantile(rain_effect, 0.95)), 3))

    # Plot 1: Ist vs Prognosen (A und B)
    dates_test = test_df["date"].reset_index(drop=True)
    y_test_plot = y_test.reset_index(drop=True)

    plt.figure()
    plt.plot(dates_test, y_test_plot, label="Ist")
    plt.plot(dates_test, pd.Series(pred_a), label="Prognose A (ohne Rain)")
    plt.plot(dates_test, pd.Series(pred_b), label="Prognose B (mit Rain)")
    plt.title("Ist vs Prognose (Testzeitraum) – Vergleich Rain")
    plt.xlabel("Datum")
    plt.ylabel("Bestellungen")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot 2: Verteilung des Rain-Effekts
    plt.figure()
    plt.hist(rain_effect, bins=40)
    plt.title("Verteilung: modellierter Rain-Effekt (pred(is_rain=1) - pred(is_rain=0))")
    plt.xlabel("Δ Bestellungen (Regen - Kein Regen)")
    plt.ylabel("Häufigkeit")
    plt.tight_layout()
    plt.show()

    # Optional: Effekt getrennt nach tatsächlichem Regenstatus im Test
    actual_is_rain = X_test_full["is_rain"].to_numpy()
    if actual_is_rain.max() <= 1 and actual_is_rain.min() >= 0:
        print("\nEffekt nach tatsächlichem is_rain (Test):")
        for val in [0, 1]:
            eff = rain_effect[actual_is_rain == val]
            if len(eff) > 0:
                print(f"is_rain={val}: mean={np.mean(eff):.3f}, n={len(eff)}")


if __name__ == "__main__":
    main()