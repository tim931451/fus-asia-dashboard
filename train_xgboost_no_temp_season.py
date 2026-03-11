import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

CSV_PATH = "engineered_features_daily.csv"


def time_series_split_75_25(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(np.floor(len(df) * 0.75))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def main():
    df = pd.read_csv(CSV_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Target
    y = pd.to_numeric(df["orders_cnt"], errors="coerce").fillna(0)

    # Features
    X = df.drop(columns=["orders_cnt"], errors="ignore")
    X = X.drop(columns=["date"], errors="ignore")

    # Drop selected columns (edit this set to run experiments)
    drop_cols = {"temperature_2m_mean", "sin_doy", "cos_doy", "weathercode", "windspeed_10m_max"}
    X = X.drop(columns=[c for c in X.columns if c in drop_cols], errors="ignore")

    # Split (time-based)
    train_df, test_df = time_series_split_75_25(df)
    train_idx = train_df.index
    test_idx = test_df.index

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_test = X.loc[test_idx]
    y_test = y.loc[test_idx]

    print("Dropped columns:", sorted(list(drop_cols & set(df.columns))))
    print("Train:", train_df["date"].min(), "->", train_df["date"].max(), "| n=", len(train_df))
    print("Test :", test_df["date"].min(), "->", test_df["date"].max(), "| n=", len(test_df))

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

    print("\n=== Test Metrics (experiment) ===")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("MAPE (approx):", round(mape, 4))

    # Plot: actual vs predicted
    plt.figure()
    plt.plot(test_df["date"].reset_index(drop=True), y_test.reset_index(drop=True), label="Ist")
    plt.plot(test_df["date"].reset_index(drop=True), pd.Series(y_pred), label="Prognose")
    plt.title("Ist vs Prognose (Experiment)")
    plt.xlabel("Datum")
    plt.ylabel("Bestellungen")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Feature importances
    imp = (
        pd.DataFrame({"feature": X_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(25)
    )
    print("\nTop 25 Features (importance):")
    print(imp.to_string(index=False))

    plt.figure()
    plt.barh(imp["feature"][::-1], imp["importance"][::-1])
    plt.title("Top 25 Feature Importances (Experiment)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()