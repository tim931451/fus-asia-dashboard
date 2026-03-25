"""
Streamlit Dashboard: FuBox Basel - Bestellungen & Wetter
Starten:  streamlit run dashboard.py
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from datetime import datetime, timedelta
from holidays import country_holidays
from sklearn.metrics import mean_absolute_error, mean_squared_error

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FU'S ASIA RESTAURANT – Wettereinfluss auf Bestellungen",
    page_icon=":ramen:",
    layout="wide",
)

WEEKDAY_NAMES = {0: "Mo", 1: "Di", 2: "Mi", 3: "Do", 4: "Fr", 5: "Sa", 6: "So"}
WEEKDAY_OPTIONS = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# Brand colours
PRIMARY   = "#C0392B"   # deep red
SECONDARY = "#E67E22"   # warm orange
ACCENT    = "#2C3E50"   # dark navy
RAIN_COL  = "#4287f5"
DRY_COL   = "#E67E22"

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---- font & base ---- */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ---- hero banner ---- */
.hero {
    background: linear-gradient(135deg, #2C3E50 0%, #C0392B 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    overflow: hidden;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 0.3rem 0; letter-spacing: 1px; }
.hero p  { font-size: 1.05rem; margin: 0 0 0.3rem 0; opacity: 0.85; }
.hero small { opacity: 0.6; font-size: 0.8rem; }
.hero img { border-radius: 8px; }

/* ---- KPI cards ---- */
.kpi-row { display: flex; gap: 0.8rem; margin-bottom: 1.5rem; flex-wrap: nowrap; }
.kpi-card {
    flex: 1; min-width: 0;
    background: white;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #C0392B;
    overflow: hidden;
}
.kpi-label { font-size: 0.65rem; color: #888; text-transform: uppercase; letter-spacing: 0.3px; white-space: nowrap; }
.kpi-value { font-size: 1.4rem; font-weight: 700; color: #2C3E50; line-height: 1.2; white-space: nowrap; }
.kpi-sub   { font-size: 0.7rem; color: #aaa; white-space: nowrap; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #2C3E50 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMarkdown * ,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMultiSelect span,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] summary,
[data-testid="stSidebar"] .stButton button { color: white !important; }
/* Input-Felder: dunkler Text auf weissem Grund */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] [data-baseweb="input"] * { color: #2C3E50 !important; }
/* Logo ganz nach oben schieben */
[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem !important; }
section[data-testid="stSidebar"] > div > div:first-child { padding-top: 0.5rem !important; }

/* ---- tabs ---- */
button[data-baseweb="tab"] { font-weight: 600; font-size: 0.9rem; }

/* ---- divider ---- */
hr { border-color: #eee; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data
def load_joined():
    df = pd.read_csv("weather_joined_remote_daily.csv")
    df["weather_date"] = pd.to_datetime(df["weather_date"], errors="coerce")
    for c in ["orders_cnt", "orders_value_sum", "temperature_2m_mean",
              "windspeed_10m_max", "rain_sum", "precipitation_sum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["orders_cnt"] = df["orders_cnt"].fillna(0)
    df = df.dropna(subset=["weather_date"]).sort_values("weather_date").reset_index(drop=True)
    # rain flag
    rain = df["rain_sum"].copy()
    rain = rain.where(rain.notna(), df.get("precipitation_sum"))
    df["is_rain_day"] = (rain.fillna(0) > 0)
    df["weekday"] = df["weather_date"].dt.weekday
    return df


@st.cache_data
def load_features():
    df = pd.read_csv("engineered_features_daily.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df["orders_cnt"] = pd.to_numeric(df["orders_cnt"], errors="coerce").fillna(0)
    return df


@st.cache_resource
def load_models():
    models = {"global": joblib.load("models/xgb_global.pkl")}
    for wd in range(7):
        try:
            models[f"wd_{wd}"] = joblib.load(f"models/xgb_weekday_{wd}.pkl")
        except FileNotFoundError:
            pass
    with open("models/feature_columns.json") as f:
        meta = json.load(f)
    return models, meta


@st.cache_data
def load_school_holidays():
    try:
        h = pd.read_csv("school_holidays_bs.csv")
        h["date"] = pd.to_datetime(h["date"], errors="coerce")
        return set(h["date"].dt.normalize().dropna())
    except FileNotFoundError:
        return set()


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------
df_joined = load_joined()
df_feat = load_features()
models, model_meta = load_models()
school_hols = load_school_holidays()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width=200)
st.sidebar.markdown("""
<div style='text-align:center; margin-top:-0.5rem; margin-bottom:0.8rem;'>
    <span style='font-size:0.75rem; opacity:0.7; letter-spacing:1px; text-transform:uppercase;'>
        Data Engineering & Wrangling
    </span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("## 🔍 Filter")
st.sidebar.markdown("---")

date_min = df_joined["weather_date"].min().date()
date_max = df_joined["weather_date"].max().date()

st.sidebar.markdown("**Zeitraum**")
col_von, col_bis = st.sidebar.columns(2)
with col_von:
    date_start = st.date_input("Von", value=date_min, min_value=date_min, max_value=date_max, key="d_start")
with col_bis:
    date_end = st.date_input("Bis", value=date_max, min_value=date_min, max_value=date_max, key="d_end")

date_range = (date_start, date_end)


selected_days = st.sidebar.multiselect(
    "Wochentage",
    options=WEEKDAY_OPTIONS,
    default=WEEKDAY_OPTIONS,
)
selected_wd = [WEEKDAY_OPTIONS.index(d) for d in selected_days]

rain_filter = st.sidebar.radio("Wetter", ["Alle", "Nur Regen", "Nur Trocken"])

# Apply filters
def apply_filters(df, date_col="weather_date"):
    mask = pd.Series(True, index=df.index)
    if len(date_range) == 2:
        mask &= (df[date_col].dt.date >= date_range[0]) & (df[date_col].dt.date <= date_range[1])
    mask &= df["weekday"].isin(selected_wd)
    if rain_filter == "Nur Regen":
        mask &= df["is_rain_day"]
    elif rain_filter == "Nur Trocken":
        mask &= ~df["is_rain_day"]
    return df[mask].copy()


filtered = apply_filters(df_joined)

# ---------------------------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------------------------
if os.path.exists("logo.png"):
    logo_b64 = __import__("base64").b64encode(open("logo.png", "rb").read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" style="height:110px; object-fit:contain; float:right; margin-left:1.5rem; margin-top:-0.3rem;" />'
else:
    logo_html = ""

st.markdown(f"""
<div class="hero">
    {logo_html}
    <h1>🍜 FU'S ASIA RESTAURANT</h1>
    <p>Wie beeinflusst das Wetter die Bestellmenge?</p>
    <small>Liestal &nbsp;|&nbsp; Daten: {date_min} – {date_max} &nbsp;|&nbsp; {len(df_joined):,} Tage analysiert</small>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------
_kdf = df_joined[df_joined["orders_value_sum"].notna() & (df_joined["orders_value_sum"] > 0)]
total_orders   = int(df_joined["orders_cnt"].sum())
total_revenue  = _kdf["orders_value_sum"].sum()
_rev_fmt = f"{total_revenue/1_000_000:.1f}M" if total_revenue >= 1_000_000 else f"{total_revenue:,.0f}"
avg_per_day    = df_joined["orders_cnt"].mean()
best_wd_idx    = df_joined.groupby("weekday")["orders_cnt"].mean().idxmax()
best_wd_name   = WEEKDAY_NAMES[best_wd_idx]
rain_days      = int(df_joined["is_rain_day"].sum())
rain_pct       = rain_days / len(df_joined) * 100

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-label">Gesamtbestellungen</div>
    <div class="kpi-value">{total_orders:,}</div>
    <div class="kpi-sub">seit {date_min}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Gesamtumsatz</div>
    <div class="kpi-value">CHF {_rev_fmt}</div>
    <div class="kpi-sub">Summe Bestellwert</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Ø Bestellungen / Tag</div>
    <div class="kpi-value">{avg_per_day:.1f}</div>
    <div class="kpi-sub">Tagesdurchschnitt</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Stärkster Wochentag</div>
    <div class="kpi-value">{best_wd_name}</div>
    <div class="kpi-sub">nach Ø Bestellungen</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Regentage</div>
    <div class="kpi-value">{rain_days}</div>
    <div class="kpi-sub">{rain_pct:.0f}% aller Tage</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab8, tab5, tab6, tab9, tab7 = st.tabs([
    "📈 Bestellungen",
    "💰 Monatlicher Umsatz",
    "🌡️ Temperatur vs. Umsatz",
    "🌧️ Regen vs. Trocken",
    "🔗 Korrelationsmatrix",
    "🤖 Modell-Performance",
    "⭐ Feature Importance",
    "📊 Prognose heute",
    "🔮 Prognose morgen",
])

# ===== TAB 1: Bestellungen Timeseries =====
with tab1:
    if filtered.empty:
        st.warning("Keine Daten für diesen Filter.")
    else:
        metric_choice = st.radio("Metrik", ["Bestellungen", "Umsatz"], horizontal=True, key="ts_metric")
        y_col = "orders_cnt" if metric_choice == "Bestellungen" else "orders_value_sum"

        monthly = (
            filtered[filtered[y_col].notna() & (filtered[y_col] > 0)]
            .set_index("weather_date")
            .resample("MS")
            .agg(value=(y_col, "sum"), tage=(y_col, "size"))
            .reset_index()
        )
        # Drop incomplete last month
        if not monthly.empty:
            last_period = filtered["weather_date"].max().to_period("M")
            monthly = monthly[monthly["weather_date"].dt.to_period("M") != last_period]

        fig = px.line(monthly, x="weather_date", y="value",
                      labels={"weather_date": "Monat", "value": metric_choice},
                      title=f"Monatliche {metric_choice}",
                      color_discrete_sequence=[PRIMARY])
        fig.update_traces(mode="lines+markers", line=dict(width=3), marker=dict(size=7))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig, use_container_width=True)

        # Daily view with rolling average
        daily = filtered.sort_values("weather_date")
        daily["rolling_7"] = daily[y_col].rolling(7, min_periods=1).mean()
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=daily["weather_date"], y=daily[y_col],
                                   mode="lines", name="Täglich", opacity=0.25,
                                   line=dict(color=PRIMARY)))
        fig2.add_trace(go.Scatter(x=daily["weather_date"], y=daily["rolling_7"],
                                   mode="lines", name="7-Tage Schnitt",
                                   line=dict(width=3, color=SECONDARY)))
        fig2.update_layout(title=f"Tägliche {metric_choice} (mit 7-Tage Schnitt)",
                           xaxis_title="Datum", yaxis_title=metric_choice,
                           plot_bgcolor="white", paper_bgcolor="white",
                           xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)

# ===== TAB 2: Monatlicher Umsatz =====
with tab2:
    if filtered.empty:
        st.warning("Keine Daten für diesen Filter.")
    else:
        rev = filtered[filtered["orders_value_sum"].notna() & (filtered["orders_value_sum"] > 0)].copy()
        monthly_rev = (
            rev.set_index("weather_date")
            .resample("MS")
            .agg(umsatz=("orders_value_sum", "sum"), bestellungen=("orders_cnt", "sum"))
            .reset_index()
        )
        if not monthly_rev.empty:
            last_p = rev["weather_date"].max().to_period("M")
            monthly_rev = monthly_rev[monthly_rev["weather_date"].dt.to_period("M") != last_p]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_rev["weather_date"], y=monthly_rev["umsatz"],
                             name="Umsatz", marker_color=PRIMARY, opacity=0.85))
        fig.add_trace(go.Scatter(x=monthly_rev["weather_date"], y=monthly_rev["bestellungen"],
                                  name="Bestellungen", yaxis="y2", mode="lines+markers",
                                  line=dict(color=SECONDARY, width=3), marker=dict(size=7)))
        fig.update_layout(
            title="Monatlicher Umsatz & Bestellungen",
            xaxis_title="Monat",
            yaxis=dict(title="Umsatz (CHF)", gridcolor="#f0f0f0"),
            yaxis2=dict(title="Bestellungen", overlaying="y", side="right"),
            legend=dict(x=0, y=1.1, orientation="h"),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3: Temperatur vs. Umsatz =====
with tab3:
    if filtered.empty:
        st.warning("Keine Daten für diesen Filter.")
    else:
        scatter_y = st.radio("Y-Achse", ["Umsatz", "Bestellungen"], horizontal=True, key="scatter_y")
        y_col_s = "orders_value_sum" if scatter_y == "Umsatz" else "orders_cnt"
        temp = filtered.dropna(subset=["temperature_2m_mean", y_col_s])
        temp = temp[temp[y_col_s] > 0]

        fig = px.scatter(
            temp, x="temperature_2m_mean", y=y_col_s,
            color="is_rain_day",
            color_discrete_map={True: RAIN_COL, False: DRY_COL},
            trendline="ols",
            labels={"temperature_2m_mean": "Temperatur (C)", y_col_s: scatter_y,
                    "is_rain_day": "Regen"},
            title=f"{scatter_y} vs. Durchschnittstemperatur",
            opacity=0.5,
        )
        st.plotly_chart(fig, use_container_width=True)

# ===== TAB 4: Regen vs. Trocken =====
with tab4:
    box_y = st.radio("Metrik", ["Umsatz", "Bestellungen"], horizontal=True, key="box_y")
    y_col_b = "orders_value_sum" if box_y == "Umsatz" else "orders_cnt"
    box_df = df_joined.copy()  # unfiltered for meaningful comparison
    box_df = box_df[box_df[y_col_b].notna() & (box_df[y_col_b] > 0)]
    box_df["Wetter"] = box_df["is_rain_day"].map({True: "Regen", False: "Trocken"})

    fig = px.box(box_df, x="Wetter", y=y_col_b, color="Wetter",
                 color_discrete_map={"Regen": RAIN_COL, "Trocken": DRY_COL},
                 title=f"{box_y}: Regen vs. Trocken",
                 labels={y_col_b: box_y})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    dry_vals = box_df[box_df["Wetter"] == "Trocken"][y_col_b]
    rain_vals = box_df[box_df["Wetter"] == "Regen"][y_col_b]
    col1.metric("Trocken (Median)", f"{dry_vals.median():.0f}", f"n = {len(dry_vals)}")
    col2.metric("Regen (Median)", f"{rain_vals.median():.0f}", f"n = {len(rain_vals)}")

    # Weekday breakdown
    st.subheader("Aufschlüsselung nach Wochentag")
    box_df["Wochentag"] = box_df["weekday"].map(WEEKDAY_NAMES)
    fig2 = px.box(box_df, x="Wochentag", y=y_col_b, color="Wetter",
                  color_discrete_map={"Regen": RAIN_COL, "Trocken": DRY_COL},
                  category_orders={"Wochentag": WEEKDAY_OPTIONS},
                  labels={y_col_b: box_y})
    st.plotly_chart(fig2, use_container_width=True)

# ===== TAB 8: Korrelationsmatrix =====
with tab8:
    st.subheader("Korrelation der Features mit Bestellmenge")
    st.caption("Pearson-Korrelation aller numerischen Features untereinander und mit dem Target (orders_cnt).")

    # Use the engineered features for correlation
    corr_cols = [
        "orders_cnt", "temperature_2m_mean", "windspeed_10m_max", "is_rain",
        "weathercode", "sin_doy", "cos_doy", "is_public_holiday",
        "is_school_holiday", "lag_7", "rolling_mean_7",
    ]
    corr_df = df_feat[[c for c in corr_cols if c in df_feat.columns]].copy()

    # Readable names
    rename = {
        "orders_cnt": "Bestellungen",
        "temperature_2m_mean": "Temperatur",
        "windspeed_10m_max": "Wind",
        "is_rain": "Regen",
        "weathercode": "Wettercode",
        "sin_doy": "Saison (sin)",
        "cos_doy": "Saison (cos)",
        "is_public_holiday": "Feiertag",
        "is_school_holiday": "Schulferien",
        "lag_7": "Bestellungen (vor 7d)",
        "rolling_mean_7": "Schnitt 7d",
    }
    corr_df = corr_df.rename(columns=rename)
    corr_matrix = corr_df.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        aspect="auto",
        title="Korrelationsmatrix",
    )
    fig.update_layout(width=800, height=700)
    st.plotly_chart(fig, use_container_width=True)

    # Highlight: correlation with target
    st.subheader("Korrelation mit Bestellmenge")
    target_corr = corr_matrix["Bestellungen"].drop("Bestellungen").sort_values(key=abs, ascending=False)
    fig2 = px.bar(
        x=target_corr.values, y=target_corr.index, orientation="h",
        labels={"x": "Korrelation", "y": "Feature"},
        title="Korrelation jedes Features mit Bestellmenge",
        color=target_corr.values,
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
    )
    fig2.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig2, use_container_width=True)

# ===== TAB 5: Modell-Performance =====
with tab5:
    model_type = st.radio("Modell", ["Global", "Pro Wochentag"], horizontal=True)

    if model_type == "Global":
        # Reproduce train/test split
        y_all = df_feat["orders_cnt"]
        X_all = df_feat.drop(columns=["orders_cnt", "date"], errors="ignore")
        n = len(df_feat)
        split = int(np.floor(n * 0.75))

        X_test = X_all.iloc[split:]
        y_test = y_all.iloc[split:]
        test_dates = df_feat["date"].iloc[split:]

        pred = np.clip(models["global"].predict(X_test), 0, None)
        mae = mean_absolute_error(y_test, pred)
        rmse = _rmse(y_test, pred)
        denom = np.maximum(y_test.to_numpy(), 1)
        mape = np.mean(np.abs((y_test.to_numpy() - pred) / denom))

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("MAPE", f"{mape:.1%}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_dates, y=y_test, name="Ist",
                                  mode="lines", line=dict(color=ACCENT, width=2)))
        fig.add_trace(go.Scatter(x=test_dates, y=pred, name="Prognose",
                                  mode="lines", line=dict(color=PRIMARY, width=2, dash="dot")))
        fig.update_layout(title="Ist vs. Prognose (Testzeitraum)",
                          xaxis_title="Datum", yaxis_title="Bestellungen",
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Per-weekday models
        df_wd = df_feat.copy()
        df_wd["weekday"] = df_wd["date"].dt.weekday
        X_all_wd = df_wd.drop(columns=["orders_cnt", "date", "weekday"], errors="ignore")
        wd_c = [c for c in X_all_wd.columns if c.startswith("wd_")]
        if wd_c:
            X_all_wd = X_all_wd.drop(columns=wd_c)

        results = []
        for wd in range(7):
            key = f"wd_{wd}"
            if key not in models:
                continue
            sub = df_wd[df_wd["weekday"] == wd].copy()
            orig_idx = sub.index
            sub = sub.sort_values("date").reset_index(drop=True)
            X_sub = X_all_wd.loc[orig_idx].copy().reset_index(drop=True)
            y_sub = sub["orders_cnt"]

            sp = int(np.floor(len(sub) * 0.75))
            y_te = y_sub.iloc[sp:]
            X_te = X_sub.iloc[sp:]

            p = np.clip(models[key].predict(X_te), 0, None)
            m = mean_absolute_error(y_te, p)
            r = _rmse(y_te, p)
            d = np.maximum(y_te.to_numpy(), 1)
            mp = np.mean(np.abs((y_te.to_numpy() - p) / d))

            results.append({"Wochentag": WEEKDAY_NAMES[wd], "n": len(sub),
                            "MAE": round(m, 2), "RMSE": round(r, 2), "MAPE": f"{mp:.1%}"})

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

        # Show one weekday in detail
        detail_wd = st.selectbox("Detail-Ansicht", WEEKDAY_OPTIONS)
        wd_idx = WEEKDAY_OPTIONS.index(detail_wd)
        key = f"wd_{wd_idx}"
        if key in models:
            sub = df_wd[df_wd["weekday"] == wd_idx].sort_values("date").reset_index(drop=True)
            X_sub = X_all_wd.loc[df_wd[df_wd["weekday"] == wd_idx].index].reset_index(drop=True)
            y_sub = sub["orders_cnt"]
            sp = int(np.floor(len(sub) * 0.75))

            pred_wd = np.clip(models[key].predict(X_sub.iloc[sp:]), 0, None)
            dates_wd = sub["date"].iloc[sp:]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dates_wd, y=y_sub.iloc[sp:], name="Ist",
                                      line=dict(color=ACCENT, width=2)))
            fig.add_trace(go.Scatter(x=dates_wd, y=pred_wd, name="Prognose",
                                      line=dict(color=PRIMARY, width=2, dash="dot")))
            fig.update_layout(title=f"{detail_wd}: Ist vs. Prognose",
                              xaxis_title="Datum", yaxis_title="Bestellungen",
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"),
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)

# ===== TAB 6: Feature Importance =====
with tab6:
    fi_model = st.radio("Modell", ["Global"] + WEEKDAY_OPTIONS, horizontal=True, key="fi_model")

    if fi_model == "Global":
        m = models["global"]
        cols = model_meta["feature_cols_global"]
    else:
        wd_idx = WEEKDAY_OPTIONS.index(fi_model)
        key = f"wd_{wd_idx}"
        if key not in models:
            st.warning(f"Kein Modell für {fi_model}")
            st.stop()
        m = models[key]
        cols = model_meta["feature_cols_weekday"]

    imp = pd.DataFrame({"Feature": cols, "Importance": m.feature_importances_})
    imp = imp.sort_values("Importance", ascending=True).tail(15)

    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 title=f"Top 15 Features ({fi_model})",
                 labels={"Importance": "Importance Score"},
                 color="Importance",
                 color_continuous_scale=[[0, "#f5d0cc"], [1, PRIMARY]])
    fig.update_layout(yaxis=dict(categoryorder="total ascending"),
                      plot_bgcolor="white", paper_bgcolor="white",
                      xaxis=dict(gridcolor="#f0f0f0", showgrid=True),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Rain counterfactual
    st.subheader("Rain-Effekt (Counterfactual)")
    st.caption("Differenz: Prognose(Regen=1) - Prognose(Regen=0) auf dem Testset")

    y_all = df_feat["orders_cnt"]
    X_all = df_feat.drop(columns=["orders_cnt", "date"], errors="ignore")
    n = len(df_feat)
    split = int(np.floor(n * 0.75))
    X_test_cf = X_all.iloc[split:].copy()

    if "is_rain" in X_test_cf.columns:
        X0 = X_test_cf.copy(); X0["is_rain"] = 0
        X1 = X_test_cf.copy(); X1["is_rain"] = 1
        p0 = np.clip(models["global"].predict(X0), 0, None)
        p1 = np.clip(models["global"].predict(X1), 0, None)
        effect = p1 - p0

        c1, c2, c3 = st.columns(3)
        c1.metric("Mittlerer Effekt", f"{np.mean(effect):+.2f}")
        c2.metric("Median Effekt", f"{np.median(effect):+.2f}")
        c3.metric("5% / 95%", f"{np.quantile(effect, 0.05):+.1f} / {np.quantile(effect, 0.95):+.1f}")

        fig = px.histogram(effect, nbins=40, title="Verteilung Rain-Effekt",
                           labels={"value": "Delta Bestellungen (Regen - Kein Regen)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ===== TAB 9: Prognose heute =====
with tab9:
    st.subheader("Bestellprognose für heute")

    today = datetime.now().date()
    st.info(f"Prognose für: **{today.strftime('%A, %d.%m.%Y')}**")

    # Fetch today's weather from Open-Meteo
    @st.cache_data(ttl=1800)
    def fetch_today_weather():
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=47.557117&longitude=7.549342"
            "&daily=temperature_2m_mean,precipitation_sum,rain_sum,windspeed_10m_max,weathercode"
            "&timezone=Europe/Zurich&forecast_days=1"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            daily = data["daily"]
            fdf = pd.DataFrame(daily)
            fdf["time"] = pd.to_datetime(fdf["time"])
            return fdf
        except Exception:
            return None

    weather_today = fetch_today_weather()

    if weather_today is None:
        st.error("Wetterdaten konnten nicht geladen werden.")
    else:
        today_row = weather_today[weather_today["time"].dt.date == today]
        if today_row.empty:
            st.error(f"Keine Wetterdaten für {today} verfügbar.")
        else:
            tw = today_row.iloc[0]
            t_temp = float(tw.get("temperature_2m_mean", 0) or 0)
            t_wind = float(tw.get("windspeed_10m_max", 0) or 0)
            t_rain = float(tw.get("rain_sum", 0) or 0)
            t_wcode = float(tw.get("weathercode", 0) or 0)
            t_is_rain = 1 if t_rain > 0 else 0

            # Wetter anzeigen
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("🌡️ Temperatur", f"{t_temp:.1f} °C")
            w2.metric("💨 Wind", f"{t_wind:.1f} km/h")
            w3.metric("🌧️ Regen", f"{t_rain:.1f} mm")
            w4.metric("☁️ Wettercode", f"{int(t_wcode)}")

            # Feature-Vektor bauen
            t_wd = today.weekday()
            t_doy = today.timetuple().tm_yday
            t_sin_doy = np.sin(2 * np.pi * t_doy / 365.25)
            t_cos_doy = np.cos(2 * np.pi * t_doy / 365.25)

            ch_hol_today = country_holidays("CH", years=[today.year])
            t_is_pub = 1 if today in ch_hol_today else 0
            t_is_school = 1 if pd.Timestamp(today) in school_hols else 0

            last_rows_t = df_feat.sort_values("date").tail(14)
            t_lag_7 = float(last_rows_t["orders_cnt"].iloc[-7]) if len(last_rows_t) >= 7 else np.nan
            t_roll_7 = float(last_rows_t["orders_cnt"].iloc[-7:].mean()) if len(last_rows_t) >= 7 else np.nan

            # Global prediction
            feat_today_g = {
                "temperature_2m_mean": t_temp,
                "windspeed_10m_max": t_wind,
                "is_rain": t_is_rain,
                "weathercode": t_wcode,
                "sin_doy": t_sin_doy,
                "cos_doy": t_cos_doy,
                "is_public_holiday": t_is_pub,
                "is_school_holiday": t_is_school,
                "lag_7": t_lag_7,
                "rolling_mean_7": t_roll_7,
            }
            for i in range(7):
                feat_today_g[f"wd_{i}"] = (t_wd == i)

            X_today_g = pd.DataFrame([feat_today_g])[model_meta["feature_cols_global"]]
            pred_today_g = max(0, float(models["global"].predict(X_today_g)[0]))

            # Weekday prediction
            feat_today_wd = {k: v for k, v in feat_today_g.items() if not k.startswith("wd_")}
            X_today_wd = pd.DataFrame([feat_today_wd])[model_meta["feature_cols_weekday"]]
            wd_key_t = f"wd_{t_wd}"
            pred_today_wd = None
            if wd_key_t in models:
                pred_today_wd = max(0, float(models[wd_key_t].predict(X_today_wd)[0]))

            st.divider()
            p1, p2 = st.columns(2)
            p1.metric("🤖 Alle-Tage-Modell", f"{pred_today_g:.0f} Bestellungen")
            if pred_today_wd is not None:
                p2.metric(f"📅 Nur-{WEEKDAY_NAMES[t_wd]}-Modell", f"{pred_today_wd:.0f} Bestellungen")

            # Counterfactual – nutzt Tagesmodell wenn vorhanden
            st.divider()
            st.subheader("Was wäre wenn?")
            feat_t_rain = feat_today_wd.copy()
            feat_t_rain["is_rain"] = 1
            feat_t_dry = feat_today_wd.copy()
            feat_t_dry["is_rain"] = 0

            if wd_key_t in models:
                cf_model = models[wd_key_t]
                cf_cols = model_meta["feature_cols_weekday"]
                cf_label = f"{WEEKDAY_NAMES[t_wd]}-Modell"
            else:
                cf_model = models["global"]
                cf_cols = model_meta["feature_cols_global"]
                cf_label = "Alle-Tage-Modell"
                feat_t_rain = feat_today_g.copy(); feat_t_rain["is_rain"] = 1
                feat_t_dry = feat_today_g.copy(); feat_t_dry["is_rain"] = 0

            X_t_rain = pd.DataFrame([feat_t_rain])[cf_cols]
            X_t_dry = pd.DataFrame([feat_t_dry])[cf_cols]

            pred_t_rain = max(0, float(cf_model.predict(X_t_rain)[0]))
            pred_t_dry = max(0, float(cf_model.predict(X_t_dry)[0]))

            st.caption(f"Berechnung mit dem **{cf_label}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🌧️ Bei Regen", f"{pred_t_rain:.0f}")
            c2.metric("☀️ Bei Trocken", f"{pred_t_dry:.0f}")
            c3.metric("Differenz", f"{pred_t_rain - pred_t_dry:+.1f}")

            # Data freshness
            last_date_t = df_feat["date"].max().date()
            days_old_t = (today - last_date_t).days
            if days_old_t > 3:
                st.warning(
                    f"Die Daten sind {days_old_t} Tage alt (letzter Eintrag: {last_date_t}). "
                    "Lag-Features könnten ungenau sein."
                )

# ===== TAB 7: Prognose morgen =====
with tab7:
    st.subheader("Bestellprognose für morgen")

    tomorrow = datetime.now().date() + timedelta(days=1)
    st.info(f"Prognose für: **{tomorrow.strftime('%A, %d.%m.%Y')}**")

    # Fetch forecast from Open-Meteo
    @st.cache_data(ttl=3600)
    def fetch_forecast():
        url = (
            "https://api.open-meteo.com/v1/forecast"
            "?latitude=47.557117&longitude=7.549342"
            "&daily=temperature_2m_mean,precipitation_sum,rain_sum,windspeed_10m_max,weathercode"
            "&timezone=Europe/Zurich&forecast_days=3"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            daily = data["daily"]
            fdf = pd.DataFrame(daily)
            fdf["time"] = pd.to_datetime(fdf["time"])
            return fdf
        except Exception as e:
            return None

    forecast = fetch_forecast()

    if forecast is None:
        st.error("Wettervorhersage konnte nicht geladen werden.")
    else:
        tmrw_row = forecast[forecast["time"].dt.date == tomorrow]
        if tmrw_row.empty:
            st.error(f"Keine Vorhersage für {tomorrow} verfügbar.")
        else:
            tmrw = tmrw_row.iloc[0]
            temp = float(tmrw.get("temperature_2m_mean", 0) or 0)
            wind = float(tmrw.get("windspeed_10m_max", 0) or 0)
            rain_sum = float(tmrw.get("rain_sum", 0) or 0)
            wcode = float(tmrw.get("weathercode", 0) or 0)
            is_rain = 1 if rain_sum > 0 else 0

            # Show weather
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Temperatur", f"{temp:.1f} C")
            w2.metric("Wind", f"{wind:.1f} km/h")
            w3.metric("Regen", f"{rain_sum:.1f} mm")
            w4.metric("Wettercode", f"{int(wcode)}")

            # Build feature vector
            wd = tomorrow.weekday()
            doy = tomorrow.timetuple().tm_yday
            sin_doy = np.sin(2 * np.pi * doy / 365.25)
            cos_doy = np.cos(2 * np.pi * doy / 365.25)

            ch_hol = country_holidays("CH", years=[tomorrow.year])
            is_pub = 1 if tomorrow in ch_hol else 0
            is_school = 1 if pd.Timestamp(tomorrow) in school_hols else 0

            # Lag features from historical data
            last_rows = df_feat.sort_values("date").tail(14)
            lag_7 = float(last_rows["orders_cnt"].iloc[-7]) if len(last_rows) >= 7 else np.nan
            roll_7 = float(last_rows["orders_cnt"].iloc[-7:].mean()) if len(last_rows) >= 7 else np.nan

            # Global model prediction
            feat_global = {
                "temperature_2m_mean": temp,
                "windspeed_10m_max": wind,
                "is_rain": is_rain,
                "weathercode": wcode,
                "sin_doy": sin_doy,
                "cos_doy": cos_doy,
                "is_public_holiday": is_pub,
                "is_school_holiday": is_school,
                "lag_7": lag_7,
                "rolling_mean_7": roll_7,
            }
            # Add weekday dummies
            for i in range(7):
                feat_global[f"wd_{i}"] = (wd == i)

            X_pred_global = pd.DataFrame([feat_global])[model_meta["feature_cols_global"]]
            pred_global = max(0, float(models["global"].predict(X_pred_global)[0]))

            # Weekday model prediction
            feat_wd = {k: v for k, v in feat_global.items() if not k.startswith("wd_")}
            X_pred_wd = pd.DataFrame([feat_wd])[model_meta["feature_cols_weekday"]]
            wd_key = f"wd_{wd}"
            pred_weekday = None
            if wd_key in models:
                pred_weekday = max(0, float(models[wd_key].predict(X_pred_wd)[0]))

            st.divider()

            # Display predictions
            p1, p2 = st.columns(2)
            p1.metric("🤖 Alle-Tage-Modell", f"{pred_global:.0f} Bestellungen")
            if pred_weekday is not None:
                p2.metric(f"📅 Nur-{WEEKDAY_NAMES[wd]}-Modell", f"{pred_weekday:.0f} Bestellungen")

            # Counterfactual – nutzt Tagesmodell wenn vorhanden
            st.divider()
            st.subheader("Was wäre wenn?")

            feat_rain_wd = feat_wd.copy()
            feat_rain_wd["is_rain"] = 1
            feat_dry_wd = feat_wd.copy()
            feat_dry_wd["is_rain"] = 0

            if wd_key in models:
                cf_model_m = models[wd_key]
                cf_cols_m = model_meta["feature_cols_weekday"]
                cf_label_m = f"{WEEKDAY_NAMES[wd]}-Modell"
            else:
                cf_model_m = models["global"]
                cf_cols_m = model_meta["feature_cols_global"]
                cf_label_m = "Alle-Tage-Modell"
                feat_rain_wd = feat_global.copy(); feat_rain_wd["is_rain"] = 1
                feat_dry_wd = feat_global.copy(); feat_dry_wd["is_rain"] = 0

            X_rain = pd.DataFrame([feat_rain_wd])[cf_cols_m]
            X_dry = pd.DataFrame([feat_dry_wd])[cf_cols_m]

            pred_rain = max(0, float(cf_model_m.predict(X_rain)[0]))
            pred_dry = max(0, float(cf_model_m.predict(X_dry)[0]))

            st.caption(f"Berechnung mit dem **{cf_label_m}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("🌧️ Bei Regen", f"{pred_rain:.0f}")
            c2.metric("☀️ Bei Trocken", f"{pred_dry:.0f}")
            c3.metric("Differenz", f"{pred_rain - pred_dry:+.1f}")

            # Data freshness warning
            last_date = df_feat["date"].max().date()
            days_old = (tomorrow - last_date).days
            if days_old > 3:
                st.warning(
                    f"Die Daten sind {days_old} Tage alt (letzter Eintrag: {last_date}). "
                    "Lag-Features könnten ungenau sein. Bitte Pipeline neu ausführen."
                )
