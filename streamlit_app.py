import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import requests
from datetime import datetime

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

st.set_page_config(
    page_title="Zindi Climate & Health Predictor",
    page_icon="🌍",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    return joblib.load("model_artifacts.pkl")


artifacts = load_artifacts()
models = artifacts["models"]
meta_model = artifacts["meta_model"]
iso_cal = artifacts["iso_calibrator"]
knn_imputers = artifacts["knn_imputer"]
kmeans_geo = artifacts["kmeans_geo"]
scaler_geo = artifacts["scaler_geo"]
target_encoders = artifacts["target_encoders"]
feature_medians = artifacts["feature_medians"]
BASE_FEATURES = artifacts["BASE_FEATURES"]
best_thr = artifacts["best_thr"]
global_mean = artifacts["global_mean"]
use_stacknet = artifacts["use_stacknet"]


def build_features(df):
    df = df.copy()
    df["deathdate"] = pd.to_datetime(df["deathdate"])
    df["month"] = df["deathdate"].dt.month
    df["year"] = df["deathdate"].dt.year
    df["day_of_year"] = df["deathdate"].dt.dayofyear
    df["season"] = df["month"].map({12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3})
    df["gender_enc"] = df["gender"].map({"Male": 0, "Female": 1})
    df["zone_enc"] = df["zone"].map({"Rural": 0, "Peri_urban": 1})

    a = df["age"]
    df["age_0_1"] = (a <= 1).astype(int)
    df["age_0_2"] = (a <= 2).astype(int)
    df["age_0_3"] = (a <= 3).astype(int)
    df["age_0_5"] = (a <= 5).astype(int)
    df["age_0_10"] = (a <= 10).astype(int)
    df["age_0_15"] = (a <= 15).astype(int)
    df["age_0_20"] = (a <= 20).astype(int)
    df["age_50p"] = (a >= 50).astype(int)
    df["age_60p"] = (a >= 60).astype(int)
    df["age_65p"] = (a >= 65).astype(int)
    df["age_sq"] = a ** 2
    df["age_cb"] = a ** 3
    df["age_log"] = np.log1p(a)
    df["age_sqrt"] = np.sqrt(a)
    df["age_inv"] = 1.0 / (a + 1.0)
    df["age_inv_sq"] = df["age_inv"] ** 2
    df["age_bucket"] = pd.cut(
        a, bins=[-1, 1, 3, 5, 10, 15, 20, 30, 40, 50, 60, 70, 200], labels=range(12)
    ).astype(float)

    df["age_inv_x_year"] = df["age_inv"] * df["year"]
    df["age_0_5_x_year"] = df["age_0_5"] * df["year"]
    df["age_0_3_x_year"] = df["age_0_3"] * df["year"]
    df["year_sq"] = df["year"] ** 2
    df["year_norm"] = (df["year"] - 2007) / (2022 - 2007)

    df["temp_range_30d"] = df["tmax_30d"] - df["tmin_30d"]
    df["temp_diff_avg_max"] = df["tmax_30d"] - df["tavg_30d"]
    df["temp_trend_30_90"] = df["tavg_30d"] - df["tavg_90d"]
    df["temp_trend_7_30"] = df["tavg_7d"] - df["tavg_30d"]
    df["heat_stress"] = (df["tmax_30d"] > 35).astype(int)
    df["temp_x_humidity"] = df["tavg_30d"] * df["ext_humidity"]
    df["heat_index"] = df["tmax_30d"] * df["ext_humidity"] / 100

    df["rain_trend_30_90"] = df["rain_sum_30d"] - df["rain_sum_90d"] / 3
    df["rain_frac_7_30"] = df["rain_sum_7d"] / (df["rain_sum_30d"] + 1)
    df["rain_intensity"] = df["max_daily_rain_30d"] / (df["rain_days_30d"] + 1)
    df["rain_days_frac"] = df["rain_days_30d"] / 30.0
    df["rain_sq"] = df["rain_sum_30d"] ** 2
    df["log_rain"] = np.log1p(df["rain_sum_30d"])

    df["ndvi_trend"] = df["ndvi_30d"] - df["ndvi_90d"]
    df["drought_index"] = df["ndvi_30d"] / (df["rain_sum_30d"] + 1)
    df["ndvi_rain"] = df["ndvi_30d"] * df["rain_sum_30d"]

    df["pressure_anom"] = df["ext_pressure"] - df["ext_pressure"].median()
    df["humidity_sq"] = df["ext_humidity"] ** 2
    df["humid_x_rain"] = df["ext_humidity"] * df["rain_sum_30d"]
    df["lat_lon"] = df["latitude"] * df["longitude"]
    df["elevation_sq"] = df["elevation"] ** 2
    df["elev_per_slope"] = df["elevation"] / (df["slope"] + 0.01)

    df["age_inv_x_rain"] = df["age_inv"] * df["rain_sum_30d"]
    df["age_inv_x_temp"] = df["age_inv"] * df["tmax_30d"]
    df["age_inv_x_ndvi"] = df["age_inv"] * df["ndvi_30d"]
    df["age_inv_x_humid"] = df["age_inv"] * df["ext_humidity"]
    df["age_0_5_x_rain"] = df["age_0_5"] * df["rain_sum_30d"]
    df["age_0_5_x_temp"] = df["age_0_5"] * df["tavg_30d"]
    df["age_0_3_x_rain"] = df["age_0_3"] * df["rain_sum_30d"]
    df["age_65p_x_heat"] = df["age_65p"] * df["tmax_30d"]
    df["age_x_month"] = df["age_inv"] * df["month"]
    df["age_inv_x_log_rain"] = df["age_inv"] * df["log_rain"]
    return df


def fetch_openmeteo(latitude, longitude, deathdate):
    try:
        dt = pd.to_datetime(deathdate)
        year = dt.year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["relative_humidity_2m_max", "surface_pressure_mean"],
            "timezone": "auto",
        }
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        daily = data.get("daily", {})
        humidity_vals = daily.get("relative_humidity_2m_max", [])
        pressure_vals = daily.get("surface_pressure_mean", [])
        ext_humidity = float(np.nanmean(humidity_vals)) if humidity_vals else np.nan
        ext_pressure = float(np.nanmean(pressure_vals)) if pressure_vals else np.nan
        return ext_humidity, ext_pressure
    except Exception:
        return np.nan, np.nan


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["ext_humidity", "ext_pressure"]:
        if col not in df.columns or df[col].isna().any():
            df[col] = knn_imputers[col].predict(df[["latitude", "longitude"]].values)
    df["geo_cluster"] = kmeans_geo.predict(
        scaler_geo.transform(df[["latitude", "longitude"]].values)
    )
    df = build_features(df)
    for col in ["zone", "geo_cluster", "year"]:
        df[f"{col}_risk"] = df[col].map(target_encoders[col]).fillna(global_mean)
    X = df[BASE_FEATURES].fillna(feature_medians)

    p_xgb = models["xgb"].predict_proba(X)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X)[:, 1]
    p_cat = models["cat"].predict_proba(X)[:, 1]
    p_et = models["et"].predict_proba(X)[:, 1]

    p_blend = 0.10 * p_xgb + 0.50 * p_lgbm + 0.30 * p_cat + 0.10 * p_et

    if use_stacknet:
        meta_X = np.column_stack([p_xgb, p_lgbm, p_cat, p_et])
        p_meta = meta_model.predict_proba(meta_X)[:, 1]
        p_final = 0.70 * p_meta + 0.30 * p_blend
    else:
        p_final = p_blend

    p_cal = iso_cal.predict(p_final)
    df["probability"] = p_cal
    df["is_climate_sensitive"] = (p_cal >= best_thr).astype(int)
    return df


st.title("🌍 Zindi Climate & Health Predictor")
st.caption("Modèle v3 — Score Zindi: 0.8328 — Rank 41")

tab1, tab2 = st.tabs(["Prédiction unitaire", "Prédiction batch (CSV)"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Genre", ["Male", "Female"])
        zone = st.selectbox("Zone", ["Rural", "Peri_urban"])
    with col2:
        latitude = st.number_input("Latitude", value=0.5, format="%.4f")
        longitude = st.number_input("Longitude", value=33.5, format="%.4f")
        deathdate = st.date_input("Date de décès", datetime(2015, 6, 15))
    with col3:
        avg_temperature = st.number_input("Température moyenne", value=22.0)
        max_temperature = st.number_input("Température max", value=28.0)
        min_temperature = st.number_input("Température min", value=16.0)

    with st.expander("Paramètres climatiques avancés"):
        c1, c2, c3 = st.columns(3)
        with c1:
            precipitation = st.number_input("Précipitation", value=2.5)
            elevation = st.number_input("Élévation (m)", value=1200.0)
            slope = st.number_input("Pente", value=0.2)
        with c2:
            hot_days_30d = st.number_input("hot_days_30d", value=3)
            max_daily_rain_30d = st.number_input("max_daily_rain_30d", value=25.0)
            rain_sum_7d = st.number_input("rain_sum_7d", value=10.0)
            rain_sum_30d = st.number_input("rain_sum_30d", value=50.0)
            rain_sum_90d = st.number_input("rain_sum_90d", value=180.0)
            rain_days_30d = st.number_input("rain_days_30d", value=8)
        with c3:
            tavg_7d = st.number_input("tavg_7d", value=21.0)
            tavg_30d = st.number_input("tavg_30d", value=21.5)
            tavg_90d = st.number_input("tavg_90d", value=22.0)
            tmax_30d = st.number_input("tmax_30d", value=27.0)
            tmin_30d = st.number_input("tmin_30d", value=16.0)
            ndvi_30d = st.number_input("ndvi_30d", value=0.6)
            ndvi_90d = st.number_input("ndvi_90d", value=0.65)
            temp_range_mean_30d = st.number_input("temp_range_mean_30d", value=8.0)

    fetch_weather = st.checkbox("Récupérer humidité/pression via Open-Meteo", value=True)

    if st.button("Prédire", type="primary"):
        input_data = {
            "ID": "PRED",
            "zone": zone,
            "gender": gender,
            "deathdate": str(deathdate),
            "age": float(age),
            "avg_temperature": float(avg_temperature),
            "max_temperature": float(max_temperature),
            "min_temperature": float(min_temperature),
            "precipitation": float(precipitation),
            "latitude": float(latitude),
            "longitude": float(longitude),
            "location": "",
            "elevation": float(elevation),
            "slope": float(slope),
            "hot_days_30d": float(hot_days_30d),
            "max_daily_rain_30d": float(max_daily_rain_30d),
            "rain_sum_7d": float(rain_sum_7d),
            "rain_sum_30d": float(rain_sum_30d),
            "rain_sum_90d": float(rain_sum_90d),
            "rain_days_30d": float(rain_days_30d),
            "tavg_7d": float(tavg_7d),
            "tavg_30d": float(tavg_30d),
            "tavg_90d": float(tavg_90d),
            "tmax_30d": float(tmax_30d),
            "tmin_30d": float(tmin_30d),
            "ndvi_30d": float(ndvi_30d),
            "ndvi_90d": float(ndvi_90d),
            "temp_range_mean_30d": float(temp_range_mean_30d),
        }
        df_input = pd.DataFrame([input_data])

        if fetch_weather:
            with st.spinner("Récupération des données météo..."):
                hum, pres = fetch_openmeteo(latitude, longitude, deathdate)
            df_input["ext_humidity"] = hum if not np.isnan(hum) else feature_medians.get("ext_humidity", 60.0)
            df_input["ext_pressure"] = pres if not np.isnan(pres) else feature_medians.get("ext_pressure", 1013.0)
        else:
            df_input["ext_humidity"] = feature_medians.get("ext_humidity", 60.0)
            df_input["ext_pressure"] = feature_medians.get("ext_pressure", 1013.0)

        result = predict_df(df_input)
        proba = float(result["probability"].iloc[0])
        label = int(result["is_climate_sensitive"].iloc[0])

        st.subheader("Résultat")
        c1, c2, c3 = st.columns(3)
        c1.metric("Probabilité", f"{proba:.3f}")
        c2.metric("Seuil optimal", f"{best_thr:.3f}")
        c3.metric("Prédiction", "Sensible" if label == 1 else "Non sensible")

        if label == 1:
            st.warning("Cas identifié comme **climatiquement sensible**.")
        else:
            st.success("Cas identifié comme **non climatiquement sensible**.")

with tab2:
    uploaded = st.file_uploader("Charger un CSV avec les colonnes du Test set", type=["csv"])
    if uploaded and st.button("Lancer la prédiction batch", type="primary"):
        df_batch = pd.read_csv(uploaded)
        if "is_climate_sensitive" in df_batch.columns:
            df_batch = df_batch.drop(columns=["is_climate_sensitive"])
        if "ID" not in df_batch.columns:
            st.error("Le CSV doit contenir une colonne 'ID'.")
        else:
            result = predict_df(df_batch)
            out = result[["ID", "probability", "is_climate_sensitive"]].copy()
            out["is_climate_sensitive"] = out["is_climate_sensitive"].astype(int)
            st.dataframe(out.head(20), use_container_width=True)
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
