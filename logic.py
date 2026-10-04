"""
logic.py  –  All business logic for the Air Quality Monitor & Predictor.
No Streamlit imports. All functions take plain inputs and return DataFrames / dicts.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POLLUTANTS = ["pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "ozone", "us_aqi"]

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5 (ug/m3)",
    "pm10": "PM10 (ug/m3)",
    "nitrogen_dioxide": "NO2 (ug/m3)",
    "sulphur_dioxide": "SO2 (ug/m3)",
    "ozone": "O3 (ug/m3)",
    "us_aqi": "US AQI",
}

AQI_CATEGORIES = [
    (50,  "Good",                       "#00e400"),
    (100, "Moderate",                   "#ffff00"),
    (150, "Unhealthy for Sensitive",    "#ff7e00"),
    (200, "Unhealthy",                  "#ff0000"),
    (300, "Very Unhealthy",             "#8f3f97"),
    (500, "Hazardous",                  "#7e0023"),
]

PRESET_CITIES = {
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
}

# India NAAQS + WHO 2021 guidelines (24-h means unless noted)
NAAQS_LIMITS = {
    "pm2_5":           {"NAAQS_24h": 60.0,  "WHO_24h": 15.0},
    "pm10":            {"NAAQS_24h": 100.0, "WHO_24h": 45.0},
    "nitrogen_dioxide":{"NAAQS_24h": 80.0,  "WHO_24h": 25.0},
    "sulphur_dioxide": {"NAAQS_24h": 80.0,  "WHO_24h": 40.0},
    "ozone":           {"NAAQS_8h":  100.0, "WHO_8h":  60.0},   # 8-hour mean
}

CITY_ISSUES = {
    "Chennai": {
        "headline": "Chennai Air-Quality Concerns",
        "issues": [
            "Vehicle exhaust on corridors like Anna Salai and GST Road frequently pushes NO₂ and PM2.5 above safe limits during peak hours.",
            "Coal-based thermal plants in Ennore (North Chennai) are documented contributors to SO₂ and PM10 pollution.",
            "Seasonal crop-residue burning in the Cauvery delta raises PM2.5 significantly between October and January.",
        ],
    },
    "Delhi": {
        "headline": "Delhi Air-Quality Concerns",
        "issues": [
            "Delhi regularly records the world's highest urban PM2.5 concentrations, especially November–February when stubble burning and weather inversions combine.",
            "Vehicular emissions account for ~40% of PM2.5 according to IIT-Kanpur/TERI source-apportionment studies.",
            "Industrial units in Anand Vihar, Wazirpur and Bawana are significant local sources of SO₂ and heavy metals.",
        ],
    },
    "Mumbai": {
        "headline": "Mumbai Air-Quality Concerns",
        "issues": [
            "Construction dust from ongoing infrastructure projects (coastal road, metro lines) keeps PM10 elevated year-round.",
            "The Chembur–Trombay industrial belt (refineries, thermal plants) is a documented SO₂ hotspot.",
            "Sea-salt aerosols combined with vehicular NO₂ create secondary PM2.5 over the western suburbs.",
        ],
    },
    "Bengaluru": {
        "headline": "Bengaluru Air-Quality Concerns",
        "issues": [
            "Traffic congestion on ORR, Hosur Road and the NICE corridor drives high NO₂ and CO concentrations.",
            "Rapid construction growth has made PM10 the primary pollutant in central and eastern zones.",
            "Lake burning (waste dumped in Bellandur and Varthur lakes) periodically releases SO₂ and toxic organics.",
        ],
    },
}

RECOMMENDATIONS = {
    "Good": {
        "exposure": [
            "Air quality is good — outdoor activities are safe for everyone.",
            "A great day for morning jogs, cycling, or outdoor exercise.",
        ],
        "improvement": [
            "Maintain green covers and tree canopies in your neighbourhood.",
            "Prefer public transport or cycling to keep emissions low.",
        ],
    },
    "Moderate": {
        "exposure": [
            "Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
            "Keep windows open to ventilate, but watch forecasts if you have asthma.",
        ],
        "improvement": [
            "Reduce idling time of vehicles.",
            "Switch to cleaner cooking fuels if using solid biomass.",
        ],
    },
    "Unhealthy for Sensitive": {
        "exposure": [
            "People with heart or lung disease, children, and older adults should limit outdoor exertion.",
            "Wear an N95 mask when commuting in heavy traffic.",
            "Keep indoor air clean with houseplants or a HEPA purifier.",
        ],
        "improvement": [
            "Avoid open burning of waste or crop residue.",
            "Carpool or use metro/bus to reduce vehicle count.",
        ],
    },
    "Unhealthy": {
        "exposure": [
            "Everyone should reduce prolonged outdoor exertion.",
            "Sensitive groups should avoid all outdoor activities.",
            "Use N95/P100 masks outdoors; run an air purifier indoors.",
            "Stay hydrated and watch for respiratory symptoms.",
        ],
        "improvement": [
            "Report industrial or vehicular violations to the local SPCB.",
            "Participate in local clean-air campaigns and car-free days.",
        ],
    },
    "Very Unhealthy": {
        "exposure": [
            "Everyone should avoid prolonged outdoor exertion.",
            "Sensitive groups should remain indoors with windows closed.",
            "Use N95 masks if you must go outside.",
            "Consult a doctor if you experience throat irritation, coughing, or shortness of breath.",
        ],
        "improvement": [
            "Support and advocate for stricter industrial emission norms.",
            "Use electric vehicles or public transport.",
            "Plant dust-trapping vegetation near roads.",
        ],
    },
    "Hazardous": {
        "exposure": [
            "Health emergency — everyone should avoid all outdoor activities.",
            "Seal doors and windows; use a HEPA purifier on maximum setting.",
            "Wear an N95/P100 mask if you must go outside even briefly.",
            "Seek medical attention for any respiratory or cardiovascular symptoms immediately.",
        ],
        "improvement": [
            "Demand immediate action from local authorities for pollution sources.",
            "Report hazardous emissions to CPCB/SPCB emergency lines.",
            "Engage community health workers for vulnerable household checks.",
        ],
    },
}

GENERIC_ISSUES = {
    "headline": "Air-Quality Overview for This Location",
    "issues": [
        "No pre-compiled issue summary is available for this location.",
        "Common sources of urban air pollution include vehicles, industry, construction dust, and seasonal biomass burning.",
        "Refer to official CPCB and state pollution control board dashboards for ground-truth station data.",
    ],
}

# ---------------------------------------------------------------------------
# AQI helper
# ---------------------------------------------------------------------------

def get_aqi_category(aqi: float) -> tuple[str, str]:
    """Return (category_label, hex_color) for a US-AQI value."""
    if np.isnan(aqi):
        return "Unknown", "#aaaaaa"
    for threshold, label, color in AQI_CATEGORIES:
        if aqi <= threshold:
            return label, color
    return "Hazardous", "#7e0023"


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_air_quality(
    lat: float,
    lon: float,
    *,
    past_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch hourly air-quality data from Open-Meteo.

    Supply either `past_days` OR (`start_date` + `end_date`).
    Returns a DataFrame indexed by datetime with pollutant columns.
    Missing values remain as NaN (callers decide how to display them).
    """
    params: dict = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(POLLUTANTS),
        "timezone": "auto",
    }
    if past_days is not None:
        params["past_days"] = int(past_days)
    elif start_date and end_date:
        params["start_date"] = start_date
        params["end_date"] = end_date
    else:
        raise ValueError("Provide past_days or both start_date and end_date.")

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    times = pd.to_datetime(hourly.get("time", []))
    df = pd.DataFrame(index=times)
    df.index.name = "time"

    for col in POLLUTANTS:
        values = hourly.get(col)
        if values is not None:
            df[col] = pd.to_numeric(values, errors="coerce")
        else:
            df[col] = np.nan

    return df


# ---------------------------------------------------------------------------
# Trend statistics
# ---------------------------------------------------------------------------

def compute_trend_stats(df: pd.DataFrame) -> dict:
    """
    For each pollutant return: mean, peak, peak_time, trend_direction.
    trend_direction: 'rising', 'falling', or 'stable' based on linear slope.
    """
    stats: dict = {}
    for col in POLLUTANTS:
        if col not in df.columns or df[col].isna().all():
            stats[col] = {
                "mean": None,
                "peak": None,
                "peak_time": None,
                "trend": "unavailable",
            }
            continue
        series = df[col].dropna()
        mean_val = float(series.mean())
        peak_idx = series.idxmax()
        peak_val = float(series.max())

        # linear slope via numpy polyfit on integer index
        x = np.arange(len(series))
        slope = float(np.polyfit(x, series.values, 1)[0])
        threshold = mean_val * 0.005  # 0.5 % of mean
        if slope > threshold:
            trend = "rising [+]"
        elif slope < -threshold:
            trend = "falling [-]"
        else:
            trend = "stable [=]"

        stats[col] = {
            "mean": mean_val,
            "peak": peak_val,
            "peak_time": peak_idx,
            "trend": trend,
        }
    return stats


# ---------------------------------------------------------------------------
# ML prediction
# ---------------------------------------------------------------------------

def _build_features(series: pd.Series) -> pd.DataFrame:
    """Build feature matrix with hour, dayofweek and lag features."""
    df = series.to_frame(name="value")
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["lag_1h"] = df["value"].shift(1)
    df["lag_3h"] = df["value"].shift(3)
    df["lag_24h"] = df["value"].shift(24)
    return df.dropna()


MIN_ROWS_FOR_TRAINING = 48  # at least 48 hours of data


def predict_pollutant(
    series: pd.Series,
    horizon: int = 24,
) -> dict:
    """
    Train a RandomForestRegressor on the series and iteratively predict
    `horizon` hours ahead.

    Returns dict with keys:
      - 'forecast_df': DataFrame with columns ['time', 'predicted']
      - 'mae': float or None
      - 'error': str or None (human-readable problem)
    """
    series = series.dropna().copy()

    if len(series) < MIN_ROWS_FOR_TRAINING:
        return {
            "forecast_df": pd.DataFrame(columns=["time", "predicted"]),
            "mae": None,
            "error": f"Not enough data ({len(series)} points; need ≥ {MIN_ROWS_FOR_TRAINING}).",
        }

    # Time-based train/test split – last 24 h as holdout
    holdout = min(24, len(series) // 5)
    train_series = series.iloc[:-holdout]
    test_series = series.iloc[-holdout:]

    train_df = _build_features(train_series)
    feat_cols = ["hour", "dayofweek", "lag_1h", "lag_3h", "lag_24h"]

    X_train = train_df[feat_cols]
    y_train = train_df["value"]

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    # Evaluate on holdout (use test lags from the full series)
    test_df = _build_features(series.iloc[-(holdout + 24):]).iloc[-holdout:]
    if not test_df.empty:
        X_test = test_df[feat_cols]
        y_pred_test = model.predict(X_test)
        mae = float(mean_absolute_error(test_series.iloc[-len(y_pred_test):], y_pred_test))
    else:
        mae = None

    # Iterative future forecast
    history = list(series.values)
    history_times = list(series.index)
    last_time = history_times[-1]

    future_times = [last_time + timedelta(hours=i + 1) for i in range(horizon)]
    predictions = []

    for ft in future_times:
        lag_1 = history[-1] if len(history) >= 1 else 0.0
        lag_3 = history[-3] if len(history) >= 3 else 0.0
        lag_24 = history[-24] if len(history) >= 24 else 0.0

        X_pred = pd.DataFrame([{
            "hour": ft.hour,
            "dayofweek": ft.dayofweek,
            "lag_1h": lag_1,
            "lag_3h": lag_3,
            "lag_24h": lag_24,
        }])
        pred = float(model.predict(X_pred)[0])
        pred = max(pred, 0.0)  # pollutant concentrations can't be negative
        predictions.append(pred)
        history.append(pred)

    forecast_df = pd.DataFrame({"time": future_times, "predicted": predictions})
    return {"forecast_df": forecast_df, "mae": mae, "error": None}


def run_predictions(df: pd.DataFrame, horizon: int = 24) -> dict:
    """Run predict_pollutant for every pollutant and return results dict."""
    results = {}
    for col in POLLUTANTS:
        if col in df.columns:
            results[col] = predict_pollutant(df[col], horizon=horizon)
        else:
            results[col] = {
                "forecast_df": pd.DataFrame(columns=["time", "predicted"]),
                "mae": None,
                "error": "Column missing from data.",
            }
    return results


# ---------------------------------------------------------------------------
# Issues panel
# ---------------------------------------------------------------------------

def get_city_issues(city_name: Optional[str]) -> dict:
    """Return the issues dict for a preset city, or the generic dict."""
    if city_name and city_name in CITY_ISSUES:
        return CITY_ISSUES[city_name]
    return GENERIC_ISSUES


def get_recommendations(aqi: float) -> dict:
    """Return recommendations dict for the current AQI value."""
    category, _ = get_aqi_category(aqi)
    return RECOMMENDATIONS.get(category, RECOMMENDATIONS["Good"])


# ---------------------------------------------------------------------------
# Compliance check (NAAQS & WHO)
# ---------------------------------------------------------------------------

def compliance_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare 24-h (or 8-h for O₃) rolling means against India NAAQS and WHO limits.
    Returns a DataFrame with columns:
        pollutant | limit_type | limit_name | limit_value | pct_exceeding
    """
    rows = []
    for pollutant, limits in NAAQS_LIMITS.items():
        if pollutant not in df.columns or df[pollutant].isna().all():
            continue
        series = df[pollutant].dropna()

        for limit_name, limit_value in limits.items():
            # determine window
            if "8h" in limit_name:
                window = 8
                label = "8-h rolling mean"
            else:
                window = 24
                label = "24-h rolling mean"

            rolled = series.rolling(window=window, min_periods=max(1, window // 2)).mean()
            n_total = rolled.dropna().shape[0]
            if n_total == 0:
                pct = None
            else:
                n_exceed = (rolled.dropna() > limit_value).sum()
                pct = round(100.0 * n_exceed / n_total, 1)

            rows.append({
                "Pollutant": POLLUTANT_LABELS.get(pollutant, pollutant),
                "Averaging Period": label,
                "Standard": limit_name.replace("_", " "),
                "Limit (µg/m³)": limit_value,
                "% Hours Exceeding": pct if pct is not None else "N/A",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reverse geocoding (Nominatim)
# ---------------------------------------------------------------------------

def reverse_geocode(lat: float, lon: float) -> Optional[str]:
    """
    Use OSM Nominatim to get a human-readable place name for the coordinates.
    Returns None on failure.
    """
    try:
        headers = {"User-Agent": "AirQualityMonitorApp/1.0 (educational project)"}
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 10},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        address = data.get("address", {})
        # pick best available granularity
        place = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or address.get("state_district")
            or data.get("display_name", "")
        )
        return place or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Non-preset city issues  (web-search based summary fallback)
# ---------------------------------------------------------------------------

def get_location_issues(lat: float, lon: float, city_name: Optional[str]) -> dict:
    """
    For non-preset cities, try Nominatim reverse-geocoding, then return a
    structured issues dict with the place name and a note that the user
    should check official sources.
    Falls back to the generic dict on any failure.
    """
    if city_name and city_name in CITY_ISSUES:
        return CITY_ISSUES[city_name]

    place = reverse_geocode(lat, lon)
    if place:
        return {
            "headline": f"Air-Quality Overview: {place}",
            "issues": [
                f"Auto-retrieved location: **{place}** (via OpenStreetMap Nominatim).",
                "No pre-compiled documented issues exist for this location in our database.",
                "Common urban air-quality concerns include vehicular exhaust, industrial emissions, "
                "construction dust, and seasonal biomass burning.",
                "⚠️ This is an auto-retrieved summary. Please verify with official sources.",
            ],
            "place": place,
            "auto": True,
        }
    return {**GENERIC_ISSUES, "auto": False}


# ---------------------------------------------------------------------------
# Date-range validation
# ---------------------------------------------------------------------------

def validate_date_range(start: date, end: date) -> Optional[str]:
    """
    Return an error message string if the range is invalid, else None.
    Open-Meteo supports up to ~3 months of historical data for this endpoint.
    """
    today = date.today()
    if start > end:
        return "Start date must be before or equal to end date."
    if end > today:
        return "End date cannot be in the future."
    if start > today:
        return "Start date cannot be in the future."
    if (end - start).days > 92:
        return "Date range cannot exceed 92 days (Open-Meteo limit for this endpoint)."
    if (today - end).days > 365 * 2:
        return "Open-Meteo free tier typically covers the last 2 years. Try a more recent range."
    return None
