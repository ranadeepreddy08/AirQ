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


# ---------------------------------------------------------------------------
# Chatbot helpers: build_context + answer_question + multi-location support
# ---------------------------------------------------------------------------

def build_context(
    lat: float,
    lon: float,
    df: pd.DataFrame,
    trend_stats: dict,
    issues: dict,
    compliance_df: pd.DataFrame,
    recs: dict,
    city_name: Optional[str] = None,
    past_days: Optional[int] = 7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    horizon: int = 24,
    forecast_results: Optional[dict] = None,
) -> str:
    """
    Build a rich plain-text context summary for the AI chatbot.
    Includes: location, time window, AQI, per-pollutant latest/mean/peak/trend,
    forecast summary with MAE, compliance, issues, recommendations.
    """
    lines: list[str] = []

    # ── Location & time window ──────────────────────────────────────────────
    place = city_name or f"{lat:.4f}°N, {lon:.4f}°E"
    lines.append(f"=== AIR QUALITY REPORT: {place.upper()} ===")
    lines.append(f"Coordinates: {lat:.4f}°N, {lon:.4f}°E")
    if start_date and end_date:
        lines.append(f"Time window: {start_date} to {end_date}")
    elif past_days:
        lines.append(f"Time window: last {past_days} day(s)")
    lines.append(f"Forecast horizon: {horizon} hours")
    lines.append(f"Data rows loaded: {len(df)}")

    # ── Current AQI ────────────────────────────────────────────────────────
    aqi_series = df["us_aqi"].dropna() if "us_aqi" in df.columns else pd.Series([], dtype=float)
    if not aqi_series.empty:
        aqi_val = float(aqi_series.iloc[-1])
        cat, _ = get_aqi_category(aqi_val)
        lines.append(f"\nCurrent US AQI: {aqi_val:.0f} ({cat})")
    else:
        lines.append("\nCurrent US AQI: Not available")

    # ── Per-pollutant: latest value, mean, peak, trend ─────────────────────
    lines.append("\nPollutant readings (latest | mean | peak | trend):")
    for pol in POLLUTANTS:
        if pol == "us_aqi":
            continue
        label = POLLUTANT_LABELS.get(pol, pol)
        s = trend_stats.get(pol, {})
        col = df[pol].dropna() if pol in df.columns else pd.Series([], dtype=float)
        latest = f"{float(col.iloc[-1]):.1f}" if not col.empty else "N/A"
        mean   = f"{s['mean']:.1f}"   if s.get("mean") is not None else "N/A"
        peak   = f"{s['peak']:.1f}"   if s.get("peak") is not None else "N/A"
        trend  = s.get("trend", "N/A")
        if mean == "N/A":
            lines.append(f"  {label}: Not available")
        else:
            lines.append(f"  {label}: latest={latest}, mean={mean}, peak={peak}, trend={trend}")

    # ── Forecast summary ────────────────────────────────────────────────────
    lines.append(f"\nForecast summary ({horizon}h RandomForest, model-based, indicative):")
    if forecast_results:
        for pol, res in forecast_results.items():
            label = POLLUTANT_LABELS.get(pol, pol)
            if res.get("error"):
                lines.append(f"  {label}: {res['error']}")
                continue
            mae = res.get("mae")
            fdf = res.get("forecast_df")
            mae_str = f"MAE={mae:.2f}" if mae is not None else "MAE=N/A"
            if fdf is not None and not fdf.empty:
                first_v = fdf["predicted"].iloc[0]
                last_v  = fdf["predicted"].iloc[-1]
                lines.append(
                    f"  {label}: first forecast={first_v:.1f}, "
                    f"end of horizon={last_v:.1f}, {mae_str}"
                )
            else:
                lines.append(f"  {label}: no forecast available, {mae_str}")
    else:
        lines.append("  (Forecast not pre-computed for this context)")

    # ── Compliance ─────────────────────────────────────────────────────────
    lines.append("\nCompliance vs NAAQS / WHO (% of rolling-mean hours exceeding limit):")
    if compliance_df is not None and not compliance_df.empty:
        for _, row in compliance_df.iterrows():
            pct = row.get("% Hours Exceeding", "N/A")
            lines.append(
                f"  {row['Pollutant']} ({row['Standard']}): "
                f"{pct}% exceed {row['Limit (µg/m³)']} µg/m³"
            )
    else:
        lines.append("  Compliance data not available.")

    # ── Issues ─────────────────────────────────────────────────────────────
    lines.append(f"\nDocumented issues — {issues.get('headline', 'General')}:")
    for iss in issues.get("issues", [])[:5]:
        lines.append(f"  - {iss}")

    # ── Recommendations ────────────────────────────────────────────────────
    aqi_cat_for_rec = "unknown"
    if not aqi_series.empty:
        aqi_cat_for_rec, _ = get_aqi_category(float(aqi_series.iloc[-1]))
    lines.append(f"\nRecommendations for AQI category '{aqi_cat_for_rec}':")
    for tip in recs.get("exposure", [])[:4]:
        lines.append(f"  Exposure: {tip}")
    for tip in recs.get("improvement", [])[:3]:
        lines.append(f"  Improvement: {tip}")

    lines.append(
        "\n[Data source: Open-Meteo Air Quality API — model-based reanalysis, "
        "NOT certified ground-sensor readings. Forecast is indicative only.]"
    )
    return "\n".join(lines)


def geocode_place(place_name: str) -> Optional[tuple[float, float, str]]:
    """
    Geocode a place name using Open-Meteo Geocoding API (no key required).
    Returns (lat, lon, display_name) or None on failure.
    """
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": place_name, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return None
        r = results[0]
        name = r.get("name", place_name)
        country = r.get("country", "")
        display = f"{name}, {country}" if country else name
        return float(r["latitude"]), float(r["longitude"]), display
    except Exception:
        return None


def build_extra_location_context(
    place_name: str,
    past_days: int = 7,
) -> str:
    """
    Fetch and summarise air quality for an extra location (for comparison).
    Returns a labelled context block or an error string.
    """
    geo = geocode_place(place_name)
    if geo is None:
        return f"\n[{place_name.upper()}]: Could not geocode this location."
    lat, lon, display = geo
    try:
        df = fetch_air_quality(lat, lon, past_days=past_days)
    except Exception as exc:
        return f"\n[{display.upper()}]: Data fetch failed — {exc}"
    if df.empty:
        return f"\n[{display.upper()}]: No data returned."

    trend = compute_trend_stats(df)
    lines = [f"\n=== EXTRA LOCATION: {display.upper()} ({lat:.4f}°N, {lon:.4f}°E) ==="]
    aqi_series = df["us_aqi"].dropna() if "us_aqi" in df.columns else pd.Series([], dtype=float)
    if not aqi_series.empty:
        aqi_val = float(aqi_series.iloc[-1])
        cat, _ = get_aqi_category(aqi_val)
        lines.append(f"  Current US AQI: {aqi_val:.0f} ({cat})")
    else:
        lines.append("  Current US AQI: Not available")
    for pol in POLLUTANTS:
        if pol == "us_aqi":
            continue
        label = POLLUTANT_LABELS.get(pol, pol)
        s = trend.get(pol, {})
        col = df[pol].dropna() if pol in df.columns else pd.Series([], dtype=float)
        latest = f"{float(col.iloc[-1]):.1f}" if not col.empty else "N/A"
        mean   = f"{s['mean']:.1f}" if s.get("mean") is not None else "N/A"
        lines.append(f"  {label}: latest={latest}, mean={mean}")
    lines.append("  [Open-Meteo model-based data]")
    return "\n".join(lines)


def _extract_place_names(question: str) -> list[str]:
    """
    Heuristic extraction of place names from a question.
    Looks for known cities, then tries patterns like 'in X', 'for X', 'at X'.
    Returns up to 3 unique names (excluding the primary location).
    """
    import re
    # Known Indian cities + common international ones
    known = [
        "Delhi", "Mumbai", "Chennai", "Bengaluru", "Bangalore",
        "Hyderabad", "Kolkata", "Pune", "Ahmedabad", "Surat",
        "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Patna",
        "London", "New York", "Beijing", "Shanghai", "Tokyo",
        "Los Angeles", "Paris", "Dubai", "Singapore",
    ]
    found = []
    q_lower = question.lower()
    for city in known:
        if city.lower() in q_lower and city not in found:
            found.append(city)

    # Pattern: "in <Word>" or "for <Word>" or "at <Word>" or "to <Word>"
    pattern_matches = re.findall(
        r'\b(?:in|for|at|to|compare|vs\.?|versus|and)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        question
    )
    for m in pattern_matches:
        if m not in found:
            found.append(m)

    return found[:3]


# ── System prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are **AirQ Guide**, a friendly and knowledgeable air quality instructor — \
think of a skilled weather presenter who also explains science in plain everyday language.

## Your job
Answer questions about air quality using the structured context data provided. \
You may also answer general knowledge questions about air pollution topics \
(e.g., what is smog, why is winter worse for air quality) from your own knowledge.

## Response style
1. **Plain-language first**: assume the reader has no science background. \
   Use everyday comparisons (e.g., "PM2.5 particles are 30× smaller than a human hair").
2. **Technical details second** (optional): only add a short "📊 Technical details" section \
   when it genuinely helps (pollutant values with units, AQI thresholds, trend, MAE). \
   Skip for simple factual questions.
3. Keep answers short and friendly — short paragraphs or bullets. \
   Ask a brief follow-up only when the request is genuinely unclear.
4. Always **name the location** when discussing specific data. \
   Never invent numbers — if data is missing say so explicitly.

## Specific behaviours
- **Comparisons**: give a clear verdict (which is better and by how much), \
  then a compact markdown table of AQI and key pollutants for each location.
- **Travel / safety**: give a clear recommendation (✅ fine / ⚠️ take care / 🚫 avoid), \
  based on current AQI and forecast. Include practical steps (mask type, best time of day, \
  indoor alternatives). Tailor advice for children, older adults, and people with \
  asthma or heart conditions. End with one line: \
  "⚕️ Not medical advice — check local official sources (CPCB, WHO)."
- **Website explanations**: when asked about any item on the page \
  (AQI, PM2.5, PM10, NO₂, SO₂, O₃, MAE, forecast, compliance check, trend), \
  explain it simply and mention that Open-Meteo data is model-based and the forecast is indicative.
- **Fallback**: for completely off-topic questions, politely say you specialise in \
  air quality and ask what the user would like to know about it.

## Data disclaimer
Always mention, at least once per session, that values come from Open-Meteo \
model-based reanalysis (not certified ground sensors) and that forecasts are indicative.
"""


def answer_question(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> tuple[str, bool]:
    """
    Answer an air-quality question using Gemini.
    Returns (answer_text, used_gemini:bool).
    Falls back to rule-based answers if the API key is missing or call fails.
    Gemini errors are printed to the terminal.
    history: list of {"role": "user"|"model", "content": str}
    """
    import os
    import re as _re

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    _fallback_reason = ""

    if api_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                system_instruction=_SYSTEM_PROMPT,
            )

            # Build Gemini-format history (last 10 turns)
            chat_history = []
            for msg in (history or [])[-10:]:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role in ("user", "model") and text:
                    chat_history.append({"role": role, "parts": [text]})

            chat = model.start_chat(history=chat_history)
            prompt = f"Context data:\n{context}\n\nQuestion: {question}"
            response = chat.send_message(prompt)
            return response.text.strip(), True

        except Exception as exc:
            _fallback_reason = str(exc)
            print(f"[AirQ Chat] Gemini error: {_fallback_reason}", flush=True)
    else:
        _fallback_reason = "GEMINI_API_KEY not set"
        print(f"[AirQ Chat] {_fallback_reason}", flush=True)

    # ── Rule-based fallback ────────────────────────────────────────────────
    answer = _rule_based_answer(question, context, _fallback_reason)
    return answer, False


def _rule_based_answer(question: str, context: str, fallback_reason: str) -> str:
    """Keyword-driven fallback when Gemini is unavailable."""
    import re
    NOTICE = (
        "\n\n---\n"
        f"⚠️ **Rule-based fallback** — Gemini API unavailable ({fallback_reason}). "
        "Add your `GEMINI_API_KEY` to `.env` for full AI responses."
    )

    q   = question.lower()
    ctx = context.lower()

    # Extract AQI from context
    aqi_match = re.search(r"current us aqi:\s*([0-9.]+)\s*\(([^)]+)\)", ctx)
    aqi_val   = float(aqi_match.group(1)) if aqi_match else None
    aqi_cat   = aqi_match.group(2).strip() if aqi_match else "unknown"

    # Extract place name from context header
    place_match = re.search(r"=== air quality report: ([^=]+) ===", ctx)
    place_name  = place_match.group(1).strip().title() if place_match else "the selected location"

    if any(w in q for w in ("aqi", "air quality", "overall", "index", "level")):
        if aqi_val is not None:
            return (
                f"**{place_name}** — Current US AQI: **{aqi_val:.0f}** ({aqi_cat.title()})\n\n"
                + _fallback_advice(aqi_cat)
                + NOTICE
            )
        return "Current AQI data is not available in the loaded dataset." + NOTICE

    if any(w in q for w in ("pm2.5", "pm2_5", "pm 2.5", "fine particle")):
        return _extract_pollutant_answer("pm2.5 (ug/m3)", ctx, "PM2.5") + NOTICE

    if any(w in q for w in ("pm10", "pm 10", "coarse particle")):
        return _extract_pollutant_answer("pm10 (ug/m3)", ctx, "PM10") + NOTICE

    if any(w in q for w in ("no2", "nitrogen dioxide", "no₂")):
        return _extract_pollutant_answer("no2 (ug/m3)", ctx, "NO₂") + NOTICE

    if any(w in q for w in ("so2", "sulphur", "sulfur")):
        return _extract_pollutant_answer("so2 (ug/m3)", ctx, "SO₂") + NOTICE

    if any(w in q for w in ("ozone", "o3", "o₃")):
        return _extract_pollutant_answer("o3 (ug/m3)", ctx, "O₃") + NOTICE

    if any(w in q for w in ("safe", "outdoor", "exercise", "walk", "run", "jog", "travel")):
        if aqi_val is not None:
            return _safety_advice(aqi_val, aqi_cat) + NOTICE
        return "AQI data is unavailable — load data first." + NOTICE

    if any(w in q for w in ("mask", "n95", "filter")):
        if aqi_val and aqi_val > 100:
            return (
                "Given the current AQI, wearing an **N95 or KN95 mask** outdoors is advisable, "
                "especially for sensitive groups.\n\n"
                "⚕️ Not medical advice — consult local health authorities."
                + NOTICE
            )
        return (
            "Current AQI is relatively low; a mask may not be essential, "
            "but sensitive groups may still benefit.\n\n"
            "⚕️ Not medical advice."
            + NOTICE
        )

    if any(w in q for w in ("recommend", "tip", "advice", "what should")):
        idx = ctx.find("recommendations")
        if idx >= 0:
            snippet = context[idx: idx + 400]
            return f"**Recommendations:**\n\n{snippet}\n\n⚕️ Not medical advice." + NOTICE
        return "Load data first to get personalised recommendations." + NOTICE

    if any(w in q for w in ("compliance", "exceed", "limit", "naaqs", "who")):
        idx = ctx.find("compliance")
        if idx >= 0:
            snippet = context[idx: idx + 500]
            return f"**Compliance summary:**\n\n{snippet}\n\n*(Indicative only; verify with CPCB.)*" + NOTICE
        return "Compliance data is not available." + NOTICE

    if any(w in q for w in ("forecast", "predict", "future", "tomorrow", "next")):
        idx = ctx.find("forecast summary")
        if idx >= 0:
            snippet = context[idx: idx + 400]
            return f"**Forecast:**\n\n{snippet}\n\n*(Model-based, indicative only.)*" + NOTICE
        return "Forecast data is not in the current context." + NOTICE

    return (
        "I can answer questions about the air quality data on this page — "
        "current AQI, pollutant levels, health advice, compliance, or forecast. "
        "What would you like to know?"
        + NOTICE
    )


def _fallback_advice(category: str) -> str:
    advice = {
        "good": "Air quality is satisfactory — outdoor activities are safe for most people. 🟢",
        "moderate": "Mostly fine, but unusually sensitive individuals may want to limit prolonged outdoor exertion. 🟡",
        "unhealthy for sensitive": (
            "Sensitive groups (elderly, children, asthma/heart conditions) should reduce outdoor time. "
            "Healthy adults are likely fine for brief activity. 🟠"
        ),
        "unhealthy": (
            "Everyone should reduce prolonged outdoor exertion. "
            "Sensitive groups should avoid outdoor activity. 🔴\n\n"
            "⚕️ Not medical advice — check CPCB or WHO sources."
        ),
        "very unhealthy": (
            "Health alert: everyone should avoid outdoor exertion. "
            "Stay indoors with air purification if possible. 🟣\n\n"
            "⚕️ Not medical advice."
        ),
        "hazardous": (
            "Health emergency: avoid outdoor activity entirely. "
            "Keep windows closed. Use N95 masks if you must go out. ⚫\n\n"
            "⚕️ Not medical advice."
        ),
    }
    return advice.get(category.lower(), "Check local authorities for guidance.")


def _safety_advice(aqi_val: float, aqi_cat: str) -> str:
    if aqi_val <= 50:
        return f"✅ AQI {aqi_val:.0f} (Good) — outdoor exercise is safe for everyone."
    if aqi_val <= 100:
        return (
            f"⚠️ AQI {aqi_val:.0f} (Moderate) — most people can exercise outdoors; "
            "very sensitive individuals may want to limit prolonged exertion."
        )
    if aqi_val <= 150:
        return (
            f"⚠️ AQI {aqi_val:.0f} (Unhealthy for Sensitive Groups) — "
            "sensitive groups should limit outdoor exertion; healthy adults can exercise briefly."
        )
    if aqi_val <= 200:
        return (
            f"🚫 AQI {aqi_val:.0f} (Unhealthy) — everyone should reduce prolonged outdoor activity.\n\n"
            "⚕️ Not medical advice — consult local health authorities."
        )
    return (
        f"🚫 AQI {aqi_val:.0f} ({aqi_cat.title()}) — outdoor activity is not recommended. "
        "Stay indoors.\n\n⚕️ Not medical advice."
    )


def _extract_pollutant_answer(label_fragment: str, ctx_lower: str, display_name: str) -> str:
    import re
    pattern = rf"{re.escape(label_fragment)}:\s*latest=([0-9.NA/]+),\s*mean=([0-9.NA/]+),\s*peak=([0-9.NA/]+),\s*trend=([^\n]+)"
    m = re.search(pattern, ctx_lower)
    if m:
        latest, mean, peak, trend = m.group(1), m.group(2), m.group(3), m.group(4).strip()
        return (
            f"**{display_name}**: latest {latest} µg/m³, mean {mean} µg/m³, "
            f"peak {peak} µg/m³, trend: {trend}.\n\n"
            "*(Data is Open-Meteo model-based, not from certified stations.)*"
        )
    return f"**{display_name}** data is not available in the current dataset."


def build_context(
    lat: float,
    lon: float,
    df: pd.DataFrame,
    trend_stats: dict,
    issues: dict,
    compliance_df: pd.DataFrame,
    recs: dict,
    city_name: Optional[str] = None,
) -> str:
    """
    Build a compact plain-text context summary for the AI chatbot.
    Keeps token count low while covering all data the user sees.
    """
    lines: list[str] = []

    # Location
    loc = city_name or f"{lat:.4f}°N, {lon:.4f}°E"
    lines.append(f"=== AIR QUALITY CONTEXT FOR {loc.upper()} ===")

    # Current AQI
    aqi_series = df["us_aqi"].dropna() if "us_aqi" in df.columns else pd.Series([], dtype=float)
    if not aqi_series.empty:
        aqi_val = float(aqi_series.iloc[-1])
        cat, _ = get_aqi_category(aqi_val)
        lines.append(f"Current US AQI: {aqi_val:.0f} ({cat})")
    else:
        lines.append("Current US AQI: Not available")

    # Pollutant means and trends
    lines.append("\nPollutant statistics (mean / peak / trend):")
    for pol in POLLUTANTS:
        if pol == "us_aqi":
            continue
        s = trend_stats.get(pol, {})
        label = POLLUTANT_LABELS.get(pol, pol)
        mean = s.get("mean")
        peak = s.get("peak")
        trend = s.get("trend", "unavailable")
        if mean is not None:
            lines.append(f"  {label}: mean={mean:.1f}, peak={peak:.1f}, trend={trend}")
        else:
            lines.append(f"  {label}: Not available")

    # Forecast summary (latest 3 predicted values from run_predictions if passed)
    # We pass recs which contains AQI category — forecast chart is per-poll, skipped here for brevity

    # Compliance summary
    lines.append("\nCompliance vs NAAQS / WHO (% hours exceeding):")
    if compliance_df is not None and not compliance_df.empty:
        for _, row in compliance_df.iterrows():
            pct = row.get("% Hours Exceeding", "N/A")
            lines.append(
                f"  {row['Pollutant']} ({row['Standard']}): "
                f"{pct}% hours exceed limit of {row['Limit (µg/m³)']} µg/m³"
            )
    else:
        lines.append("  Compliance data not available.")

    # Issues
    lines.append(f"\nDocumented issues ({issues.get('headline', '')}):")
    for iss in issues.get("issues", [])[:5]:
        lines.append(f"  - {iss}")

    # Recommendations
    lines.append(f"\nRecommendations (AQI category: {recs.get('category', 'unknown')}):")
    for tip in recs.get("exposure", [])[:4]:
        lines.append(f"  Exposure: {tip}")
    for tip in recs.get("improvement", [])[:3]:
        lines.append(f"  Improvement: {tip}")

    lines.append("\n[Data: Open-Meteo model-based reanalysis, not certified station readings]")
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are an air quality assistant. Your job is to answer questions ONLY about air quality, \
pollution levels, health effects of pollutants, and practical advice based on the data \
provided in the context below. Do not answer questions unrelated to air quality.

Rules:
- Base all answers on the supplied context data. If data is missing, say so explicitly.
- Give practical, actionable advice.
- For health guidance, always add: "Note: this is general air quality information, \
not medical advice. Consult a doctor for personal health decisions."
- Be concise (3–6 sentences unless more detail is requested).
- If asked about a pollutant not in the context, say the data is unavailable.
"""


def answer_question(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> str:
    """
    Answer an air-quality question using Gemini 3.8 Flash.
    Falls back to simple rule-based answers if the API key is missing or call fails.
    history: list of {"role": "user"|"model", "parts": [str]} for multi-turn.
    """
    import os

    # Load .env if present (python-dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if api_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-3.8-flash",
                system_instruction=_SYSTEM_PROMPT,
            )

            # Build chat history
            chat_history = []
            for msg in (history or []):
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role in ("user", "model") and text:
                    chat_history.append({"role": role, "parts": [text]})

            chat = model.start_chat(history=chat_history)
            prompt = f"Context data:\n{context}\n\nQuestion: {question}"
            response = chat.send_message(prompt)
            return response.text.strip()

        except Exception as exc:
            # Fall through to rule-based fallback
            _fallback_reason = str(exc)
    else:
        _fallback_reason = "GEMINI_API_KEY not set"

    # ── Rule-based fallback ────────────────────────────────────────────────
    q = question.lower()
    ctx = context.lower()

    # Extract current AQI from context
    import re
    aqi_match = re.search(r"current us aqi:\s*([0-9.]+)\s*\(([^)]+)\)", ctx)
    aqi_val = float(aqi_match.group(1)) if aqi_match else None
    aqi_cat = aqi_match.group(2).strip() if aqi_match else "unknown"

    if any(w in q for w in ("aqi", "air quality", "overall", "index", "level")):
        if aqi_val is not None:
            return (
                f"The current US AQI is {aqi_val:.0f} ({aqi_cat}). "
                + _fallback_advice(aqi_cat)
            )
        return "Current AQI data is not available in the loaded dataset."

    if any(w in q for w in ("pm2.5", "pm2_5", "pm 2.5", "fine particle")):
        return _extract_pollutant_answer("pm2.5 (ug/m3)", ctx, "PM2.5")

    if any(w in q for w in ("pm10", "pm 10", "coarse particle")):
        return _extract_pollutant_answer("pm10 (ug/m3)", ctx, "PM10")

    if any(w in q for w in ("no2", "nitrogen dioxide", "no₂")):
        return _extract_pollutant_answer("no2 (ug/m3)", ctx, "NO₂")

    if any(w in q for w in ("so2", "sulphur", "sulfur")):
        return _extract_pollutant_answer("so2 (ug/m3)", ctx, "SO₂")

    if any(w in q for w in ("ozone", "o3", "o₃")):
        return _extract_pollutant_answer("o3 (ug/m3)", ctx, "O₃")

    if any(w in q for w in ("safe", "outdoor", "exercise", "walk", "run", "jog")):
        if aqi_val is not None:
            return _safety_advice(aqi_val, aqi_cat)
        return "AQI data is unavailable. Please try again after loading data."

    if any(w in q for w in ("mask", "n95", "filter")):
        if aqi_val and aqi_val > 100:
            return "Given the current AQI level, wearing an N95 or KN95 mask outdoors is advisable, especially for sensitive groups. Note: this is general air quality information, not medical advice."
        return "Current AQI levels are relatively low; a mask may not be necessary, but sensitive groups may still benefit. Note: not medical advice — consult a doctor."

    if any(w in q for w in ("recommend", "tip", "advice", "what should")):
        # Find recommendations section in context
        idx = ctx.find("recommendations")
        if idx >= 0:
            snippet = context[idx: idx + 400]
            return f"Based on the current data:\n{snippet}\n\nNote: not medical advice."
        return "Load data first to get personalised recommendations."

    if any(w in q for w in ("compliance", "exceed", "limit", "naaqs", "who")):
        idx = ctx.find("compliance")
        if idx >= 0:
            snippet = context[idx: idx + 400]
            return f"Compliance summary:\n{snippet}\n\n(Indicative only; verify with CPCB.)"
        return "Compliance data is not available."

    # Generic fallback
    return (
        "I can answer questions about the air quality data shown on this page — "
        "current AQI, pollutant levels, health advice, compliance, or recommendations. "
        "Please ask something specific about air quality. "
        f"(Note: Gemini API unavailable — {_fallback_reason})"
    )


def _fallback_advice(category: str) -> str:
    advice = {
        "good": "Air quality is satisfactory; outdoor activities are safe for most people.",
        "moderate": "Unusually sensitive individuals should consider limiting prolonged outdoor exertion.",
        "unhealthy for sensitive": "Sensitive groups (elderly, children, those with heart/lung conditions) should reduce outdoor activity.",
        "unhealthy": "Everyone should reduce prolonged or heavy outdoor exertion. Sensitive groups should avoid outdoor activity.",
        "very unhealthy": "Health alert: everyone should avoid outdoor exertion. Stay indoors with air purification if possible.",
        "hazardous": "Health emergency: everyone should avoid outdoor activity entirely. Keep windows closed.",
    }
    return advice.get(category.lower(), "Check local authorities for guidance.")


def _safety_advice(aqi_val: float, aqi_cat: str) -> str:
    if aqi_val <= 50:
        return f"AQI is {aqi_val:.0f} (Good) — outdoor exercise is safe for everyone."
    if aqi_val <= 100:
        return f"AQI is {aqi_val:.0f} (Moderate) — most people can exercise outdoors; very sensitive individuals may want to limit prolonged exertion."
    if aqi_val <= 150:
        return f"AQI is {aqi_val:.0f} (Unhealthy for Sensitive Groups) — sensitive groups should limit outdoor exertion. Healthy adults can still exercise briefly."
    if aqi_val <= 200:
        return f"AQI is {aqi_val:.0f} (Unhealthy) — everyone should reduce prolonged outdoor activity. Note: not medical advice."
    return f"AQI is {aqi_val:.0f} ({aqi_cat}) — outdoor activity is not recommended. Stay indoors. Note: not medical advice."


def _extract_pollutant_answer(label_fragment: str, ctx_lower: str, display_name: str) -> str:
    import re
    pattern = rf"{re.escape(label_fragment)}:\s*mean=([0-9.]+),\s*peak=([0-9.]+),\s*trend=([^\n]+)"
    m = re.search(pattern, ctx_lower)
    if m:
        mean, peak, trend = m.group(1), m.group(2), m.group(3).strip()
        return (
            f"{display_name}: mean {mean} µg/m³, peak {peak} µg/m³, trend: {trend}. "
            "Refer to WHO guidelines for health context. Note: data is model-based, not from certified stations."
        )
    return f"{display_name} data is not available in the current dataset."
