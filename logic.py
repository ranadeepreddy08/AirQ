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
# Chatbot: context, multi-location, Gemini call, rule-based fallback
# ---------------------------------------------------------------------------
import os
import re
import time

_GEO_CACHE: dict = {}
_EXTRA_CACHE: dict = {}


def place_name_for(lat: float, lon: float, city: Optional[str] = None) -> Optional[str]:
    """Preset city name, else cached reverse-geocoded place name (or None)."""
    if city and city in CITY_ISSUES:
        return city
    key = (round(lat, 2), round(lon, 2))
    if key in _GEO_CACHE:
        return _GEO_CACHE[key]
    name = reverse_geocode(lat, lon)
    if name:
        _GEO_CACHE[key] = name
    return name


def geocode_place(place_name: str) -> Optional[tuple[float, float, str]]:
    """Open-Meteo geocoding (no key). Returns (lat, lon, display_name) or None."""
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
        country = r.get("country", "")
        display = f"{r.get('name', place_name)}, {country}" if country else r.get("name", place_name)
        return float(r["latitude"]), float(r["longitude"]), display
    except Exception:
        return None


# ── Place-name extraction ───────────────────────────────────────────────────

_KNOWN_PLACES = [
    "Delhi", "Mumbai", "Chennai", "Bengaluru", "Bangalore", "Hyderabad", "Kolkata",
    "Pune", "Ahmedabad", "Surat", "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Patna",
    "Coimbatore", "Madurai", "Kochi", "Goa", "London", "New York", "Beijing",
    "Shanghai", "Tokyo", "Los Angeles", "Paris", "Dubai", "Singapore", "Bangkok", "Sydney",
]
_ALIASES = {"bangalore": "bengaluru"}
_NOT_PLACES = {
    "Air", "Quality", "The", "This", "That", "What", "Why", "How", "Is", "It", "Explain",
    "Compare", "Please", "Today", "Tomorrow", "Winter", "Summer", "Monsoon", "Autumn",
    "Spring", "Morning", "Evening", "Night", "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December", "India", "AQI",
}


def _canon(name: str) -> str:
    n = name.strip().lower()
    return _ALIASES.get(n, n)


def extract_place_names(question: str, exclude=()) -> list[str]:
    """Up to 3 place names mentioned in the question, excluding the primary location."""
    q = question.lower()
    cands: list[str] = []
    for p in _KNOWN_PLACES:
        if re.search(rf"\b{re.escape(p.lower())}\b", q):
            cands.append(p)

    # Generic "in/to/for X" capture only for travel/compare style questions,
    # so ordinary questions don't trigger random geocoding lookups.
    if re.search(r"\b(compare|comparison|versus|vs|travel|trip|visit|visiting|going to|fly|flying|move|moving|relocate)\b", q):
        for m in re.findall(
            r"\b(?:in|to|for|at|vs\.?|versus|and|than|or|with)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)",
            question,
        ):
            if m.split()[0] not in _NOT_PLACES:
                cands.append(m)

    ex = [_canon(e) for e in exclude if e]
    out, seen = [], set()
    for c in cands:
        k = _canon(c)
        if k in seen or any(k in e or e in k for e in ex):
            continue
        seen.add(k)
        out.append(c if k == c.lower() else k.title())
    return out[:3]


def build_extra_location_context(place_name: str, past_days: int = 7) -> str:
    """Fetch and summarise air quality (plus a 24h AQI forecast) for another place."""
    key = (place_name.lower(), past_days)
    hit = _EXTRA_CACHE.get(key)
    if hit and time.time() - hit[0] < 900:
        return hit[1]

    geo = geocode_place(place_name)
    if geo is None:
        return f"\n[{place_name.upper()}]: Could not find this location, so no data is available for it."
    lat, lon, display = geo
    try:
        df = fetch_air_quality(lat, lon, past_days=past_days)
    except Exception as exc:
        return f"\n[{display.upper()}]: Data fetch failed ({exc})."
    if df.empty:
        return f"\n[{display.upper()}]: No data returned."

    trend = compute_trend_stats(df)
    L = [f"\n=== EXTRA LOCATION: {display.upper()} (lat {lat:.4f}, lon {lon:.4f}) ==="]
    aqi_s = df["us_aqi"].dropna() if "us_aqi" in df.columns else pd.Series([], dtype=float)
    if not aqi_s.empty:
        v = float(aqi_s.iloc[-1])
        L.append(f"  Current US AQI: {v:.0f} ({get_aqi_category(v)[0]})")
        res = predict_pollutant(df["us_aqi"], horizon=24)
        fdf = res["forecast_df"]
        if not fdf.empty:
            L.append(
                f"  US AQI forecast next 24h: start={fdf['predicted'].iloc[0]:.0f}, "
                f"peak={fdf['predicted'].max():.0f}, end={fdf['predicted'].iloc[-1]:.0f} "
                "(model-based, indicative)"
            )
    else:
        L.append("  Current US AQI: Not available")
    for pol in POLLUTANTS:
        if pol == "us_aqi":
            continue
        col = df[pol].dropna() if pol in df.columns else pd.Series([], dtype=float)
        s = trend.get(pol, {})
        latest = f"{float(col.iloc[-1]):.1f}" if not col.empty else "N/A"
        mean = f"{s['mean']:.1f}" if s.get("mean") is not None else "N/A"
        L.append(f"  {POLLUTANT_LABELS[pol]}: latest={latest}, mean={mean}")
    L.append("  [Open-Meteo model-based data]")
    text = "\n".join(L)
    _EXTRA_CACHE[key] = (time.time(), text)
    return text


def build_extra_contexts(question: str, exclude=(), past_days: int = 7) -> str:
    return "".join(
        build_extra_location_context(n, past_days)
        for n in extract_place_names(question, exclude)
    )


# ── Primary context ─────────────────────────────────────────────────────────

def build_context(
    lat: float,
    lon: float,
    df: pd.DataFrame,
    trend_stats: dict,
    issues: dict,
    compliance_df: pd.DataFrame,
    recs: dict,
    place_name: Optional[str] = None,
    past_days: Optional[int] = 7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    horizon: int = 24,
    forecast_results: Optional[dict] = None,
) -> str:
    L: list[str] = []
    place = place_name or f"{lat:.4f}N, {lon:.4f}E"
    L.append(f"=== AIR QUALITY REPORT: {place.upper()} ===")
    L.append(f"Selected location: {place} (lat {lat:.4f}, lon {lon:.4f})")
    if start_date and end_date:
        L.append(f"Time window analysed: {start_date} to {end_date}")
    elif past_days:
        L.append(f"Time window analysed: last {past_days} day(s)")
    L.append(f"Forecast horizon: {horizon} hours")
    if not df.empty:
        L.append(f"Latest data point: {df.index[-1]:%Y-%m-%d %H:%M} (local time at the location)")

    aqi_s = df["us_aqi"].dropna() if "us_aqi" in df.columns else pd.Series([], dtype=float)
    aqi_val = float(aqi_s.iloc[-1]) if not aqi_s.empty else None
    if aqi_val is not None:
        L.append(f"\nCurrent US AQI: {aqi_val:.0f} ({get_aqi_category(aqi_val)[0]})")
    else:
        L.append("\nCurrent US AQI: Not available")

    L.append("\nPollutant readings:")
    for pol in POLLUTANTS:
        if pol == "us_aqi":
            continue
        label = POLLUTANT_LABELS[pol]
        s = trend_stats.get(pol, {})
        col = df[pol].dropna() if pol in df.columns else pd.Series([], dtype=float)
        if s.get("mean") is None or col.empty:
            L.append(f"  {label}: Not available")
        else:
            L.append(
                f"  {label}: latest={float(col.iloc[-1]):.1f}, mean={s['mean']:.1f}, "
                f"peak={s['peak']:.1f}, trend={s.get('trend', 'N/A')}"
            )

    L.append(f"\nForecast summary ({horizon}h RandomForest, model-based, indicative):")
    if forecast_results:
        for pol, res in forecast_results.items():
            label = POLLUTANT_LABELS.get(pol, pol)
            fdf = res.get("forecast_df")
            if res.get("error") or fdf is None or fdf.empty:
                L.append(f"  {label}: no forecast ({res.get('error') or 'empty'})")
                continue
            mae = res.get("mae")
            L.append(
                f"  {label}: start={fdf['predicted'].iloc[0]:.1f}, peak={fdf['predicted'].max():.1f}, "
                f"end={fdf['predicted'].iloc[-1]:.1f}, "
                f"MAE={'N/A' if mae is None else format(mae, '.2f')}"
            )
    else:
        L.append("  (not computed)")

    L.append("\nCompliance vs India NAAQS / WHO (% of rolling-mean hours above limit):")
    shown = False
    if compliance_df is not None and not compliance_df.empty:
        for _, row in compliance_df.iterrows():
            pct = row.get("% Hours Exceeding")
            if isinstance(pct, (int, float)):
                L.append(f"  {row['Pollutant']} ({row['Standard']}): {pct}% above {row['Limit (µg/m³)']} µg/m³")
                shown = True
    if not shown:
        L.append("  Not available.")

    if issues and issues is not GENERIC_ISSUES:
        L.append(f"\nDocumented issues - {issues.get('headline', '')}:")
        for iss in issues.get("issues", [])[:5]:
            L.append(f"  - {iss}")

    cat = get_aqi_category(aqi_val)[0] if aqi_val is not None else "unknown"
    L.append(f"\nStandard recommendations for AQI category '{cat}':")
    for tip in recs.get("exposure", [])[:4]:
        L.append(f"  Exposure: {tip}")
    for tip in recs.get("improvement", [])[:3]:
        L.append(f"  Improvement: {tip}")

    L.append("\n[Data: Open-Meteo model-based reanalysis, not certified ground sensors. Forecast is indicative.]")
    return "\n".join(L)


# ── System prompt ───────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are **AirQ Guide**, a friendly air-quality instructor, like a skilled weather presenter \
who explains science in plain everyday language, and can go technical when asked.

## Data you receive
Each message includes "Context data" for the location the user selected on the website, and \
sometimes extra "EXTRA LOCATION" sections for other places the user mentioned. Use them for \
anything about specific places or the page's numbers. Never invent numbers. If something is \
missing or a place could not be found, say so and give general guidance instead.

## Style
1. Plain language first. Assume no science background. Use everyday comparisons \
(e.g. PM2.5 particles are about 30 times thinner than a human hair).
2. Then, only when it helps, add a short "Technical details" section with values and units, \
AQI category, thresholds, trend, forecast and MAE.
3. Short paragraphs or bullets. Be warm and concise. Ask a brief follow-up only if the \
request is truly unclear.
4. Always name the location you are talking about. If asked "which location is this", give \
the place name and coordinates from the context.
5. Use markdown: **bold**, bullet lists, and tables for comparisons.

## Specific situations
- Comparisons: give a clear verdict first (which is cleaner and roughly by how much), then a \
compact table of US AQI and key pollutants for each place.
- Travel and safety: give a clear call (Fine / Take care / Avoid or limit outdoor time) using \
current AQI and the forecast for the time the user mentions. Add practical steps (mask type, \
best time of day, indoor options). Tailor advice for children, older adults, and people with \
asthma or heart conditions. End with: "Not medical advice. Check local official sources (CPCB, WHO)."
- Explaining the website: Overview tab (big AQI card, pollutant cards, recent readings table), \
Trends tab (line chart per pollutant with mean, peak, trend), Prediction tab (blue history line, \
red dashed forecast, MAE = average forecast error on recent hours, lower is better), Issues & Tips \
tab (documented concerns, compliance chart vs NAAQS/WHO limits, recommendations), Chat tab (you). \
Explain AQI scale, PM2.5, PM10, NO2, SO2, O3 simply when asked.
- General air-pollution knowledge (what is smog, why winter is worse, how purifiers work): \
answer from your own knowledge in simple language.
- Unrelated topics: politely say you specialise in air quality and offer to help with that.

## Data note
Once per conversation, mention that values come from Open-Meteo model-based data (not \
certified ground sensors) and forecasts are indicative.
"""


# ── Gemini call ─────────────────────────────────────────────────────────────

def answer_question(
    question: str,
    context: str,
    history: list[dict] | None = None,
) -> tuple[str, bool]:
    """Returns (answer_text, used_gemini). Falls back to rule-based answers on any failure."""
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    model_name = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
    reason = ""

    if api_key:
        try:
            import google.generativeai as genai  # type: ignore

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=model_name, system_instruction=_SYSTEM_PROMPT)
            chat_history = []
            for msg in (history or [])[-10:]:
                role, text = msg.get("role", "user"), msg.get("content", "")
                if role in ("user", "model") and text:
                    chat_history.append({"role": role, "parts": [text]})
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(f"Context data:\n{context}\n\nQuestion: {question}")
            return response.text.strip(), True
        except Exception as exc:
            reason = str(exc)[:200]
            print(f"[AirQ Chat] Gemini error: {reason}", flush=True)
    else:
        reason = "GEMINI_API_KEY not set"
        print(f"[AirQ Chat] {reason}", flush=True)

    return _rule_based_answer(question, context, reason), False


# ── Rule-based fallback ─────────────────────────────────────────────────────

def _safety_advice(aqi: float, cat: str) -> str:
    if aqi <= 50:
        return f"✅ AQI {aqi:.0f} ({cat}): outdoor activity is fine for everyone."
    if aqi <= 100:
        return f"⚠️ AQI {aqi:.0f} ({cat}): fine for most people; very sensitive people may want to limit long exertion."
    if aqi <= 150:
        return f"⚠️ AQI {aqi:.0f} ({cat}): sensitive groups should limit outdoor exertion; others can be out briefly."
    return (f"🚫 AQI {aqi:.0f} ({cat}): limit outdoor activity, use an N95 mask if you must go out.\n\n"
            "Not medical advice. Check local official sources.")


def _rule_based_answer(question: str, context: str, reason: str) -> str:
    q = question.lower()
    notice = f"\n\n---\n⚠️ Basic mode: the AI model is unavailable ({reason}). Answers are limited."

    place_m = re.search(r"=== AIR QUALITY REPORT: (.+?) ===", context)
    place = place_m.group(1).title() if place_m else "the selected location"
    aqi_m = re.search(r"Current US AQI: (\d+) \(([^)]+)\)", context)
    aqi = float(aqi_m.group(1)) if aqi_m else None
    cat = aqi_m.group(2) if aqi_m else "Unknown"

    def section(title: str, n: int = 700):
        i = context.find(title)
        return context[i:i + n].strip() if i >= 0 else None

    if any(w in q for w in ("which location", "which city", "where is", "what place", "location")):
        loc = re.search(r"Selected location: (.+)", context)
        return f"This is **{loc.group(1) if loc else place}**." + notice

    pol_keys = {
        "pm2.5": "pm2_5", "pm 2.5": "pm2_5", "pm10": "pm10", "pm 10": "pm10",
        "no2": "nitrogen_dioxide", "nitrogen": "nitrogen_dioxide",
        "so2": "sulphur_dioxide", "sulphur": "sulphur_dioxide", "sulfur": "sulphur_dioxide",
        "ozone": "ozone", "o3": "ozone",
    }
    for kw, pol in pol_keys.items():
        if kw in q:
            m = re.search(rf"^\s*{re.escape(POLLUTANT_LABELS[pol])}:.*$", context, re.M)
            return (f"**{place}**\n\n{m.group(0).strip()}" if m else "That pollutant's data is not available.") + notice

    if any(w in q for w in ("safe", "outdoor", "exercise", "walk", "run", "jog", "travel", "mask")):
        return (f"**{place}**: " + _safety_advice(aqi, cat) if aqi is not None else "AQI data is unavailable.") + notice
    if any(w in q for w in ("aqi", "air quality", "overall", "index", "level")):
        return (f"**{place}**: current US AQI is **{aqi:.0f}** ({cat})." if aqi is not None else "AQI data is unavailable.") + notice
    if any(w in q for w in ("forecast", "predict", "tomorrow", "next", "future")):
        return (section("Forecast summary") or "No forecast available.") + notice
    if any(w in q for w in ("compliance", "exceed", "limit", "naaqs", "who")):
        return (section("Compliance") or "No compliance data available.") + notice
    if any(w in q for w in ("recommend", "tip", "advice", "what should")):
        return (section("Standard recommendations") or "No recommendations available.") + notice

    return ("I can help with the AQI, pollutant levels, safety advice, forecast and compliance for "
            f"{place}. What would you like to know?") + notice