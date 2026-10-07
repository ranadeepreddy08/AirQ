"""
main.py  –  FastAPI app exposing JSON endpoints backed by logic.py.
No Streamlit imports. Uses a simple in-memory 15-minute TTL cache.
Static files served from /static.
"""

from __future__ import annotations

import math
import os
import time
import warnings
from datetime import date
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import logic

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Air Quality Monitor & Predictor API",
    description=(
        "JSON endpoints for air-quality data, trends, ML forecasts, "
        "compliance checks, issues, and recommendations. "
        "Backed by Open-Meteo (no API key required)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files if the /static directory exists
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ---------------------------------------------------------------------------
# In-memory 15-minute TTL cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, tuple[float, Any]] = {}   # key -> (timestamp, value)
_TTL = 15 * 60  # 900 seconds


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.monotonic() - ts > _TTL:
        del _CACHE[key]
        return None
    return value


def _cache_set(key: str, value: Any) -> None:
    _CACHE[key] = (time.monotonic(), value)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any) -> float | None:
    """Convert to float; return None for NaN/Inf."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to JSON-safe list of dicts (NaN → null)."""
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.floating,)):
                v = _safe_float(v)
            elif isinstance(v, float):
                v = _safe_float(v)
            elif isinstance(v, (pd.Timestamp, np.datetime64)):
                v = pd.Timestamp(v).isoformat()
            clean[k] = v
        records.append(clean)
    return records


def _trend_stats_to_json(stats: dict) -> dict:
    """Serialise compute_trend_stats output to JSON-safe dict."""
    out: dict[str, Any] = {}
    for pollutant, s in stats.items():
        out[pollutant] = {
            "label": logic.POLLUTANT_LABELS.get(pollutant, pollutant),
            "mean": _safe_float(s.get("mean")),
            "peak": _safe_float(s.get("peak")),
            "peak_time": (
                pd.Timestamp(s["peak_time"]).isoformat()
                if s.get("peak_time") is not None
                else None
            ),
            "trend": s.get("trend", "unavailable"),
        }
    return out


def _error_response(detail: str, status_code: int = 500) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "error": detail},
    )


# ---------------------------------------------------------------------------
# Shared data-fetch helper (cached)
# ---------------------------------------------------------------------------

def _fetch_df(
    lat: float,
    lon: float,
    past_days: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Fetch air-quality DataFrame with 15-min cache."""
    cache_key = f"data:{lat}:{lon}:{past_days}:{start_date}:{end_date}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    df = logic.fetch_air_quality(
        lat, lon,
        past_days=past_days,
        start_date=start_date,
        end_date=end_date,
    )
    _cache_set(cache_key, df)
    return df


def _resolve_time_params(
    days: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
) -> tuple[Optional[int], Optional[str], Optional[str]]:
    """Validate and normalise time window params; raise HTTPException on error."""
    if days is not None:
        if not (1 <= days <= 92):
            raise HTTPException(status_code=422, detail="days must be between 1 and 92.")
        return days, None, None
    if start_date and end_date:
        try:
            s = date.fromisoformat(start_date)
            e = date.fromisoformat(end_date)
        except ValueError:
            raise HTTPException(status_code=422, detail="Dates must be ISO format YYYY-MM-DD.")
        err = logic.validate_date_range(s, e)
        if err:
            raise HTTPException(status_code=422, detail=err)
        return None, start_date, end_date
    # default: last 7 days
    return 7, None, None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "Air Quality API is running.",
        "docs": "/docs",
        "endpoints": ["/api/data", "/api/trends", "/api/forecast",
                      "/api/issues", "/api/compliance", "/api/recommendations"],
    }


# ── /api/data ──────────────────────────────────────────────────────────────

@app.get("/api/data", summary="Fetch raw hourly air-quality data")
def get_data(
    lat: float = Query(..., description="Latitude", ge=-90, le=90),
    lon: float = Query(..., description="Longitude", ge=-180, le=180),
    days: Optional[int] = Query(None, description="Past N days (1–92). Mutually exclusive with start_date/end_date."),
    start_date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD (with end_date)."),
    end_date:   Optional[str] = Query(None, description="ISO date YYYY-MM-DD (with start_date)."),
):
    past_days, sd, ed = _resolve_time_params(days, start_date, end_date)
    try:
        df = _fetch_df(lat, lon, past_days, sd, ed)
    except Exception as exc:
        return _error_response(f"Failed to fetch data: {exc}")

    if df.empty:
        return _error_response("No data returned for this location and time period.", 404)

    # Reset index so 'time' becomes a column
    df_out = df.reset_index()
    df_out["time"] = df_out["time"].astype(str)

    # Replace NaN with None (JSON null) for each pollutant column
    for col in logic.POLLUTANTS:
        if col in df_out.columns:
            df_out[col] = df_out[col].where(df_out[col].notna(), other=None)

    return {
        "ok": True,
        "lat": lat,
        "lon": lon,
        "rows": len(df_out),
        "pollutants": logic.POLLUTANTS,
        "pollutant_labels": logic.POLLUTANT_LABELS,
        "data": _df_to_records(df_out),
    }


# ── /api/trends ─────────────────────────────────────────────────────────────

@app.get("/api/trends", summary="Trend statistics for each pollutant")
def get_trends(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: Optional[int] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    past_days, sd, ed = _resolve_time_params(days, start_date, end_date)
    try:
        df = _fetch_df(lat, lon, past_days, sd, ed)
    except Exception as exc:
        return _error_response(f"Failed to fetch data: {exc}")

    if df.empty:
        return _error_response("No data returned.", 404)

    stats = logic.compute_trend_stats(df)
    aqi_series = df["us_aqi"].dropna()
    current_aqi = _safe_float(aqi_series.iloc[-1]) if not aqi_series.empty else None
    aqi_category, aqi_color = logic.get_aqi_category(
        float("nan") if current_aqi is None else current_aqi
    )

    return {
        "ok": True,
        "current_aqi": current_aqi,
        "aqi_category": aqi_category,
        "aqi_color": aqi_color,
        "trends": _trend_stats_to_json(stats),
    }


# ── /api/forecast ────────────────────────────────────────────────────────────

@app.get("/api/forecast", summary="RandomForest forecast with MAE for one pollutant")
def get_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    pollutant: str = Query("pm2_5", description=f"One of: {logic.POLLUTANTS}"),
    horizon: int = Query(24, description="Forecast horizon in hours (24 or 48)."),
    days: Optional[int] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    if pollutant not in logic.POLLUTANTS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid pollutant '{pollutant}'. Choose from: {logic.POLLUTANTS}.",
        )
    if horizon not in (24, 48):
        raise HTTPException(status_code=422, detail="horizon must be 24 or 48.")

    past_days, sd, ed = _resolve_time_params(days, start_date, end_date)
    try:
        df = _fetch_df(lat, lon, past_days, sd, ed)
    except Exception as exc:
        return _error_response(f"Failed to fetch data: {exc}")

    if df.empty:
        return _error_response("No data returned.", 404)

    if pollutant not in df.columns or df[pollutant].isna().all():
        return {
            "ok": True,
            "pollutant": pollutant,
            "label": logic.POLLUTANT_LABELS.get(pollutant, pollutant),
            "horizon_hours": horizon,
            "mae": None,
            "forecast": [],
            "error": "Pollutant data not available for this location/period.",
        }

    result = logic.predict_pollutant(df[pollutant], horizon=horizon)

    forecast_records = []
    if not result["forecast_df"].empty:
        fdf = result["forecast_df"].copy()
        fdf["time"] = fdf["time"].astype(str)
        fdf["predicted"] = fdf["predicted"].apply(_safe_float)
        forecast_records = fdf.to_dict(orient="records")

    return {
        "ok": True,
        "pollutant": pollutant,
        "label": logic.POLLUTANT_LABELS.get(pollutant, pollutant),
        "horizon_hours": horizon,
        "mae": _safe_float(result["mae"]),
        "forecast": forecast_records,
        "error": result.get("error"),
    }


# ── /api/issues ──────────────────────────────────────────────────────────────

@app.get("/api/issues", summary="Documented air-quality issues and links for a location")
def get_issues(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    city: Optional[str] = Query(None, description="City name (e.g. 'Chennai'). Auto-detected from lat/lon if omitted."),
):
    issues = logic.get_location_issues(lat, lon, city)
    return {
        "ok": True,
        "headline": issues.get("headline", ""),
        "issues": issues.get("issues", []),
        "place": issues.get("place"),
        "auto_geocoded": issues.get("auto", False),
        "links": [
            {"label": "CPCB",              "url": "https://cpcb.nic.in"},
            {"label": "CPCB AQI Dashboard","url": "https://app.cpcbccr.com/AQI_India/"},
            {"label": "Open-Meteo API",    "url": "https://open-meteo.com"},
        ],
    }


# ── /api/compliance ──────────────────────────────────────────────────────────

@app.get("/api/compliance", summary="Compliance check vs. NAAQS and WHO guidelines")
def get_compliance(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: Optional[int] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
):
    past_days, sd, ed = _resolve_time_params(days, start_date, end_date)
    try:
        df = _fetch_df(lat, lon, past_days, sd, ed)
    except Exception as exc:
        return _error_response(f"Failed to fetch data: {exc}")

    if df.empty:
        return _error_response("No data returned.", 404)

    compliance_df = logic.compliance_check(df)
    if compliance_df.empty:
        return {"ok": True, "note": "Not enough data to compute compliance.", "rows": []}

    # Serialise safely (% Hours Exceeding can be float or "N/A")
    rows = []
    for rec in compliance_df.to_dict(orient="records"):
        pct = rec.get("% Hours Exceeding")
        if isinstance(pct, float) and math.isnan(pct):
            pct = None
        rec["% Hours Exceeding"] = pct
        rows.append(rec)

    return {"ok": True, "rows": rows}


# ── /api/recommendations ──────────────────────────────────────────────────────

@app.get("/api/recommendations", summary="Exposure and improvement recommendations by AQI")
def get_recommendations(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    days: Optional[int] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date:   Optional[str] = Query(None),
    aqi: Optional[float] = Query(None, description="Provide AQI directly to skip fetching data."),
):
    if aqi is not None:
        current_aqi = aqi
    else:
        past_days, sd, ed = _resolve_time_params(days, start_date, end_date)
        try:
            df = _fetch_df(lat, lon, past_days, sd, ed)
        except Exception as exc:
            return _error_response(f"Failed to fetch data: {exc}")
        if df.empty:
            return _error_response("No data returned.", 404)
        aqi_series = df["us_aqi"].dropna()
        current_aqi = float(aqi_series.iloc[-1]) if not aqi_series.empty else float("nan")

    aqi_category, aqi_color = logic.get_aqi_category(current_aqi)
    recs = logic.get_recommendations(current_aqi)

    return {
        "ok": True,
        "current_aqi": _safe_float(current_aqi),
        "aqi_category": aqi_category,
        "aqi_color": aqi_color,
        "exposure": recs.get("exposure", []),
        "improvement": recs.get("improvement", []),
    }


# ── /api/chat ─────────────────────────────────────────────────────────────

from pydantic import BaseModel  # already pulled in by FastAPI


class ChatMessage(BaseModel):
    role: str          # "user" | "model"
    content: str


class ChatRequest(BaseModel):
    question: str
    lat: float
    lon: float
    days: Optional[int] = 7
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    city: Optional[str] = None
    history: List[ChatMessage] = []


@app.post("/api/chat", summary="Chat with the air-quality assistant (Gemini-backed)")
def chat_endpoint(req: ChatRequest):
    try:
        past_days, sd, ed = _resolve_time_params(req.days, req.start_date, req.end_date)

        df = _fetch_df(req.lat, req.lon, past_days, sd, ed)
        if df.empty:
            return {"ok": False, "answer": "No air-quality data available for this location and time period."}

        preset = req.city if req.city in logic.CITY_ISSUES else None
        place = logic.place_name_for(req.lat, req.lon, preset)

        trend_stats = logic.compute_trend_stats(df)
        issues = logic.get_city_issues(preset)
        compliance_df = logic.compliance_check(df)
        aqi_series = df["us_aqi"].dropna()
        current_aqi = float(aqi_series.iloc[-1]) if not aqi_series.empty else float("nan")
        recs = logic.get_recommendations(current_aqi)

        # Forecasts for all pollutants (cached 15 min, first call takes a few seconds)
        horizon = 24
        fkey = f"fc:{req.lat}:{req.lon}:{past_days}:{sd}:{ed}:{horizon}"
        forecast_results = _cache_get(fkey)
        if forecast_results is None:
            forecast_results = logic.run_predictions(df, horizon=horizon)
            _cache_set(fkey, forecast_results)

        context = logic.build_context(
            lat=req.lat, lon=req.lon, df=df,
            trend_stats=trend_stats, issues=issues,
            compliance_df=compliance_df, recs=recs,
            place_name=place, past_days=past_days,
            start_date=sd, end_date=ed,
            horizon=horizon, forecast_results=forecast_results,
        )

        # Other places mentioned in the question (comparison / travel)
        context += logic.build_extra_contexts(
            req.question, exclude=[place or "", preset or ""], past_days=past_days or 7
        )

        history = [{"role": m.role, "content": m.content} for m in req.history]
        answer, used_gemini = logic.answer_question(req.question, context, history)
        return {"ok": True, "answer": answer, "used_gemini": used_gemini, "place": place}

    except Exception as exc:
        return {"ok": False, "answer": f"Sorry, I encountered an error: {exc}"}