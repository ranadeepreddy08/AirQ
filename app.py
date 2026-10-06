"""
app.py  –  Streamlit UI for the Air Quality Monitor & Predictor.
All business logic lives in logic.py; this file is UI-only.
Design: Linear dark system (DESIGN.md) — canvas #010102, accent #5e6ad2.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import folium
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from streamlit_folium import st_folium

import logic

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Air Quality Monitor & Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global Plotly template — Linear dark
# ---------------------------------------------------------------------------

_LINEAR_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#010102",
        plot_bgcolor="#0f1011",
        font=dict(family="Inter, SF Pro Display, -apple-system, system-ui, sans-serif",
                  color="#f7f8f8", size=13),
        title=dict(font=dict(size=18, weight=600, color="#f7f8f8"),
                   x=0, xanchor="left", pad=dict(l=4)),
        xaxis=dict(gridcolor="#23252a", linecolor="#23252a",
                   tickcolor="#62666d", tickfont=dict(color="#8a8f98", size=12),
                   title_font=dict(color="#d0d6e0", size=13)),
        yaxis=dict(gridcolor="#23252a", linecolor="#23252a",
                   tickcolor="#62666d", tickfont=dict(color="#8a8f98", size=12),
                   title_font=dict(color="#d0d6e0", size=13)),
        legend=dict(bgcolor="#0f1011", bordercolor="#23252a", borderwidth=1,
                    font=dict(color="#d0d6e0", size=12)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#141516", bordercolor="#5e6ad2",
                        font=dict(color="#f7f8f8", size=13)),
        margin=dict(l=12, r=12, t=48, b=12),
    )
)
pio.templates["linear_dark"] = _LINEAR_TEMPLATE
pio.templates.default = "linear_dark"

# ---------------------------------------------------------------------------
# AQI category → design-system badge colour (keep green→maroon scale)
# ---------------------------------------------------------------------------

AQI_BADGE_COLORS = {
    "Good":                  {"bg": "#27a644", "text": "#ffffff"},
    "Moderate":              {"bg": "#b8860b", "text": "#ffffff"},
    "Unhealthy for Sensitive Groups": {"bg": "#d97706", "text": "#ffffff"},
    "Unhealthy":             {"bg": "#dc2626", "text": "#ffffff"},
    "Very Unhealthy":        {"bg": "#9333ea", "text": "#ffffff"},
    "Hazardous":             {"bg": "#7f1d1d", "text": "#fecaca"},
}

def _badge_colors(category: str) -> dict:
    return AQI_BADGE_COLORS.get(category, {"bg": "#23252a", "text": "#d0d6e0"})

# ---------------------------------------------------------------------------
# Design-system CSS injection
# ---------------------------------------------------------------------------

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* ── Root tokens ──────────────────────────────────────────────── */
    :root {
        --canvas:       #010102;
        --surface-1:    #0f1011;
        --surface-2:    #141516;
        --surface-3:    #18191a;
        --hairline:     #23252a;
        --hairline-str: #34343a;
        --ink:          #f7f8f8;
        --ink-muted:    #d0d6e0;
        --ink-subtle:   #8a8f98;
        --ink-tertiary: #62666d;
        --accent:       #5e6ad2;
        --accent-hover: #828fff;
        --success:      #27a644;
    }

    /* ── Global resets ────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', 'SF Pro Display', -apple-system, system-ui, sans-serif !important;
        background-color: var(--canvas) !important;
        color: var(--ink) !important;
    }

    /* ── Sidebar ──────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--surface-1) !important;
        border-right: 1px solid var(--hairline) !important;
    }
    [data-testid="stSidebar"] * { color: var(--ink-muted) !important; }
    [data-testid="stSidebar"] strong { color: var(--ink) !important; }

    /* ── Main area ────────────────────────────────────────────────── */
    [data-testid="stAppViewContainer"] > .main {
        background-color: var(--canvas) !important;
    }

    /* ── Metrics ──────────────────────────────────────────────────── */
    [data-testid="metric-container"] {
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 14px 18px;
    }
    [data-testid="metric-container"] label { color: var(--ink-subtle) !important; font-size: 12px; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: var(--ink) !important; font-size: 1.5rem !important; font-weight: 600;
    }

    /* ── Tab bar ──────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {
        background: var(--surface-1);
        border-radius: 9999px;
        padding: 4px 6px;
        gap: 4px;
        border: 1px solid var(--hairline);
        display: inline-flex;
    }
    [data-testid="stTabs"] button[role="tab"] {
        border-radius: 9999px !important;
        padding: 6px 18px !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        color: var(--ink-subtle) !important;
        background: transparent !important;
        border: none !important;
        letter-spacing: 0;
        transition: background 0.18s, color 0.18s;
    }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: var(--surface-2) !important;
        color: var(--ink) !important;
        border: 1px solid var(--hairline-str) !important;
    }

    /* ── Cards ────────────────────────────────────────────────────── */
    .lnr-card {
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 12px;
    }
    .lnr-card-feat {
        background: var(--surface-2);
        border: 1px solid var(--hairline-str);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 12px;
    }

    /* ── Hero AQI card ────────────────────────────────────────────── */
    .hero-aqi {
        background: var(--surface-1);
        border: 1px solid var(--hairline-str);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 24px;
    }
    .hero-aqi-value {
        font-size: 3.2rem;
        font-weight: 700;
        letter-spacing: -2px;
        color: var(--ink);
        line-height: 1.0;
    }
    .hero-aqi-label {
        font-size: 14px;
        color: var(--ink-subtle);
        font-weight: 500;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }
    .hero-aqi-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0;
        margin-top: 6px;
    }
    .hero-aqi-coord {
        font-size: 12px;
        color: var(--ink-tertiary);
        margin-top: 4px;
        font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
    }

    /* ── Stat card ────────────────────────────────────────────────── */
    .stat-card {
        background: var(--surface-1);
        border: 1px solid var(--hairline);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
        color: var(--ink-subtle);
        font-size: 13px;
    }

    /* ── Divider ──────────────────────────────────────────────────── */
    hr { border-color: var(--hairline) !important; }

    /* ── DataFrames ───────────────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--hairline);
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Buttons ──────────────────────────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 8px 14px !important;
        transition: background 0.18s;
    }
    .stButton > button:hover { background: var(--accent-hover) !important; }

    /* ── Footer ───────────────────────────────────────────────────── */
    .lnr-footer {
        background: var(--canvas);
        border-top: 1px solid var(--hairline);
        padding: 28px 0 16px;
        margin-top: 48px;
        color: var(--ink-subtle);
        font-size: 12px;
        line-height: 1.6;
    }
    .lnr-footer a { color: var(--accent); text-decoration: none; }
    .lnr-footer a:hover { color: var(--accent-hover); }

    /* ── Headings ─────────────────────────────────────────────────── */
    h1, h2, h3 {
        letter-spacing: -0.6px;
        font-weight: 600;
        color: var(--ink) !important;
    }
    .disclaimer { font-size: 12px; color: var(--ink-tertiary); font-style: italic; }

    /* ── Plotly container rounding ────────────────────────────────── */
    [data-testid="stPlotlyChart"] > div {
        border: 1px solid var(--hairline);
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached data fetcher (wraps logic.fetch_air_quality)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch(
    lat: float,
    lon: float,
    past_days: int | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    return logic.fetch_air_quality(
        lat,
        lon,
        past_days=past_days,
        start_date=start_date,
        end_date=end_date,
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.2rem;font-weight:700;letter-spacing:-0.4px;"
        "color:#f7f8f8;margin-bottom:2px'>🌬️ Air Quality</div>"
        "<div style='font-size:12px;color:#8a8f98;margin-bottom:16px'>"
        "Powered by Open-Meteo · No API key required</div>",
        unsafe_allow_html=True,
    )
    st.divider()

    # City presets
    preset_options = ["Custom / Map click"] + list(logic.PRESET_CITIES.keys())
    selected_city = st.selectbox("📍 Preset city", preset_options, index=1)

    if selected_city != "Custom / Map click":
        default_lat = logic.PRESET_CITIES[selected_city]["lat"]
        default_lon = logic.PRESET_CITIES[selected_city]["lon"]
    else:
        default_lat = 13.0827
        default_lon = 80.2707

    st.divider()

    # Time window mode
    time_mode = st.radio(
        "⏱️ Time window",
        ["Last N days", "Custom date range"],
        horizontal=True,
    )

    past_days_val: int | None = None
    start_date_str: str | None = None
    end_date_str: str | None = None
    date_error: str | None = None

    if time_mode == "Last N days":
        past_days_val = st.slider("📅 Past days", min_value=1, max_value=30, value=7)
    else:
        today = date.today()
        default_start = today - timedelta(days=14)
        col_s, col_e = st.columns(2)
        with col_s:
            start_d = st.date_input("From", value=default_start, max_value=today)
        with col_e:
            end_d = st.date_input("To", value=today, max_value=today)
        date_error = logic.validate_date_range(start_d, end_d)
        if date_error:
            st.error(f"📅 {date_error}")
        else:
            start_date_str = start_d.strftime("%Y-%m-%d")
            end_date_str = end_d.strftime("%Y-%m-%d")

    st.divider()

    horizon = st.selectbox("🔮 Forecast horizon", [24, 48], format_func=lambda h: f"{h} hours")

    st.divider()
    st.markdown(
        "<div style='font-size:12px;color:#8a8f98'>"
        "<strong style='color:#d0d6e0'>Official sources</strong><br>"
        "· <a href='https://cpcb.nic.in' target='_blank' style='color:#5e6ad2'>CPCB</a><br>"
        "· <a href='https://app.cpcbccr.com/AQI_India/' target='_blank' style='color:#5e6ad2'>"
        "CPCB AQI Dashboard</a></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------

st.markdown(
    "<h2 style='letter-spacing:-0.6px;font-size:1.4rem;margin-bottom:8px'>🗺️ Location</h2>",
    unsafe_allow_html=True,
)

if "map_lat" not in st.session_state:
    st.session_state.map_lat = default_lat
    st.session_state.map_lon = default_lon

if selected_city != "Custom / Map click":
    st.session_state.map_lat = default_lat
    st.session_state.map_lon = default_lon

m = folium.Map(
    location=[st.session_state.map_lat, st.session_state.map_lon],
    zoom_start=10,
    tiles="OpenStreetMap",
)
folium.Marker(
    [st.session_state.map_lat, st.session_state.map_lon],
    tooltip="Selected location",
    icon=folium.Icon(color="blue", icon="info-sign"),
).add_to(m)

map_data = st_folium(m, height=350, use_container_width=True, returned_objects=["last_clicked"])

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    st.session_state.map_lat = clicked["lat"]
    st.session_state.map_lon = clicked["lng"]
    if selected_city != "Custom / Map click":
        st.info("ℹ️ Map clicked — coordinates updated. Switch to 'Custom / Map click' to use a fully custom location.")

lat = st.session_state.map_lat
lon = st.session_state.map_lon

st.markdown(
    f"<div style='font-size:12px;color:#62666d;font-family:ui-monospace,monospace;margin-top:4px'>"
    f"Active coordinates: {lat:.4f}°N, {lon:.4f}°E</div>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Guard: skip if date range invalid
# ---------------------------------------------------------------------------

if time_mode == "Custom date range" and date_error:
    st.error("⛔ Fix the date range in the sidebar before fetching data.")
    st.stop()

# ---------------------------------------------------------------------------
# Data fetch
# ---------------------------------------------------------------------------

with st.spinner("🔄 Fetching air-quality data from Open-Meteo…"):
    try:
        df = cached_fetch(
            lat=lat,
            lon=lon,
            past_days=past_days_val,
            start_date=start_date_str,
            end_date=end_date_str,
        )
        fetch_error = None
    except Exception as exc:
        df = pd.DataFrame()
        fetch_error = str(exc)

if fetch_error:
    st.error(f"🚨 Could not fetch data: {fetch_error}")
    st.info("Check your internet connection or try again later. Open-Meteo is a free API and may occasionally be slow.")
    st.stop()

if df.empty:
    st.warning("⚠️ No data returned for this location and time period.")
    st.stop()

# ---------------------------------------------------------------------------
# Shared computed values
# ---------------------------------------------------------------------------

city_name: str | None = selected_city if selected_city != "Custom / Map click" else None
trend_stats = logic.compute_trend_stats(df)
current_aqi = df["us_aqi"].dropna().iloc[-1] if not df["us_aqi"].dropna().empty else float("nan")
aqi_category, aqi_color = logic.get_aqi_category(current_aqi)

# ---------------------------------------------------------------------------
# Hero AQI card + gauge (shown once, above tabs)
# ---------------------------------------------------------------------------

bc = _badge_colors(aqi_category)
aqi_display = f"{current_aqi:.0f}" if not pd.isna(current_aqi) else "—"
aqi_cat_display = aqi_category if not pd.isna(current_aqi) else "No data"

col_hero, col_gauge = st.columns([1, 1], gap="large")

with col_hero:
    st.markdown(
        f"""
        <div class="hero-aqi">
          <div>
            <div class="hero-aqi-label">US Air Quality Index</div>
            <div class="hero-aqi-value">{aqi_display}</div>
            <div>
              <span class="hero-aqi-badge"
                style="background:{bc['bg']};color:{bc['text']}">
                {aqi_cat_display}
              </span>
            </div>
            <div class="hero-aqi-coord">{lat:.4f}°N · {lon:.4f}°E</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_gauge:
    if not pd.isna(current_aqi):
        aqi_val = float(current_aqi)
        # Gauge colour steps matching AQI bands
        gauge_color = bc["bg"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=aqi_val,
            number=dict(font=dict(size=36, color="#f7f8f8", family="Inter")),
            gauge=dict(
                axis=dict(
                    range=[0, 500],
                    tickvals=[0, 50, 100, 150, 200, 300, 500],
                    ticktext=["0", "50", "100", "150", "200", "300", "500"],
                    tickfont=dict(color="#8a8f98", size=11),
                    linecolor="#23252a",
                ),
                bar=dict(color=gauge_color, thickness=0.55),
                bgcolor="#0f1011",
                borderwidth=1,
                bordercolor="#23252a",
                steps=[
                    dict(range=[0, 50],   color="#0f2d17"),
                    dict(range=[50, 100],  color="#2d2400"),
                    dict(range=[100, 150], color="#2d1800"),
                    dict(range=[150, 200], color="#2d0000"),
                    dict(range=[200, 300], color="#1e0033"),
                    dict(range=[300, 500], color="#1a0000"),
                ],
                threshold=dict(
                    line=dict(color=gauge_color, width=3),
                    thickness=0.85,
                    value=aqi_val,
                ),
            ),
            title=dict(text="AQI Gauge", font=dict(color="#8a8f98", size=13)),
            domain=dict(x=[0, 1], y=[0, 1]),
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#010102",
            plot_bgcolor="#010102",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=24, r=24, t=48, b=8),
            height=220,
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    else:
        st.markdown(
            "<div class='lnr-card' style='color:#8a8f98;text-align:center;padding:48px 0'>"
            "AQI gauge unavailable — no data</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tab_overview, tab_trends, tab_prediction, tab_issues = st.tabs(
    ["📊 Overview", "📈 Trends", "🔮 Prediction", "⚠️ Issues & Tips"]
)

# ============================================================
# TAB 1: OVERVIEW
# ============================================================

with tab_overview:
    st.subheader("Current Air Quality Snapshot")

    # Summary metric cards
    col_grid = st.columns(3)
    display_pollutants = [p for p in logic.POLLUTANTS if p != "us_aqi"]
    for idx, pol in enumerate(display_pollutants):
        s = trend_stats.get(pol, {})
        mean_v = s.get("mean")
        peak_v = s.get("peak")
        trend_v = s.get("trend", "unavailable")
        label = logic.POLLUTANT_LABELS.get(pol, pol)
        with col_grid[idx % 3]:
            with st.container():
                st.markdown(f"**{label}**")
                if mean_v is not None:
                    st.metric(
                        label="Mean",
                        value=f"{mean_v:.1f}",
                        delta=trend_v,
                        delta_color="inverse" if "rising" in trend_v else "normal",
                    )
                    st.caption(f"Peak: {peak_v:.1f}")
                else:
                    st.markdown(
                        '<div class="stat-card">Not available</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.subheader("Recent Readings (last 24 rows)")
    display_df = df.tail(24).copy()
    for col in logic.POLLUTANTS:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda v: f"{v:.1f}" if pd.notna(v) else "Not available"
            )
    st.dataframe(display_df.rename(columns=logic.POLLUTANT_LABELS), use_container_width=True)


# ============================================================
# TAB 2: TRENDS
# ============================================================

with tab_trends:
    st.subheader("Pollutant Trends Over Time")

    selected_pol = st.selectbox(
        "Select pollutant",
        logic.POLLUTANTS,
        format_func=lambda p: logic.POLLUTANT_LABELS[p],
        key="trend_pol",
    )

    series = df[selected_pol].dropna()
    label = logic.POLLUTANT_LABELS[selected_pol]

    if series.empty:
        st.warning(f"No data available for {label}.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line=dict(color="#5e6ad2", width=2.0),
                fill="tozeroy",
                fillcolor="rgba(94,106,210,0.10)",
            )
        )
        fig.update_layout(
            title=f"{label} — Historical",
            xaxis_title="Time",
            yaxis_title=label,
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)

        s = trend_stats.get(selected_pol, {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean", f"{s['mean']:.1f}" if s.get("mean") is not None else "N/A")
        with c2:
            st.metric("Peak", f"{s['peak']:.1f}" if s.get("peak") is not None else "N/A")
        with c3:
            pt = s.get("peak_time")
            st.metric("Peak Time", pt.strftime("%Y-%m-%d %H:%M") if pt else "N/A")
        with c4:
            st.metric("Trend", s.get("trend", "N/A"))


# ============================================================
# TAB 3: PREDICTION
# ============================================================

with tab_prediction:
    st.subheader("ML Forecast (RandomForest)")
    st.caption(
        "Features: hour of day, day of week, 1 h / 3 h / 24 h lags. "
        "Trained on available history; MAE evaluated on a time-based holdout."
    )

    pred_pol = st.selectbox(
        "Pollutant to forecast",
        logic.POLLUTANTS,
        format_func=lambda p: logic.POLLUTANT_LABELS[p],
        key="pred_pol",
    )

    with st.spinner("⚙️ Training model and generating forecast…"):
        result = logic.predict_pollutant(df[pred_pol], horizon=int(horizon))

    if result["error"]:
        st.warning(f"⚠️ {result['error']}")
    else:
        forecast_df = result["forecast_df"]
        mae = result["mae"]
        label = logic.POLLUTANT_LABELS[pred_pol]

        history_series = df[pred_pol].dropna()

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=history_series.index,
                y=history_series.values,
                mode="lines",
                name="History",
                line=dict(color="#5e6ad2", width=1.8),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["predicted"],
                mode="lines+markers",
                name=f"Forecast ({horizon}h)",
                line=dict(color="#e05c5c", width=2.2, dash="dot"),
                marker=dict(size=5, color="#e05c5c"),
            )
        )
        fig2.update_layout(
            title=f"{label} — History + {horizon}-Hour Forecast",
            xaxis_title="Time",
            yaxis_title=label,
            height=440,
        )
        st.plotly_chart(fig2, use_container_width=True)

        if mae is not None:
            unit = label.split("(")[-1].replace(")", "").strip()
            st.markdown(
                f"<div class='lnr-card-feat' style='display:inline-block;padding:10px 20px'>"
                f"<span style='color:#8a8f98;font-size:12px'>Holdout MAE</span><br>"
                f"<span style='font-size:1.4rem;font-weight:600;color:#f7f8f8'>{mae:.2f}</span>"
                f"<span style='color:#8a8f98;font-size:13px'> {unit}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("MAE could not be computed (not enough data for holdout).")

        with st.expander("📋 Raw forecast table"):
            st.dataframe(
                forecast_df.rename(columns={"time": "Forecast Time", "predicted": label}),
                use_container_width=True,
            )


# ============================================================
# TAB 4: ISSUES & TIPS
# ============================================================

with tab_issues:

    # ---- City issues panel ----
    st.subheader("📋 Documented Air-Quality Issues")

    with st.spinner("🔍 Retrieving location information…"):
        issues_dict = logic.get_location_issues(lat, lon, city_name)

    st.markdown(f"### {issues_dict['headline']}")

    if issues_dict.get("auto"):
        st.info(
            "🤖 **Auto-retrieved summary** — location identified via OpenStreetMap Nominatim. "
            "Verify all claims with official sources."
        )

    for issue in issues_dict["issues"]:
        st.markdown(f"- {issue}")

    st.markdown(
        '<p class="disclaimer">Illustrative summary. Verify with '
        '<a href="https://cpcb.nic.in" target="_blank">CPCB</a> / '
        '<a href="https://app.cpcbccr.com/AQI_India/" target="_blank">CPCB AQI Dashboard</a> '
        'official sources.</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ---- Recommendations ----
    st.subheader("💡 Recommendations")

    recs = logic.get_recommendations(current_aqi)
    if not pd.isna(current_aqi):
        bc2 = _badge_colors(aqi_category)
        st.markdown(
            f"<span style='font-size:14px;color:#d0d6e0'>Current AQI: "
            f"<strong>{current_aqi:.0f}</strong> &nbsp;"
            f"<span style='background:{bc2['bg']};color:{bc2['text']};"
            f"padding:2px 10px;border-radius:9999px;font-size:12px'>"
            f"{aqi_category}</span></span>",
            unsafe_allow_html=True,
        )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("**🚶 Exposure-reduction tips**")
        for tip in recs.get("exposure", []):
            st.markdown(f"- {tip}")
    with col_r2:
        st.markdown("**🌱 Local-improvement practices**")
        for tip in recs.get("improvement", []):
            st.markdown(f"- {tip}")

    st.divider()

    # ---- Compliance check ----
    st.subheader("📏 Compliance vs. NAAQS & WHO Guidelines")
    st.caption(
        "Computed from model-based forecast/reanalysis data supplied by Open-Meteo. "
        "**Indicative only** — not a substitute for certified monitoring station readings."
    )

    with st.spinner("📊 Computing compliance statistics…"):
        compliance_df = logic.compliance_check(df)

    if compliance_df.empty:
        st.info("Not enough data to compute compliance statistics.")
    else:
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)

        chart_df = compliance_df.copy()
        chart_df = chart_df[chart_df["% Hours Exceeding"] != "N/A"].copy()
        chart_df["% Hours Exceeding"] = pd.to_numeric(
            chart_df["% Hours Exceeding"], errors="coerce"
        )
        chart_df = chart_df.dropna(subset=["% Hours Exceeding"])
        chart_df["Label"] = chart_df["Pollutant"] + " (" + chart_df["Standard"] + ")"

        if not chart_df.empty:
            bar_colors = [
                "#e05c5c" if v > 50 else "#d97706" if v > 20 else "#5e6ad2"
                for v in chart_df["% Hours Exceeding"]
            ]
            fig3 = go.Figure(
                go.Bar(
                    x=chart_df["Label"],
                    y=chart_df["% Hours Exceeding"],
                    marker_color=bar_colors,
                    text=chart_df["% Hours Exceeding"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                    textfont=dict(color="#d0d6e0", size=11),
                )
            )
            fig3.update_layout(
                title="% of Hours Exceeding Guideline Limits",
                xaxis_title="Pollutant / Standard",
                yaxis_title="% Hours Exceeding",
                yaxis=dict(range=[0, max(105, chart_df["% Hours Exceeding"].max() + 10)]),
                height=400,
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            '<p class="disclaimer">Computed from model-based data, indicative only. '
            'Verify with <a href="https://cpcb.nic.in" target="_blank">CPCB</a> official sources.</p>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="lnr-footer">
      <strong style="color:#d0d6e0">Air Quality Monitor & Predictor</strong>
      &nbsp;·&nbsp;
      Data: <a href="https://open-meteo.com" target="_blank">Open-Meteo Air Quality API</a>
      (model-based reanalysis, not certified station data)
      &nbsp;·&nbsp;
      ML: RandomForest per-pollutant forecast, time-based holdout MAE
      <br>
      <span style="color:#62666d">
        ⚠️ Limitations: forecasts are statistical approximations; values may differ from
        ground-truth station readings. Always verify health decisions with
        <a href="https://cpcb.nic.in" target="_blank">CPCB</a> or WHO certified sources.
        Open-Meteo data is © Open-Meteo contributors, CC BY 4.0.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)
