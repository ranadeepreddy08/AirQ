"""
app.py  –  Streamlit UI for the Air Quality Monitor & Predictor.
All business logic lives in logic.py; this file is UI-only.
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta

import folium
import pandas as pd
import plotly.graph_objects as go
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
# Minimal CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    .aqi-card {
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 10px;
        color: #111;
        font-weight: 600;
    }
    .stat-card {
        background: #f0f2f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
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
    st.title("🌬️ Air Quality Monitor")
    st.caption("Powered by Open-Meteo · No API key required")
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
        "**Official sources**\n"
        "- [CPCB](https://cpcb.nic.in)\n"
        "- [CPCB AQI Dashboard](https://app.cpcbccr.com/AQI_India/)"
    )

# ---------------------------------------------------------------------------
# Map (tab-agnostic, displayed once at the top)
# ---------------------------------------------------------------------------

st.header("🗺️ Location")

# Session-state for lat/lon from map click
if "map_lat" not in st.session_state:
    st.session_state.map_lat = default_lat
    st.session_state.map_lon = default_lon

# Update if preset changed
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

# If user clicked the map, update coordinates
if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    st.session_state.map_lat = clicked["lat"]
    st.session_state.map_lon = clicked["lng"]
    if selected_city != "Custom / Map click":
        st.info("ℹ️ Map clicked — coordinates updated. Switch to 'Custom / Map click' to use a fully custom location.")

lat = st.session_state.map_lat
lon = st.session_state.map_lon

st.caption(f"**Active coordinates:** {lat:.4f}°N, {lon:.4f}°E")

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
# Resolve city name for non-preset locations
# ---------------------------------------------------------------------------

city_name: str | None = selected_city if selected_city != "Custom / Map click" else None
is_preset = city_name is not None

# ---------------------------------------------------------------------------
# Compute shared values (used across tabs)
# ---------------------------------------------------------------------------

trend_stats = logic.compute_trend_stats(df)
current_aqi = df["us_aqi"].dropna().iloc[-1] if not df["us_aqi"].dropna().empty else float("nan")
aqi_category, aqi_color = logic.get_aqi_category(current_aqi)

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

    # AQI banner
    if not pd.isna(current_aqi):
        st.markdown(
            f"""
            <div class="aqi-card" style="background:{aqi_color};">
                <span style="font-size:1.4rem;">US AQI: {current_aqi:.0f}</span>
                &nbsp;&nbsp;
                <span style="font-size:1.1rem;">{aqi_category}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.warning("AQI data not available for this location/period.")

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
                line=dict(color="#4a90d9", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(74,144,217,0.12)",
            )
        )
        fig.update_layout(
            title=f"{label} — Historical",
            xaxis_title="Time",
            yaxis_title=label,
            height=400,
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)  # noqa: deprecated but still valid

        # Stat cards in a row
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
                line=dict(color="#4a90d9", width=1.5),
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=forecast_df["time"],
                y=forecast_df["predicted"],
                mode="lines+markers",
                name=f"Forecast ({horizon}h)",
                line=dict(color="#e05c5c", width=2, dash="dot"),
                marker=dict(size=5),
            )
        )
        fig2.update_layout(
            title=f"{label} — History + {horizon}-Hour Forecast",
            xaxis_title="Time",
            yaxis_title=label,
            height=430,
            hovermode="x unified",
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

        if mae is not None:
            st.success(f"**Holdout MAE:** {mae:.2f} {label.split('(')[-1].replace(')', '').strip()}")
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
        st.markdown(f"**Current AQI: {current_aqi:.0f} — {aqi_category}**")

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
        # Display table
        st.dataframe(compliance_df, use_container_width=True, hide_index=True)

        # Bar chart
        chart_df = compliance_df.copy()
        chart_df = chart_df[chart_df["% Hours Exceeding"] != "N/A"].copy()
        chart_df["% Hours Exceeding"] = pd.to_numeric(
            chart_df["% Hours Exceeding"], errors="coerce"
        )
        chart_df = chart_df.dropna(subset=["% Hours Exceeding"])
        chart_df["Label"] = chart_df["Pollutant"] + " (" + chart_df["Standard"] + ")"

        if not chart_df.empty:
            fig3 = go.Figure(
                go.Bar(
                    x=chart_df["Label"],
                    y=chart_df["% Hours Exceeding"],
                    marker_color=[
                        "#e05c5c" if v > 50 else "#f0a500" if v > 20 else "#4a90d9"
                        for v in chart_df["% Hours Exceeding"]
                    ],
                    text=chart_df["% Hours Exceeding"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                )
            )
            fig3.update_layout(
                title="% of Hours Exceeding Guideline Limits",
                xaxis_title="Pollutant / Standard",
                yaxis_title="% Hours Exceeding",
                yaxis=dict(range=[0, max(105, chart_df["% Hours Exceeding"].max() + 10)]),
                height=400,
                template="plotly_white",
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown(
            '<p class="disclaimer">Computed from model-based data, indicative only. '
            'Verify with <a href="https://cpcb.nic.in" target="_blank">CPCB</a> official sources.</p>',
            unsafe_allow_html=True,
        )
