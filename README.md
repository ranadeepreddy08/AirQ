# AirQ: Air Quality Monitor & Predictor

Interactive web app to monitor, analyze, and forecast air quality for any
location selected on a map.

## Features
- Select a location on a map or choose a preset city
- Choose a time window (last N days or a custom date range)
- Pollutants: PM2.5, PM10, NO₂, SO₂, O₃ (shown as "Not available" when missing)
- Trend analysis: mean, peak, trend direction, current AQI
- ML forecast (24/48h): Random Forest per pollutant with lag features, MAE shown
- Compliance check against India NAAQS and WHO guideline limits
- Environmental issues info and practical recommendations by AQI category

## Tech stack
Python, FastAPI, scikit-learn, pandas, Leaflet, Chart.js, Open-Meteo API

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload
Open http://localhost:8000/static/index.html

## Project structure
- `logic.py`: data fetching, trends, ML, compliance, recommendations
- `main.py`: FastAPI endpoints
- `static/`: frontend (HTML, CSS, JS)

## Data source and limitations
- Open-Meteo Air Quality API (free, no key). Data is model-based, not
  ground-sensor readings.
- Forecasts use only past pollutant data (no weather inputs) and are indicative.
- The compliance check is indicative only. Verify with CPCB/SPCB sources.
