# AirQ: Air Quality Monitor & Predictor

Interactive web app to monitor, analyze, and forecast air quality for any location selected on a map, with an AI chat assistant that explains the data in plain language.

## Features
- Select a location on a map or choose a preset city (Chennai, Delhi, Mumbai, Bengaluru)
- Choose a time window: last N days or a custom date range
- Pollutants: PM2.5, PM10, NO₂, SO₂, O₃ plus US AQI (shown as "Not available" when missing)
- Trend analysis: mean, peak, peak time, trend direction, current AQI with category
- ML forecast (24/48h): Random Forest per pollutant with lag features, MAE shown
- Compliance check against India NAAQS and WHO guideline limits
- Environmental issues: curated summaries for preset cities, an auto-detected place summary and official links for other locations
- Practical recommendations by AQI category
- **Chat assistant (AirQ Guide):** explains the data in simple or technical terms, compares locations, and gives travel and safety guidance

## Chat assistant
- Answers using the selected location's computed data (AQI, pollutants, forecast, compliance)
- Can compare other places, for example "Compare this with Delhi" or "Is it safe to travel to Mumbai tomorrow?"
- Powered by the Gemini API. If no key is set or the call fails, it falls back to simple rule-based answers
- General air-quality information only, not medical advice

## Tech stack
Python, FastAPI, scikit-learn, pandas, Leaflet, Chart.js, Gemini API  
Data: Open-Meteo Air Quality and Geocoding APIs, OpenStreetMap Nominatim

## Run locally
Requires Python 3.10+. From the project folder:

```
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000/static/index.html

### Enable the chat assistant (optional)
1. Get a free API key from Google AI Studio (aistudio.google.com).
2. Copy `.env.example` to `.env` and fill in:
```
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.8-flash
```
3. Restart the server. Without a key, the rest of the app still works.

## How it works
1. The frontend sends the selected location and time window to the FastAPI backend.
2. `logic.py` fetches hourly data from Open-Meteo, computes trends, and trains a Random Forest on the history to forecast the next 24/48 hours.
3. The compliance check compares rolling averages against NAAQS and WHO limits.
4. The chat endpoint builds a text summary of this data and sends it to Gemini with the user's question.

## Project structure
- `logic.py`: data fetching, trends, ML forecast, compliance, recommendations, chat logic
- `main.py`: FastAPI endpoints
- `static/`: frontend (HTML, CSS, JS)
- `requirements.txt`, `.env.example`

## Data source and limitations
- Open-Meteo data is model-based, not ground-sensor readings.
- Forecasts use only past pollutant data (no weather inputs) and are indicative.
- The compliance check is indicative only. Verify with CPCB/SPCB sources.
- The environmental issues text for preset cities is a curated summary. Verify with official sources.
- The chat assistant can make mistakes and is not a medical advisor.
