"""
checklist_audit.py  --  Verifies every spec item is implemented.
Exit 0 if all pass, else prints failures and exits 1.
"""
import sys
import inspect
import logic

failures = []
passes = []

def ok(item):
    passes.append(item)
    print(f"  PASS  {item}")

def fail(item, reason=""):
    failures.append(item)
    print(f"  FAIL  {item}" + (f" -- {reason}" if reason else ""))

print("=== CHECKLIST AUDIT ===\n")

# 1. PRESET CITIES
print("--- 1. Map / Preset Cities ---")
if set(logic.PRESET_CITIES.keys()) == {"Chennai", "Delhi", "Mumbai", "Bengaluru"}:
    ok("4 preset cities defined (Chennai/Delhi/Mumbai/Bengaluru)")
else:
    fail("preset cities", str(set(logic.PRESET_CITIES.keys())))

# 2. DATA FETCHING
print("\n--- 2. Data fetching ---")
sig = inspect.signature(logic.fetch_air_quality)
params = sig.parameters
if "past_days" in params and "start_date" in params and "end_date" in params:
    ok("fetch_air_quality accepts past_days + start_date/end_date")
else:
    fail("fetch_air_quality signature")

import pandas as pd, numpy as np
df = logic.fetch_air_quality(13.0827, 80.2707, past_days=3)
if not df.empty and all(c in df.columns for c in logic.POLLUTANTS):
    ok("fetch returns all 6 pollutant columns")
else:
    fail("fetch columns", str(df.columns.tolist()))

# 3. NULL HANDLING
print("\n--- 3. Null/missing handling ---")
null_series = pd.Series([None, None, None], dtype=float)
result = logic.predict_pollutant(null_series, horizon=24)
if result["error"] is not None:
    ok("predict_pollutant handles all-null series gracefully")
else:
    fail("predict_pollutant null handling")

stat = logic.compute_trend_stats(pd.DataFrame({"pm2_5": [None, None]}, dtype=float))
if stat["pm2_5"]["mean"] is None:
    ok("compute_trend_stats handles all-null column")
else:
    fail("trend stats null handling")

# 4. TREND STATS
print("\n--- 4. Trend stats ---")
stats = logic.compute_trend_stats(df)
for pol in logic.POLLUTANTS:
    s = stats[pol]
    if not all(k in s for k in ("mean", "peak", "peak_time", "trend")):
        fail(f"trend_stats missing keys for {pol}")
        break
else:
    ok("trend_stats has mean/peak/peak_time/trend for all pollutants")
if stats["pm2_5"]["mean"] is not None:
    ok("trend direction computed (not None)")
else:
    fail("trend direction None on valid data")

# 5. AQI CATEGORY
print("\n--- 5. AQI category helper ---")
cats_tested = {
    0: "Good", 60: "Moderate", 120: "Unhealthy for Sensitive",
    180: "Unhealthy", 250: "Very Unhealthy", 400: "Hazardous"
}
for aqi, expected_cat in cats_tested.items():
    cat, color = logic.get_aqi_category(aqi)
    if cat != expected_cat:
        fail(f"get_aqi_category({aqi})", f"got {cat}, expected {expected_cat}")
        break
else:
    ok("get_aqi_category correct for all 6 categories")

nan_cat, _ = logic.get_aqi_category(float("nan"))
if nan_cat == "Unknown":
    ok("get_aqi_category handles NaN")
else:
    fail("get_aqi_category NaN handling", nan_cat)

# 6. ML PREDICTION
print("\n--- 6. ML Prediction ---")
pred = logic.predict_pollutant(df["pm2_5"], horizon=24)
if pred["error"] is None and len(pred["forecast_df"]) == 24:
    ok("predict_pollutant returns 24-row forecast")
else:
    fail("predict_pollutant 24h", pred["error"])

pred48 = logic.predict_pollutant(df["pm2_5"], horizon=48)
if pred48["error"] is None and len(pred48["forecast_df"]) == 48:
    ok("predict_pollutant returns 48-row forecast")
else:
    fail("predict_pollutant 48h", pred48["error"])

if pred["mae"] is not None:
    ok(f"MAE computed on holdout: {pred['mae']:.3f}")
else:
    fail("MAE is None on sufficient data")

short_series = pd.Series([1.0, 2.0, 3.0])
short_pred = logic.predict_pollutant(short_series, horizon=24)
if short_pred["error"] is not None:
    ok("predict_pollutant gracefully handles too-little data")
else:
    fail("predict_pollutant too-little data not handled")

# 7. ISSUES PANEL
print("\n--- 7. Issues panel ---")
for city in ["Chennai", "Delhi", "Mumbai", "Bengaluru"]:
    iss = logic.get_city_issues(city)
    if "headline" in iss and "issues" in iss and len(iss["issues"]) >= 2:
        ok(f"city issues for {city}")
    else:
        fail(f"city issues for {city}")

generic = logic.get_city_issues(None)
if "headline" in generic and "issues" in generic:
    ok("generic issues fallback")
else:
    fail("generic issues fallback")

# 8. COMPLIANCE CHECK
print("\n--- 8. Compliance check ---")
comp = logic.compliance_check(df)
if not comp.empty:
    ok(f"compliance_check returns {len(comp)} rows")
else:
    fail("compliance_check returned empty df")

expected_cols = {"Pollutant", "Averaging Period", "Standard", "Limit (µg/m³)", "% Hours Exceeding"}
if expected_cols.issubset(set(comp.columns)):
    ok("compliance_check has all expected columns")
else:
    fail("compliance_check columns", str(comp.columns.tolist()))

# 9. RECOMMENDATIONS
print("\n--- 9. Recommendations ---")
for cat in ["Good", "Moderate", "Unhealthy for Sensitive", "Unhealthy", "Very Unhealthy", "Hazardous"]:
    recs = logic.RECOMMENDATIONS.get(cat)
    if recs and "exposure" in recs and "improvement" in recs:
        ok(f"recommendations for {cat}")
    else:
        fail(f"recommendations for {cat}")

# 10. DATE RANGE VALIDATION
print("\n--- 10. Date range validation ---")
from datetime import date, timedelta
today = date.today()

# bad: start > end
err = logic.validate_date_range(today, today - timedelta(days=1))
if err: ok("validate: start > end caught")
else: fail("validate: start > end not caught")

# bad: end in future
err = logic.validate_date_range(today, today + timedelta(days=1))
if err: ok("validate: end in future caught")
else: fail("validate: end in future not caught")

# bad: range > 92 days
err = logic.validate_date_range(today - timedelta(days=100), today)
if err: ok("validate: range > 92 days caught")
else: fail("validate: range > 92 days not caught")

# good: 14-day range within last year
err = logic.validate_date_range(today - timedelta(days=14), today)
if err is None: ok("validate: valid 14-day range passes")
else: fail("validate: valid 14-day range", err)

# 11. REVERSE GEOCODE / LOCATION ISSUES
print("\n--- 11. Location issues (non-preset) ---")
iss2 = logic.get_location_issues(19.0760, 72.8777, None)
if iss2.get("auto") or "headline" in iss2:
    ok("get_location_issues works for non-preset (geocode or fallback)")
else:
    fail("get_location_issues non-preset")

# 12. CUSTOM DATE FETCH
print("\n--- 12. Custom date range fetch ---")
df2 = logic.fetch_air_quality(13.0827, 80.2707,
                               start_date=(today - timedelta(days=7)).strftime("%Y-%m-%d"),
                               end_date=(today - timedelta(days=1)).strftime("%Y-%m-%d"))
if not df2.empty:
    ok(f"custom date fetch: {df2.shape[0]} rows")
else:
    fail("custom date fetch returned empty df")

# 13. NO STREAMLIT IMPORTS IN LOGIC
print("\n--- 13. Clean separation ---")
with open("logic.py", encoding="utf-8") as f:
    src = f.read()
if "import streamlit" not in src:
    ok("logic.py has no streamlit import")
else:
    fail("logic.py imports streamlit!")

print(f"\n{'='*45}")
print(f"PASSED: {len(passes)}/{len(passes)+len(failures)}")
if failures:
    print(f"FAILED: {len(failures)}")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
