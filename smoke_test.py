"""Quick smoke test for logic.py -- run with: python smoke_test.py"""
import sys
import logic
import pandas as pd

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  OK {label}: {result}")
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        errors.append(f"{label}: {e}")

print("=== 1. Fetch data (Chennai, 3 days) ===")
df = logic.fetch_air_quality(13.0827, 80.2707, past_days=3)
print(f"  Shape: {df.shape}")
print(f"  Columns: {list(df.columns)}")
print(f"  Non-null: {df.notna().sum().to_dict()}")

print("\n=== 2. Trend stats ===")
stats = logic.compute_trend_stats(df)
for pol, s in stats.items():
    mn = f"{s['mean']:.1f}" if s['mean'] is not None else 'N/A'
    print(f"  {pol}: mean={mn}, trend={s['trend']}")

print("\n=== 3. AQI category ===")
aqi_series = df['us_aqi'].dropna()
aqi = float(aqi_series.iloc[-1]) if not aqi_series.empty else float('nan')
cat, color = logic.get_aqi_category(aqi)
print(f"  AQI={aqi:.0f}, category={cat}, color={color}")

print("\n=== 4. ML prediction ===")
result = logic.predict_pollutant(df['pm2_5'], horizon=24)
print(f"  error={result['error']}")
print(f"  forecast_rows={len(result['forecast_df'])}")
print(f"  MAE={result['mae']}")

print("\n=== 5. Compliance check ===")
comp = logic.compliance_check(df)
print(comp.to_string())

print("\n=== 6. City issues ===")
iss = logic.get_location_issues(13.0827, 80.2707, 'Chennai')
print(f"  headline={iss['headline']}")

print("\n=== 7. Non-preset geocode ===")
iss2 = logic.get_location_issues(19.0760, 72.8777, None)
print(f"  headline={iss2['headline']}, auto={iss2.get('auto')}")

print("\n=== 8. Recommendations ===")
recs = logic.get_recommendations(aqi)
print(f"  exposure[0]={recs['exposure'][0]}")

print("\n=== 9. Date validation ===")
from datetime import date
err = logic.validate_date_range(date(2024, 1, 1), date(2024, 2, 1))
print(f"  Good range: {err}")
err2 = logic.validate_date_range(date(2024, 2, 1), date(2024, 1, 1))
print(f"  Bad range: {err2}")
err3 = logic.validate_date_range(date(2024, 1, 1), date(2024, 6, 1))
print(f"  Too long: {err3}")

print("\n=== 10. Custom date fetch ===")
df2 = logic.fetch_air_quality(13.0827, 80.2707, start_date="2024-10-01", end_date="2024-10-07")
print(f"  Shape: {df2.shape}")

if errors:
    print(f"\n{'='*40}")
    print(f"FAILURES: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nALL CHECKS PASSED")
