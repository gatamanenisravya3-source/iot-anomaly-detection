"""
Anomaly Detection for IoT Weather Monitoring
---------------------------------------------
Simulates a stream of IoT sensor readings (temperature, humidity,
pressure, rainfall) like the ones this project originally uploaded to
a Blynk dashboard, then flags anomalies / change points using two
complementary methods:

  1. Rolling z-score       - fast, interpretable, good for single sensors
  2. Isolation Forest      - multivariate, catches joint/contextual anomalies

Run:
    pip install -r requirements.txt
    python anomaly_detection.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------
# 1. Simulate sensor data
# ---------------------------------------------------------------------
def generate_sensor_data(n_hours: int = 24 * 14, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)

    # daily seasonality + slow drift + noise
    temperature = 22 + 6 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.8, n_hours)
    humidity = 55 + 15 * np.sin(2 * np.pi * t / 24 + np.pi / 3) + rng.normal(0, 2.5, n_hours)
    pressure = 1013 + 3 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 0.6, n_hours)
    rainfall = np.clip(rng.exponential(0.4, n_hours) - 0.3, 0, None)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n_hours, freq="h"),
        "temperature": temperature,
        "humidity": humidity,
        "pressure": pressure,
        "rainfall": rainfall,
    })

    # inject a handful of obvious anomalies / change points
    anomaly_idx = rng.choice(n_hours, size=6, replace=False)
    df.loc[anomaly_idx[0], "temperature"] += 12       # heat spike
    df.loc[anomaly_idx[1], "temperature"] -= 10        # cold snap
    df.loc[anomaly_idx[2], "humidity"] += 35            # humidity spike
    df.loc[anomaly_idx[3], "pressure"] -= 15            # pressure drop (storm)
    df.loc[anomaly_idx[4], "rainfall"] += 15            # flash flood reading
    df.loc[anomaly_idx[5:], "pressure"] += 10           # sustained change point

    df["is_injected_anomaly"] = False
    df.loc[anomaly_idx, "is_injected_anomaly"] = True
    return df


# ---------------------------------------------------------------------
# 2. Rolling z-score anomaly detection (per sensor)
# ---------------------------------------------------------------------
def zscore_anomalies(df: pd.DataFrame, columns, window: int = 24, threshold: float = 3.0) -> pd.DataFrame:
    flagged = pd.DataFrame(index=df.index)
    for col in columns:
        rolling_mean = df[col].rolling(window, min_periods=5).mean()
        rolling_std = df[col].rolling(window, min_periods=5).std().replace(0, np.nan)
        z = (df[col] - rolling_mean) / rolling_std
        flagged[f"{col}_zscore"] = z
        flagged[f"{col}_anomaly"] = z.abs() > threshold
    return flagged


# ---------------------------------------------------------------------
# 3. Isolation Forest (multivariate) anomaly detection
# ---------------------------------------------------------------------
def isolation_forest_anomalies(df: pd.DataFrame, columns, contamination: float = 0.03) -> pd.Series:
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(df[columns])
    return pd.Series(preds == -1, index=df.index, name="isoforest_anomaly")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    sensor_cols = ["temperature", "humidity", "pressure", "rainfall"]
    df = generate_sensor_data()

    z_results = zscore_anomalies(df, sensor_cols)
    iso_flags = isolation_forest_anomalies(df, sensor_cols)

    df = pd.concat([df, z_results, iso_flags], axis=1)
    any_zscore_flag = df[[f"{c}_anomaly" for c in sensor_cols]].any(axis=1)

    print(f"Simulated {len(df)} hourly readings across {sensor_cols}")
    print(f"Injected anomalies: {df['is_injected_anomaly'].sum()}")
    print(f"Z-score flagged points: {any_zscore_flag.sum()}")
    print(f"Isolation Forest flagged points: {df['isoforest_anomaly'].sum()}")

    combined_flag = any_zscore_flag | df["isoforest_anomaly"]
    detected_injected = df.loc[df["is_injected_anomaly"], :]
    recall = (combined_flag & df["is_injected_anomaly"]).sum() / max(df["is_injected_anomaly"].sum(), 1)
    print(f"Recall on injected anomalies (either method flags): {recall:.2%}")

    print("\nFlagged rows (z-score OR isolation forest):")
    cols_to_show = ["timestamp"] + sensor_cols + ["isoforest_anomaly", "is_injected_anomaly"]
    print(df.loc[combined_flag, cols_to_show].to_string(index=False))

    out_path = "sensor_readings_with_anomalies.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[OK] Saved full annotated dataset to {out_path}")


if __name__ == "__main__":
    main()
