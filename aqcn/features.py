"""
features.py - Build the per-sensor feature matrices for ML calibration.

For every NO2 low-cost sensor (cal1) in Manchester / London / York we assemble
one DataFrame indexed by hourly UTC timestamp with:

  PREDICTORS (X)                         SOURCE
    no2_raw       sensor NO2 (cal1)          the sensor itself
    o3            sensor O3               same device   (if present)
    no            sensor NO               same device   (if present)
    pm25          sensor PM2.5            same device   (if present)
    temp          temperature            co-located reference station
    rh            relative humidity      co-located reference station
    pressure      atmospheric pressure   co-located reference station
    hour_sin/cos  diurnal time encoding  timestamp
    dow_sin/cos   weekly time encoding   timestamp

  TARGET (y)
    ref_no2       reference NO2 (ratified ground truth)   reference station

Reference meteorology is a legitimate external covariate (any deployment gets it
from a weather service). Reference NO2/O3/NO are NEVER used as predictors - only
the sensor's own channels are, to avoid label leakage.

Output: dict[sensor_id] -> DataFrame, cached as data/processed/feature_frames.pkl
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

CITY_CODES = {"Manchester": "MCH", "London": "LON", "York": "YRK"}
SENSOR_VERSION = "cal1"
REF_VERSION = "Ratified"

# Sensor-side measurands we pull as predictors (sensor's OWN channels)
SENSOR_FEATURES = {"NO2": "no2_raw", "O3": "o3", "NO": "no", "PM2.5": "pm25"}
# Reference-side meteorology (legitimate external covariates)
MET_FEATURES = {"Temperature": "temp", "RelHumidity": "rh", "Pressure": "pressure"}
TARGET_MEASURAND = "NO2"

MIN_PAIRED_HOURS = 1000  # need this many (sensor, reference) hours to be usable

FEATURE_CACHE = PROCESSED / "feature_frames.pkl"


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical encodings of hour-of-day and day-of-week."""
    idx = df.index
    hour = idx.hour.values
    dow = idx.dayofweek.values
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def _load_sensor_long() -> pd.DataFrame:
    """Stream the sensor CSV; keep cal1, target cities, feature measurands."""
    keep_meas = set(SENSOR_FEATURES.keys())
    chunks = []
    for chunk in pd.read_csv(RAW / "QUANT_SensorSystems_hourly.csv",
                             chunksize=200_000, low_memory=False):
        sub = chunk[
            (chunk["version"] == SENSOR_VERSION)
            & (chunk["location"].isin(CITY_CODES.keys()))
            & (chunk["measurand"].isin(keep_meas))
        ]
        if len(sub):
            chunks.append(sub[["time", "location", "instrument",
                               "sensornumber", "measurand", "measurement"]])
    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df["city"] = df["location"].map(CITY_CODES)
    df["sensor_id"] = (df["city"] + "__" + df["instrument"]
                       + "_S" + df["sensornumber"].astype(int).astype(str))
    return df


def _load_reference_wide() -> dict[str, pd.DataFrame]:
    """Per-city reference frame: met features + target NO2."""
    ref = pd.read_csv(RAW / "QUANT_Reference_hourly.csv", low_memory=False)
    ref = ref[(ref["version"] == REF_VERSION)
              & (ref["location"].isin(CITY_CODES.keys()))]
    ref["time"] = pd.to_datetime(ref["time"])
    ref["city"] = ref["location"].map(CITY_CODES)

    out: dict[str, pd.DataFrame] = {}
    wanted = list(MET_FEATURES.keys()) + [TARGET_MEASURAND]
    for city_code in CITY_CODES.values():
        c = ref[(ref["city"] == city_code) & (ref["measurand"].isin(wanted))]
        wide = c.pivot_table(index="time", columns="measurand",
                             values="measurement", aggfunc="first")
        rename = dict(MET_FEATURES)
        rename[TARGET_MEASURAND] = "ref_no2"
        wide = wide.rename(columns=rename)
        out[city_code] = wide.sort_index()
    return out


def build_feature_frames(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Assemble per-sensor feature DataFrames. Returns dict[sensor_id] -> DataFrame.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if use_cache and FEATURE_CACHE.exists():
        with open(FEATURE_CACHE, "rb") as f:
            frames = pickle.load(f)
        print(f"[features] Loaded {len(frames)} sensor frames from cache.")
        return frames

    print("[features] Loading sensor long table...")
    long = _load_sensor_long()
    print(f"[features]   {len(long):,} rows, "
          f"{long['sensor_id'].nunique()} sensors")

    print("[features] Loading reference (met + target)...")
    ref_by_city = _load_reference_wide()

    frames: dict[str, pd.DataFrame] = {}
    for sid, g in long.groupby("sensor_id"):
        city = sid.split("__")[0]
        if city not in ref_by_city:
            continue

        # Pivot this sensor's own measurands to wide
        sw = g.pivot_table(index="time", columns="measurand",
                           values="measurement", aggfunc="first")
        sw = sw.rename(columns=SENSOR_FEATURES)

        if "no2_raw" not in sw.columns:
            continue

        # Join reference met + target (co-located, same timestamps)
        ref = ref_by_city[city]
        df = sw.join(ref, how="outer")

        # Restrict to the sensor's active span
        active = sw["no2_raw"].dropna()
        if len(active) == 0:
            continue
        df = df.loc[active.index.min():active.index.max()]

        # Build complete hourly grid
        full = pd.date_range(df.index.min(), df.index.max(), freq="h")
        df = df.reindex(full)
        df.index.name = "time"

        df = _add_time_features(df)

        # Ensure all expected columns exist (some sensors lack o3/no/pm25)
        for col in ["no2_raw", "o3", "no", "pm25", "temp", "rh", "pressure", "ref_no2"]:
            if col not in df.columns:
                df[col] = np.nan

        # Keep only sensors with enough paired (no2_raw, ref_no2) hours
        paired = (df["no2_raw"].notna() & df["ref_no2"].notna()).sum()
        if paired < MIN_PAIRED_HOURS:
            continue

        frames[sid] = df

    with open(FEATURE_CACHE, "wb") as f:
        pickle.dump(frames, f)
    print(f"[features] Built {len(frames)} usable sensor frames "
          f"(>= {MIN_PAIRED_HOURS} paired h). Cached.")

    _print_summary(frames)
    return frames


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the predictor columns that are usable for this sensor.

    Drops feature channels that are entirely missing for the sensor (e.g. a
    device with no on-board NO). no2_raw + met + time features are always kept.
    """
    base = ["no2_raw", "temp", "rh", "pressure",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    optional = ["o3", "no", "pm25"]
    cols = [c for c in base if c in df.columns]
    for c in optional:
        if c in df.columns and df[c].notna().sum() > 0:
            cols.append(c)
    return cols


def _print_summary(frames: dict[str, pd.DataFrame]):
    print("\n" + "=" * 70)
    print("FEATURE FRAMES SUMMARY")
    print("=" * 70)
    by_city: dict[str, int] = {}
    for sid in frames:
        city = sid.split("__")[0]
        by_city[city] = by_city.get(city, 0) + 1
    for city, n in sorted(by_city.items()):
        print(f"  {city}: {n} sensors")
    print(f"  TOTAL: {len(frames)} sensors")

    # Feature availability tally
    has_o3 = sum(1 for d in frames.values() if d["o3"].notna().sum() > 0)
    has_no = sum(1 for d in frames.values() if d["no"].notna().sum() > 0)
    has_pm = sum(1 for d in frames.values() if d["pm25"].notna().sum() > 0)
    print(f"\n  With O3 channel : {has_o3}/{len(frames)}")
    print(f"  With NO channel : {has_no}/{len(frames)}")
    print(f"  With PM2.5      : {has_pm}/{len(frames)}")

    ex = next(iter(frames))
    print(f"\n  Example frame [{ex}]: shape {frames[ex].shape}")
    print(f"  Columns: {list(frames[ex].columns)}")
    print(f"  Usable predictors: {feature_columns(frames[ex])}")
    print("=" * 70)


if __name__ == "__main__":
    build_feature_frames(use_cache=False)
