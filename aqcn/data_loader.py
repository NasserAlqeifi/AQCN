"""
data_loader.py - Load and clean the QUANT dataset.

Loads ALL sensors (all 49 instruments, all 3 cities) for the NO2 measurand,
calibration version "cal1", against Ratified reference monitors.

Output
------
sensors_df  : wide DataFrame - index = hourly UTC, columns = "<CITY>__<instrument>_S<n>"
              e.g. "MCH__Prax1_S1", "LON__AP1_S1", "YRK__PA1_S1"
reference_df: wide DataFrame - columns = "MCH__NO2", "LON__NO2", "YRK__NO2"
sensor_meta : dict  sensor_id -> {city, city_code, instrument, sensornumber, lat, lon}
"""

import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
ZENODO_RECORD = "10775692"

TARGET_MEASURAND = "NO2"
TARGET_VERSION = "cal1"
REF_VERSION = "Ratified"
MIN_COVERAGE_PCT = 2.0  # drop sensors with <2% valid readings

CITY_CODES = {"Manchester": "MCH", "London": "LON", "York": "YRK"}
CITY_CENTERS = {
    "Manchester": (53.4808, -2.2426),
    "London":     (51.5074, -0.1278),
    "York":       (53.9600, -1.0873),
}
CITY_FULL = {v: k for k, v in CITY_CODES.items()}


def _illustrative_gps(instrument: str, sensornumber: int, city: str) -> tuple[float, float]:
    """
    Deterministic illustrative GPS position within ~2 km of city centre.
    QUANT provides no per-sensor GPS; these positions are for visualisation only.
    """
    seed = int(hashlib.md5(f"{instrument}_{sensornumber}_{city}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    clat, clon = CITY_CENTERS.get(city, (53.48, -2.24))
    spread = 0.014  # ~1.5 km in latitude degrees
    lat = clat + rng.uniform(-spread, spread)
    lon = clon + rng.uniform(-spread * 1.6, spread * 1.6)
    return round(lat, 5), round(lon, 5)


def download_quant():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    needed = [
        "QUANT_SensorSystems_hourly.csv",
        "QUANT_Reference_hourly.csv",
        "QUANT_DuplicateRef_hourly.csv",
    ]
    if all((DATA_RAW / f).exists() for f in needed):
        print("[data_loader] Raw data present.")
        return
    print(f"[data_loader] Fetching Zenodo record {ZENODO_RECORD}...")
    req = urllib.request.Request(
        f"https://zenodo.org/api/records/{ZENODO_RECORD}",
        headers={"User-Agent": "AQCN/1.0"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        record = json.loads(resp.read())
    for f in record["files"]:
        dest = DATA_RAW / f["key"]
        if dest.exists():
            continue
        size_mb = f.get("size", 0) / 1e6
        print(f"[data_loader] Downloading {f['key']} ({size_mb:.1f} MB)...")
        urllib.request.urlretrieve(f["links"]["self"], dest)
    for zp in DATA_RAW.glob("*.zip"):
        with zipfile.ZipFile(zp) as zf:
            zf.extractall(DATA_RAW)


def _load_sensors_wide() -> tuple[pd.DataFrame, dict]:
    """
    Read sensor file, filter to NO2/cal1 for all 3 cities.
    Returns (wide_df, sensor_meta_dict).
    """
    sensor_path = DATA_RAW / "QUANT_SensorSystems_hourly.csv"
    chunks = []
    for chunk in pd.read_csv(sensor_path, chunksize=200_000, low_memory=False):
        sub = chunk[
            (chunk["measurand"] == TARGET_MEASURAND) &
            (chunk["version"] == TARGET_VERSION) &
            (chunk["location"].isin(CITY_CODES.keys()))
        ].copy()
        if len(sub) == 0:
            continue
        sub["city_code"] = sub["location"].map(CITY_CODES)
        sub["sensor_id"] = (
            sub["city_code"] + "__"
            + sub["instrument"] + "_S"
            + sub["sensornumber"].astype(int).astype(str)
        )
        chunks.append(sub[["time", "sensor_id", "measurement", "location",
                            "instrument", "sensornumber"]])

    if not chunks:
        raise RuntimeError(f"No data found for {TARGET_MEASURAND}/{TARGET_VERSION}")

    df = pd.concat(chunks, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])

    # Build sensor meta before pivoting
    sensor_meta: dict = {}
    for _, row in df.drop_duplicates("sensor_id").iterrows():
        sid = row["sensor_id"]
        city = row["location"]
        instr = row["instrument"]
        snum = int(row["sensornumber"])
        lat, lon = _illustrative_gps(instr, snum, city)
        sensor_meta[sid] = {
            "city":         city,
            "city_code":    CITY_CODES[city],
            "instrument":   instr,
            "sensornumber": snum,
            "lat":          lat,
            "lon":          lon,
        }

    wide = df.pivot_table(
        index="time", columns="sensor_id", values="measurement", aggfunc="first"
    )
    wide.columns.name = None
    wide = wide.sort_index()

    full_range = pd.date_range(wide.index.min(), wide.index.max(), freq="h")
    wide = wide.reindex(full_range)
    wide.index.name = "time"

    # Drop sensors with less than MIN_COVERAGE_PCT valid data
    coverage = wide.notna().mean() * 100
    keep = coverage[coverage >= MIN_COVERAGE_PCT].index.tolist()
    wide = wide[keep]
    sensor_meta = {k: v for k, v in sensor_meta.items() if k in keep}

    return wide, sensor_meta


def _load_reference_wide() -> pd.DataFrame:
    """
    Load ratified reference NO2 for all 3 cities.
    Returns wide DataFrame with columns MCH__NO2, LON__NO2, YRK__NO2.
    """
    ref_path = DATA_RAW / "QUANT_Reference_hourly.csv"
    ref = pd.read_csv(ref_path, low_memory=False)
    ref = ref[
        (ref["version"] == REF_VERSION) &
        (ref["measurand"] == TARGET_MEASURAND) &
        (ref["location"].isin(CITY_CODES.keys()))
    ].copy()
    ref["time"] = pd.to_datetime(ref["time"])
    ref["col"] = ref["location"].map(CITY_CODES) + "__NO2"

    wide = ref.pivot_table(
        index="time", columns="col", values="measurement", aggfunc="first"
    )
    wide.columns.name = None
    wide = wide.sort_index()
    wide.index.name = "time"
    return wide


def load_data(use_cache: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (sensors_df, reference_df, sensor_meta).

    sensors_df  : index=time, columns=sensor_ids (city-prefixed)
    reference_df: index=time, columns=city NO2 references
    sensor_meta : {sensor_id: {city, city_code, instrument, sensornumber, lat, lon}}
    """
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    s_cache = DATA_PROCESSED / "sensors_df.parquet"
    r_cache = DATA_PROCESSED / "reference_df.parquet"
    m_cache = DATA_PROCESSED / "sensor_meta.json"

    if use_cache and s_cache.exists() and r_cache.exists() and m_cache.exists():
        sensors_df  = pd.read_parquet(s_cache)
        reference_df = pd.read_parquet(r_cache)
        with open(m_cache) as f:
            sensor_meta = json.load(f)
        # Validate: if cached sensors are few (old format), regenerate
        if len(sensors_df.columns) >= 10:
            print("[data_loader] Loaded from cache.")
            _print_summary(sensors_df, reference_df, sensor_meta)
            return sensors_df, reference_df, sensor_meta
        print("[data_loader] Cache outdated (too few sensors) - regenerating...")

    download_quant()

    print("[data_loader] Loading all sensors (all cities, NO2, cal1)...")
    sensors_df, sensor_meta = _load_sensors_wide()

    print("[data_loader] Loading reference monitors...")
    reference_df = _load_reference_wide()

    # Align time ranges
    common_start = max(sensors_df.index.min(), reference_df.index.min())
    common_end   = min(sensors_df.index.max(), reference_df.index.max())
    full_range = pd.date_range(common_start, common_end, freq="h")
    sensors_df   = sensors_df.reindex(full_range)
    reference_df = reference_df.reindex(full_range)
    sensors_df.index.name = reference_df.index.name = "time"

    sensors_df.to_parquet(s_cache)
    reference_df.to_parquet(r_cache)
    with open(m_cache, "w") as f:
        json.dump(sensor_meta, f)
    print("[data_loader] Cached to data/processed/")

    _print_summary(sensors_df, reference_df, sensor_meta)
    return sensors_df, reference_df, sensor_meta


def _print_summary(sensors_df, reference_df, sensor_meta):
    print()
    print("=" * 65)
    print("DATASET SUMMARY")
    print("=" * 65)
    print(f"  Measurand : {TARGET_MEASURAND}  |  Version: {TARGET_VERSION}")
    print(f"  Time range: {sensors_df.index[0]} to {sensors_df.index[-1]}")
    n_h = len(sensors_df)
    print(f"  Hours     : {n_h:,}  ({n_h/24:.0f} days)")
    print()
    for city, code in CITY_CODES.items():
        city_sensors = [s for s in sensors_df.columns if s.startswith(code + "__")]
        ref_col = f"{code}__NO2"
        ref_pct = (
            reference_df[ref_col].notna().mean() * 100
            if ref_col in reference_df.columns else 0
        )
        print(f"  {city} ({code}):")
        print(f"    Sensors  : {len(city_sensors)}")
        print(f"    Reference: {ref_pct:.1f}% valid")
        for sid in city_sensors:
            pct = sensors_df[sid].notna().mean() * 100
            print(f"      {sid:30s}  {pct:5.1f}% valid")
    print()
    print(f"  Total sensors: {len(sensors_df.columns)}")
    print("=" * 65)
