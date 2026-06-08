"""
download_data.py - Fetch the QUANT dataset from Zenodo into data/raw/.

QUANT: A Three-Year, Multi-City Air Quality Dataset of Commercial Air Sensors
and Reference Data for Performance Evaluation.
DOI: 10.5281/zenodo.10775692

Run once before any experiment:  python download_data.py
"""
import json
import urllib.request
from pathlib import Path

RAW = Path("data/raw")
ZENODO = "10775692"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    needed = ["QUANT_SensorSystems_hourly.csv", "QUANT_Reference_hourly.csv",
              "QUANT_DuplicateRef_hourly.csv"]
    if all((RAW / f).exists() for f in needed):
        print("[download] All QUANT files already present.")
        return
    print(f"[download] Fetching Zenodo record {ZENODO} ...")
    req = urllib.request.Request(f"https://zenodo.org/api/records/{ZENODO}",
                                 headers={"User-Agent": "AQCN/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        record = json.loads(r.read())
    for f in record["files"]:
        dest = RAW / f["key"]
        if dest.exists():
            continue
        print(f"[download] {f['key']} ({f.get('size',0)/1e6:.0f} MB) ...")
        urllib.request.urlretrieve(f["links"]["self"], dest)
    print("[download] Done -> data/raw/")


if __name__ == "__main__":
    main()
