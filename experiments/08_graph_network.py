"""
run_gnn_comparison.py - Graph Neural Network calibration vs Random Forest.

The frontier 2024-2026 direction for network-scale calibration is graph neural
networks: model each sensor as a node, let neighbours exchange information.

In QUANT all sensors in a city are co-located, so they share one reference
target and observe the same true pollution. The GNN exploits this: at each hour
it builds a graph of the active same-city sensors (fully connected) and uses
message-passing so each sensor's calibrated value can draw on its neighbours'
readings - a learned, smart consensus that independent per-sensor models cannot do.

Implemented in plain PyTorch (masked-mean message passing, GraphSAGE-style) to
avoid heavy graph libraries. Evaluated identically to the calibrator bake-off:
per-sensor RMSE / R2 / d on the held-out future window, vs Random Forest.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from aqcn.features import build_feature_frames
from aqcn.evaluator import all_metrics

torch.manual_seed(42); np.random.seed(42)
RESULTS = Path("results")

FEATS = ["no2_raw", "o3", "no", "pm25", "temp", "rh", "pressure",
         "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TRAIN_FRAC = 0.6


def build_city_tensors(frames, city, sensors):
    """Return aligned X [T,N,F], present [T,N], y [T], index, for one city."""
    # union timeline
    idx = None
    for s in sensors:
        idx = frames[s].index if idx is None else idx.union(frames[s].index)
    idx = idx.sort_values()
    N, T, F = len(sensors), len(idx), len(FEATS)

    X = np.full((T, N, F), np.nan, dtype=np.float32)
    yser = pd.Series(np.nan, index=idx)
    for i, s in enumerate(sensors):
        d = frames[s].reindex(idx)
        for f, col in enumerate(FEATS):
            X[:, i, f] = d[col].values
        # city reference: fill from any sensor's ref_no2
        yser = yser.fillna(d["ref_no2"])
    present = ~np.isnan(X[:, :, 0])           # sensor active = has no2_raw
    return X, present, yser.values.astype(np.float32), idx


class GraphSAGECalib(nn.Module):
    """Two masked-mean message-passing layers + readout."""
    def __init__(self, f_in, hidden=64):
        super().__init__()
        self.l1_self = nn.Linear(f_in, hidden)
        self.l1_neigh = nn.Linear(f_in, hidden)
        self.l2_self = nn.Linear(hidden, hidden)
        self.l2_neigh = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)

    @staticmethod
    def neigh_mean(x, present):
        # x: [T,N,F]; present: [T,N] bool -> mean of OTHER present nodes
        m = present.unsqueeze(-1).float()
        s = (x * m).sum(dim=1, keepdim=True)          # [T,1,F]
        c = m.sum(dim=1, keepdim=True)                # [T,1,1]
        # exclude self
        neigh_sum = s - x * m
        neigh_cnt = (c - m).clamp_min(1.0)
        return neigh_sum / neigh_cnt

    def forward(self, x, present):
        a = self.neigh_mean(x, present)
        h = torch.relu(self.l1_self(x) + self.l1_neigh(a))
        a2 = self.neigh_mean(h, present)
        h = torch.relu(self.l2_self(h) + self.l2_neigh(a2))
        return self.out(h).squeeze(-1)                 # [T,N]


def main():
    RESULTS.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = build_feature_frames(use_cache=True)

    by_city = {}
    for s in frames:
        by_city.setdefault(s.split("__")[0], []).append(s)

    rf_csv = RESULTS / "calibrator_comparison_per_sensor.csv"
    rf = pd.read_csv(rf_csv) if rf_csv.exists() else None

    gnn_rows = []
    for city, sensors in by_city.items():
        if len(sensors) < 3:
            print(f"[{city}] only {len(sensors)} sensors - skipping GNN (needs a network)")
            continue
        X, present, y, idx = build_city_tensors(frames, city, sensors)
        T, N, F = X.shape

        # impute feature NaNs with train-column medians; standardise
        cut = int(T * TRAIN_FRAC)
        flat_tr = X[:cut].reshape(-1, F)
        med = np.nanmedian(flat_tr, axis=0); med = np.where(np.isnan(med), 0, med)
        Xi = np.where(np.isnan(X), med, X)
        mean = np.nanmean(Xi[:cut].reshape(-1, F), axis=0)
        std = np.nanstd(Xi[:cut].reshape(-1, F), axis=0); std[std == 0] = 1
        Xs = (Xi - mean) / std

        ymask = ~np.isnan(y)
        ym = float(np.nanmean(y[:cut][ymask[:cut]])); ys = float(np.nanstd(y[:cut][ymask[:cut]]) or 1)
        yz = (y - ym) / ys

        Xt = torch.tensor(Xs, device=device)
        Pt = torch.tensor(present, device=device)
        Yt = torch.tensor(np.nan_to_num(yz), device=device)
        Ymask = torch.tensor(ymask, device=device)

        model = GraphSAGECalib(F).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=2e-3)
        tr_t = torch.arange(cut, device=device)

        print(f"[{city}] {N} sensors, {T} hours - training GNN...")
        for epoch in range(200):
            model.train(); opt.zero_grad()
            pred = model(Xt[tr_t], Pt[tr_t])              # [cut,N]
            # loss on nodes that are present AND have a target
            valid = Pt[tr_t] & Ymask[tr_t].unsqueeze(1)
            tgt = Yt[tr_t].unsqueeze(1).expand_as(pred)
            loss = ((pred[valid] - tgt[valid]) ** 2).mean()
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred_all = model(Xt, Pt).cpu().numpy() * ys + ym   # [T,N]

        te = np.zeros(T, dtype=bool); te[cut:] = True
        for i, s in enumerate(sensors):
            mask = te & present[:, i] & ymask
            if mask.sum() < 20:
                continue
            m = all_metrics(y[mask], pred_all[mask, i])
            gnn_rows.append({"sensor_id": s, "model": "GNN",
                             "RMSE": round(m["RMSE"], 4), "R2": round(m["R2"], 4),
                             "MAE": round(m["MAE"], 4), "d": round(m["d"], 4)})

    gdf = pd.DataFrame(gnn_rows)
    gdf.to_csv(RESULTS / "gnn_per_sensor.csv", index=False)

    # ---- compare GNN vs RF on the SAME sensors ----
    L = ["=" * 64, "GRAPH NEURAL NETWORK vs RANDOM FOREST (network-scale calibration)",
         "Per-sensor, temporal 60/40 split, held-out test vs reference",
         "=" * 64]
    gnn_mean = {"RMSE": gdf["RMSE"].mean(), "R2": gdf["R2"].mean(),
                "d": gdf["d"].mean(), "n": len(gdf)}
    L.append(f"\nGNN  : RMSE={gnn_mean['RMSE']:.3f}  R2={gnn_mean['R2']:.3f}  "
             f"d={gnn_mean['d']:.3f}  ({gnn_mean['n']} sensors)")
    if rf is not None:
        common = set(gdf["sensor_id"]) & set(rf[rf.model == "RandomForest"].sensor_id)
        rf_c = rf[(rf.model == "RandomForest") & (rf.sensor_id.isin(common))]
        gnn_c = gdf[gdf.sensor_id.isin(common)]
        L.append(f"RF   : RMSE={rf_c['RMSE'].mean():.3f}  R2={rf_c['R2'].mean():.3f}  "
                 f"d={rf_c['d'].mean():.3f}  ({len(common)} matched sensors)")
        delta = (rf_c['RMSE'].mean() - gnn_c['RMSE'].mean()) / rf_c['RMSE'].mean() * 100
        L.append(f"\nGNN vs RF: {delta:+.1f}% RMSE ({'GNN better' if delta>0 else 'RF better'})")
        # per-city
        L.append("\nPer-city mean RMSE:")
        for city in ["MCH", "LON", "YRK"]:
            gc = gnn_c[gnn_c.sensor_id.str.startswith(city + "__")]
            rc = rf_c[rf_c.sensor_id.str.startswith(city + "__")]
            if len(gc):
                L.append(f"  {city}: GNN={gc['RMSE'].mean():.3f}  RF={rc['RMSE'].mean():.3f}")
    report = "\n".join(L)
    (RESULTS / "gnn_comparison.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nSaved results/gnn_comparison.{csv,txt}")


if __name__ == "__main__":
    main()
