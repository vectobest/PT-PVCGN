"""Shared evaluation code for step 13 (error analysis, validation) and step 14 (final test).

Every model in the pre-registered set is scored with the same metric code and the same slices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metro import baselines as bl
from metro.batches import ROOT, load_city
from metro.data import SLOTS_PER_DAY
from metro.metrics import mean_over_horizons, per_horizon
from metro.train import CKPT, build_model, predict

RUNS = ROOT / "ml_pipeline" / "runs"
# (name in tables, variant, run-name tag) per city; seeds are discovered from finished runs
DEEP_MODELS = {
    "hz": [("PVCGN", "pvcgn", "_e200"), ("+P", "p", "_e200"), ("PT-PVCGN (+P+T)", "pt", "_e200"),
           ("PVCGN [100 ep]", "pvcgn", ""), ("+P [100 ep]", "p", ""), ("PT-PVCGN (+P+T) [100 ep]", "pt", "")],
    "sh": [("PVCGN", "pvcgn", ""), ("PT-PVCGN (+P+T)", "pt", "")],
}
MAIN_MODEL = "PT-PVCGN (+P+T)"
# Test-set numbers reported for PVCGN in the original paper (Tables II and III), per horizon, for context only.
PAPER_PVCGN = {
    "hz": {"mae": [22.68, 23.33, 24.22, 24.93], "rmse": [37.76, 39.34, 40.95, 42.61], "mape": [0.1370, 0.1381, 0.1445, 0.1549]},
    "sh": {"mae": [23.29, 24.16, 25.33, 26.29], "rmse": [44.97, 47.83, 52.02, 55.27], "mape": [0.1683, 0.1723, 0.1792, 0.1869]},
}
SPLIT_SIZES = {"hz": (5 / 25, 2 / 25), "sh": (21 / 92, 9 / 92)}


def finished_seeds(city: str, variant: str, tag: str) -> list[int]:
    seeds = []
    for f in sorted(RUNS.glob(f"{city}_{variant}_s*{tag}.json")):
        d = json.loads(f.read_text())
        c = d["config"]
        if d.get("finished") and c["variant"] == variant and c.get("tag", "") == tag:
            seeds.append(c["seed"])
    return sorted(seeds)


def deep_predictions(ct, idx: np.ndarray, device: str) -> dict[str, list[np.ndarray]]:
    """{model name: [per-seed raw-count predictions [B, 4, N, 2]]} from the saved best-on-validation checkpoints."""
    out = {}
    batch = 32 if ct.city == "hz" else 8
    ct_dev = load_city(ct.city, device)
    for name, variant, tag in DEEP_MODELS[ct.city]:
        preds = []
        for seed in finished_seeds(ct.city, variant, tag):
            model = build_model(ct_dev, variant).to(device)
            model.load_state_dict(torch.load(CKPT / f"{ct.city}_{variant}_s{seed}{tag}.pt", map_location=device))
            preds.append(predict(model, ct_dev, idx, batch).astype(np.float32))
            del model
        out[name] = preds
    return out


def baseline_predictions(ct, idx: np.ndarray) -> dict[str, list[np.ndarray]]:
    """Baselines fitted on training windows only."""
    tr = ct.windows("train")
    table = bl.historical_average_table(ct)
    return {
        "Persistence": [bl.persistence(ct, idx)],
        "Seasonal naive (same-type day)": [bl.seasonal(ct, idx, "ref_type")],
        "Seasonal naive (weekly)": [bl.seasonal(ct, idx, "ref_week")],
        "Historical average": [bl.historical_average(ct, idx, table)],
        "Ridge (lags+refs)": [bl.fit_predict_tabular(ct, tr, idx, "ridge")],
        "GBDT (lags+refs)": [bl.fit_predict_tabular(ct, tr, idx, "hgb")],
    }


def slices(ct, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks broadcastable to [B, 4, N, 2]."""
    b, n = len(idx), ct.num_nodes
    target_slot = idx[:, 1][:, None] + np.arange(4, 8)                    # [B, 4]
    minutes = 330 + 15 * target_slot                                       # interval end, minutes after midnight
    rush = ((minutes > 450) & (minutes <= 570)) | ((minutes > 1050) & (minutes <= 1170))  # 07:30-09:30, 17:30-19:30
    train_total = ct.raw[ct.split == "train"].sum((0, 1, 3))
    top = train_total >= np.quantile(train_total, 0.75)                    # top 1/4 stations by training ridership
    dates = pd.DatetimeIndex(ct.dates[idx[:, 0]]).strftime("%Y-%m-%d")
    wd = ct.workday.cpu().numpy()[idx[:, 0]].astype(bool)
    out = {
        "all": np.ones((b, 1, 1, 1), bool),
        "rush hours": rush[:, :, None, None],
        "top 1/4 stations": top[None, None, :, None],
        "working days": wd[:, None, None, None],
        "non-working days": ~wd[:, None, None, None],
    }
    if ct.city == "sh":
        out["holiday (Sep 15-17)"] = np.isin(dates, ["2016-09-15", "2016-09-16", "2016-09-17"])[:, None, None, None]
        out["make-up workday (Sep 18)"] = (dates == "2016-09-18")[:, None, None, None]
    return {k: np.broadcast_to(v, (b, 4, n, 2)) for k, v in out.items()}


def score(preds: dict[str, list[np.ndarray]], true: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    """{model: {slice: {per_horizon_mean, per_horizon_std, mean, std, n_seeds}}} - std over seeds."""
    res = {}
    for name, seed_preds in preds.items():
        res[name] = {}
        for sname, m in masks.items():
            if not m.any():
                continue
            ph = [per_horizon(p, true, m) for p in seed_preds]               # seeds x horizons x metrics
            arr = {k: np.array([[h[k] for h in s] for s in ph]) for k in ("mae", "rmse", "mape")}
            res[name][sname] = {
                "n_seeds": len(seed_preds),
                "per_horizon": {k: v.mean(0).tolist() for k, v in arr.items()},
                "per_horizon_std": {k: v.std(0).tolist() for k, v in arr.items()},
                "mean": {k: float(v.mean(1).mean()) for k, v in arr.items()},
                "std": {k: float(v.mean(1).std()) for k, v in arr.items()},
            }
    return res


def eval_table(raw: np.ndarray, dates) -> pd.DataFrame:
    """Identical to the table frozen at step 6 (ml_pipeline/02_prep.py)."""
    dates = pd.DatetimeIndex(dates)
    n = raw.shape[2]
    origins, horizons, stations, dirs, ys = [], [], [], [], []
    for d in range(len(dates)):
        for w in range(SLOTS_PER_DAY - 7):
            origin = dates[d] + pd.Timedelta(minutes=330 + 15 * (w + 3))
            y = raw[d, w + 4:w + 8]
            origins.append(np.repeat(np.datetime64(origin, "ns"), 4 * n * 2))
            horizons.append(np.repeat(np.arange(1, 5, dtype=np.int8), n * 2))
            stations.append(np.tile(np.repeat(np.arange(n, dtype=np.int16), 2), 4))
            dirs.append(np.tile(np.array([0, 1], dtype=np.int8), 4 * n))
            ys.append(y.reshape(-1))
    return pd.DataFrame({"origin_time": np.concatenate(origins), "horizon": np.concatenate(horizons),
                         "station": np.concatenate(stations), "direction": np.concatenate(dirs),
                         "y": np.concatenate(ys).astype(np.float32)})


def frozen_test_frame(ct) -> pd.DataFrame:
    """Rebuild the exact test frame frozen by guard.split (same rows, same order)."""
    from ml_pipeline.guard import _time_indices
    df = eval_table(ct.raw, ct.dates).drop_duplicates().reset_index(drop=True)
    ts, vs = SPLIT_SIZES[ct.city]
    n = len(df)
    n_test, n_val = int(round(n * ts)), int(round(n * vs))
    _, _, te = _time_indices(df, "origin_time", n - n_val - n_test, n_val)
    return df.iloc[np.sort(te)].reset_index(drop=True)
