"""Loading the PVCGN benchmark (HZMetro / SHMetro) and rebuilding continuous daily series.

The released pickles store overlapping (x, y) windows: per day, 66 windows of 4 input and
4 output 15-minute intervals covering 05:30-23:30 (73 intervals). We rebuild the per-day
series [days, 73, N, 2] so that features from the history (e.g. the same slot on the previous
day) can be built without leaking the future.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reference_PVCGN" / "data"
CITIES = {
    "hz": ("hangzhou", 80),
    "sh": ("shanghai", 288),
}
SPLITS = ("train", "val", "test")
WINDOWS_PER_DAY = 66
SEQ_LEN = 4
SLOTS_PER_DAY = WINDOWS_PER_DAY + 2 * SEQ_LEN - 1  # 73 intervals, 05:30 .. 23:30


def load_split(city: str, split: str) -> dict:
    folder, _ = CITIES[city]
    with open(DATA_DIR / folder / f"{split}.pkl", "rb") as fh:
        d = pickle.load(fh)
    return {k: np.asarray(v) for k, v in d.items()}


def load_graphs(city: str) -> dict[str, np.ndarray]:
    folder, _ = CITIES[city]
    out = {}
    for name in ("conn", "sml", "cor"):
        with open(DATA_DIR / folder / f"graph_{city}_{name}.pkl", "rb") as fh:
            out[name] = np.asarray(pickle.load(fh), dtype=np.float64)
    return out


def windows_to_days(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild [days, 73, N, 2] series and [days, 73] timestamps from overlapping windows."""
    x, y, xt, yt = d["x"], d["y"], d["xtime"], d["ytime"]
    n_days = len(x) // WINDOWS_PER_DAY
    assert n_days * WINDOWS_PER_DAY == len(x), "windows are not whole days"
    series, stamps = [], []
    for k in range(n_days):
        s = slice(k * WINDOWS_PER_DAY, (k + 1) * WINDOWS_PER_DAY)
        xs, ys, xts, yts = x[s], y[s], xt[s], yt[s]
        day = np.concatenate([xs[0], ys[:, 0], ys[-1, 1:]], axis=0)
        ts = np.concatenate([xts[0], yts[:, 0], yts[-1, 1:]], axis=0)
        # consistency: every window must be a contiguous slice of the rebuilt day
        for w in range(WINDOWS_PER_DAY):
            assert np.array_equal(day[w:w + SEQ_LEN], xs[w])
            assert np.array_equal(day[w + SEQ_LEN:w + 2 * SEQ_LEN], ys[w])
        series.append(day)
        stamps.append(ts)
    return np.stack(series).astype(np.float32), np.stack(stamps)


def long_table(city: str) -> pd.DataFrame:
    """One row per (interval end time, station): inflow and outflow counts, all splits."""
    frames = []
    for split in SPLITS:
        series, stamps = windows_to_days(load_split(city, split))
        days, slots, n, _ = series.shape
        ts = np.repeat(stamps.reshape(-1), n)
        frames.append(pd.DataFrame({
            "city": city,
            "split_released": split,
            "timestamp": pd.to_datetime(ts),
            "station": np.tile(np.arange(n), days * slots),
            "inflow": series[..., 0].reshape(-1),
            "outflow": series[..., 1].reshape(-1),
        }))
    df = pd.concat(frames, ignore_index=True)
    df["slot"] = (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute - 330) // 15
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df
