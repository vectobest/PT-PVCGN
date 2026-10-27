"""Metrics identical to reference_PVCGN/lib/metrics.py (zero targets masked), per horizon."""
from __future__ import annotations

import numpy as np


def masked_metrics(pred: np.ndarray, true: np.ndarray, select: np.ndarray | None = None) -> dict:
    """MAE / RMSE / MAPE over entries with a non-zero target (and inside ``select`` when given)."""
    mask = true != 0
    if select is not None:
        mask &= np.broadcast_to(select, true.shape)
    err = (pred - true)[mask].astype(np.float64)
    t = true[mask].astype(np.float64)
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "mape": float(np.abs(err / t).mean())}


def per_horizon(pred: np.ndarray, true: np.ndarray, select: np.ndarray | None = None) -> list[dict]:
    """pred/true [B, H, N, 2]; select broadcastable to that shape. One dict per horizon."""
    out = []
    for h in range(true.shape[1]):
        sel = None if select is None else np.broadcast_to(select, true.shape)[:, h]
        out.append(masked_metrics(pred[:, h], true[:, h], sel))
    return out


def mean_over_horizons(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
