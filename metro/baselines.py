"""Non-deep baselines: persistence, seasonal naive, historical average, ridge and gradient boosting.

All statistics and fitted models use training days only.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def persistence(ct, idx):
    last = ct.raw[idx[:, 0], idx[:, 1] + 3]                       # [B, N, 2]
    return np.repeat(last[:, None], 4, axis=1)


def seasonal(ct, idx, which: str):
    ref = getattr(ct, which).cpu().numpy()[idx[:, 0]]
    return ct.raw[ref[:, None], idx[:, 1][:, None] + np.arange(4, 8)]


def historical_average_table(ct):
    """[2 day types, 73, N, 2] mean count per (working-day flag, slot, station) over training days."""
    wd = ct.workday.cpu().numpy()
    tr = ct.split == "train"
    return np.stack([ct.raw[tr & (wd == k)].mean(0) for k in (0, 1)])


def historical_average(ct, idx, table):
    wd = ct.workday.cpu().numpy()[idx[:, 0]]
    return table[wd[:, None], idx[:, 1][:, None] + np.arange(4, 8)]


def lag_features(ct, idx, h: int, direction: int, table, station_mean):
    """One row per (window, station) for target horizon h (0..3) and direction (0 in, 1 out)."""
    d, w = idx[:, 0], idx[:, 1]
    wd = ct.workday.cpu().numpy()[d]
    rt = ct.ref_type.cpu().numpy()[d]
    rw = ct.ref_week.cpu().numpy()[d]
    n = ct.num_nodes
    recent = ct.raw[d[:, None], w[:, None] + np.arange(4)]                  # [B, 4, N, 2]
    ref_t = ct.raw[rt[:, None], w[:, None] + np.arange(8)]                  # [B, 8, N, 2]
    ref_w = ct.raw[rw[:, None], w[:, None] + np.arange(8)]
    target_slot = w + 4 + h
    cols = [
        recent[..., direction].transpose(0, 2, 1),                        # 4 recent lags, same direction
        recent[..., 1 - direction].transpose(0, 2, 1),                    # 4 recent lags, other direction
        ref_t[..., direction].transpose(0, 2, 1),                         # same-type day, slots w..w+7
        ref_w[..., direction].transpose(0, 2, 1),                         # weekly ref, slots w..w+7
        table[wd, target_slot][..., direction][..., None],                # historical average at target
        np.repeat(station_mean[None, :, direction, None], len(idx), 0),   # station level (train)
        np.repeat(np.stack([np.sin(2 * np.pi * target_slot / 73), np.cos(2 * np.pi * target_slot / 73),
                            wd.astype(float)], 1)[:, None], n, 1),
    ]
    return np.concatenate(cols, -1).reshape(len(idx) * n, -1)


def fit_predict_tabular(ct, train_idx, eval_idx, kind: str, seed: int = 0):
    """Fit one model per (horizon, direction) on training windows, predict eval windows -> [B, 4, N, 2]."""
    table = historical_average_table(ct)
    station_mean = ct.raw[ct.split == "train"].mean((0, 1))                 # [N, 2]
    y_tr = ct.targets(train_idx)
    out = np.zeros((len(eval_idx), 4, ct.num_nodes, 2), dtype=np.float32)
    for h in range(4):
        for k in range(2):
            Xtr = lag_features(ct, train_idx, h, k, table, station_mean)
            ytr = y_tr[:, h, :, k].reshape(-1)
            if kind == "ridge":
                model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            else:
                model = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.1,
                                                      max_leaf_nodes=63, early_stopping=False,
                                                      random_state=seed)
            model.fit(Xtr, ytr)
            Xev = lag_features(ct, eval_idx, h, k, table, station_mean)
            out[:, h, :, k] = np.clip(model.predict(Xev), 0, None).reshape(len(eval_idx), ct.num_nodes)
    return out
