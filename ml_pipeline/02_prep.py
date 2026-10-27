import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from metro.data import SLOTS_PER_DAY
    from metro.features import build_city, sample_index
    from ml_pipeline import guard
    return Path, SLOTS_PER_DAY, build_city, guard, json, np, pd, plt, sample_index


@app.cell
def _(build_city):
    # Step 5 - data engineering: chronological day tensors + calendar + reference days per city
    cities = {c: build_city(c) for c in ("hz", "sh")}
    {c: (cd.series.shape, int(cd.workday.sum()), int(cd.holiday.sum()), int((cd.ref_type < 0).sum()))
     for c, cd in cities.items()}
    return (cities,)


@app.cell
def _(cities, guard, np, plt):
    # Step 4 - cleaning checks (report only; every rule keeps the data)
    _f, _axes = plt.subplots(1, 2, figsize=(13, 4))
    _report = {}
    for _ax, (_c, _cd) in zip(_axes, cities.items()):
        _daily = _cd.series.sum(axis=(1, 3))  # [D, N] passengers per station-day
        _closed = _daily == 0
        _med = np.median(_cd.series[_cd.split == "train"], axis=0)  # [73, N, 2], train only
        _ratio = _cd.series / (_med[None] + 5)
        _spikes = int((_ratio > 10).sum())
        _report[_c] = dict(station_days_closed=int(_closed.sum()), stations_ever_closed=int(_closed.any(0).sum()),
                           cells_over_10x_train_median=_spikes, cells_total=int(_cd.series.size))
        _im = _ax.imshow(np.log1p(_daily.T), aspect="auto", cmap="viridis")
        _ax.set_title(f"{_c.upper()}: log(1 + daily passengers) per station")
        _ax.set_xlabel("day index")
        _ax.set_ylabel("station")
        _f.colorbar(_im, ax=_ax)
    _txt = "; ".join(f"{k.upper()}: {v['station_days_closed']} closed station-days, "
                     f"{v['cells_over_10x_train_median']} of {v['cells_total']:,} cells >10x their train median"
                     for k, v in _report.items())
    guard.fig(4, "cleaning_checks", _f,
              f"{_txt}. No station was closed on any day, and the rare spikes are plausible surges, so no row is "
              "removed; zero targets are masked in MAE/MAPE exactly as in the reference metric code.")
    _report
    return


@app.cell
def _(cities, guard, np, plt):
    # Step 5 figure: calendar and reference-day lags
    _f, _axes = plt.subplots(2, 1, figsize=(13, 5))
    for _ax, (_c, _cd) in zip(_axes, cities.items()):
        _idx = np.arange(len(_cd.dates))
        _lag_type = np.where(_cd.ref_type >= 0, _idx - _cd.ref_type, np.nan)
        _lag_week = np.where(_cd.ref_week >= 0, _idx - _cd.ref_week, np.nan)
        _ax.bar(_idx - 0.2, _lag_type, 0.4, label="days back: same day type")
        _ax.bar(_idx + 0.2, _lag_week, 0.4, label="days back: weekly reference")
        for _d in np.where(~_cd.workday)[0]:
            _ax.axvspan(_d - 0.5, _d + 0.5, color="grey", alpha=0.15, lw=0)
        _ax.set_title(f"{_c.upper()}: reference days (grey = non-working day)")
        _ax.legend(fontsize=7, loc="upper left")
    _f.tight_layout()
    _no_ref = {c: int((cd.ref_type < 0).sum()) for c, cd in cities.items()}
    guard.fig(5, "reference_days", _f,
              f"Every forecast also sees the same slots on the most recent earlier day of the same type "
              f"(working / non-working) and on the same weekday a week earlier when that day has the same type. "
              f"Days with no earlier reference ({_no_ref['hz']} HZ, {_no_ref['sh']} SH, all at the start of training) are dropped.")
    return


@app.cell
def _(SLOTS_PER_DAY, cities, np, pd):
    # Step 6 input: the evaluation table, one row per (forecast origin, horizon, station, direction)
    def eval_table(cd):
        n = cd.series.shape[2]
        origins, horizons, stations, dirs, ys = [], [], [], [], []
        for _d in range(len(cd.dates)):
            for _w in range(SLOTS_PER_DAY - 7):
                _origin = cd.dates[_d] + pd.Timedelta(minutes=330 + 15 * (_w + 3))
                _y = cd.series[_d, _w + 4:_w + 8]  # [4, N, 2]
                origins.append(np.repeat(np.datetime64(_origin, "ns"), 4 * n * 2))
                horizons.append(np.repeat(np.arange(1, 5, dtype=np.int8), n * 2))
                stations.append(np.tile(np.repeat(np.arange(n, dtype=np.int16), 2), 4))
                dirs.append(np.tile(np.array([0, 1], dtype=np.int8), 4 * n))
                ys.append(_y.reshape(-1))
        return pd.DataFrame({"origin_time": np.concatenate(origins), "horizon": np.concatenate(horizons),
                             "station": np.concatenate(stations), "direction": np.concatenate(dirs),
                             "y": np.concatenate(ys).astype(np.float32)})

    tables = {c: eval_table(cd) for c, cd in cities.items()}
    {c: t.shape for c, t in tables.items()}
    return (tables,)


@app.cell
def _(cities, guard, np, tables):
    # Step 6 - split with the guard (chronological), sized to reproduce the released splits exactly
    sizes = {"hz": (5 / 25, 2 / 25), "sh": (21 / 92, 9 / 92)}
    split_check = {}
    for _c, (_ts, _vs) in sizes.items():
        _tr, _va, _te = guard.split(tables[_c], target="y", time_col="origin_time",
                                    test_size=_ts, val_size=_vs, state_dir=f"ml_pipeline/{_c}")
        _cd = cities[_c]
        for _name, _part in (("train", _tr), ("val", _va), ("test", _te)):
            _days = set(_part["origin_time"].dt.normalize().unique())
            _released = set(_cd.dates[_cd.split == _name])
            assert _days == _released, (_c, _name)
        split_check[_c] = {k: len(v) for k, v in (("train", _tr), ("val", _va), ("test", _te))}
        del _tr, _va, _te
    split_check
    return (split_check,)


@app.cell
def _(cities, guard, plt, split_check):
    _f, _axes = plt.subplots(2, 1, figsize=(12, 3.5))
    for _ax, (_c, _cd) in zip(_axes, cities.items()):
        for _name, _col in (("train", "C0"), ("val", "C1"), ("test", "C2")):
            _days = _cd.dates[_cd.split == _name]
            _ax.barh(0, (_days.max() - _days.min()).days + 1, left=_days.min(), color=_col, label=_name)
        _ax.set_yticks([])
        _ax.set_title(f"{_c.upper()} chronological split (guard-verified, test frozen)")
        _ax.legend(loc="center left", fontsize=7)
    _f.tight_layout()
    guard.fig(6, "split_timeline", _f,
              "guard.split reproduces the paper's released date split exactly (rows: "
              + "; ".join(f"{c.upper()} {v['train']:,}/{v['val']:,}/{v['test']:,}" for c, v in split_check.items())
              + " train/val/test); test fingerprints are frozen for the single final evaluation.")
    return


@app.cell
def _(cities, guard, np, plt, sample_index):
    # Step 7 - how strong is each engineered input on TRAINING data only (target = 60-min-ahead inflow)
    _f, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _out = {}
    for _ax, (_c, _cd) in zip(_axes, cities.items()):
        _idx = sample_index(_cd, "train")
        _d, _w = _idx[:, 0], _idx[:, 1]
        _tgt = _cd.series[_d, _w + 7, :, 0].ravel()
        _cands = {
            "last observed (t)": _cd.series[_d, _w + 3, :, 0].ravel(),
            "same-type day": _cd.series[_cd.ref_type[_d], _w + 7, :, 0].ravel(),
            "weekly ref": _cd.series[_cd.ref_week[_d], _w + 7, :, 0].ravel(),
        }
        _r = {k: float(np.corrcoef(_tgt, v)[0, 1]) for k, v in _cands.items()}
        _ax.bar(list(_r), list(_r.values()), color=["C7", "C0", "C1"])
        _ax.set_ylim(0.5, 1)
        _ax.set_title(f"{_c.upper()} train: r with inflow 60 min ahead")
        _out[_c] = _r
    guard.fig(7, "reference_feature_strength", _f,
              "On training data the value 60 min ahead correlates "
              + "; ".join(f"{c.upper()} r={v['same-type day']:.3f} with the same-type reference day vs r={v['last observed (t)']:.3f} with the last observation"
                          for c, v in _out.items())
              + " - the periodic inputs carry most of the long-horizon signal.")
    return


@app.cell
def _(cities, guard, json, np, plt):
    # Step 8 - preprocessing: one z-score scaler per city, fit on training days only
    scalers = {}
    for _c, _cd in cities.items():
        _tr = _cd.series[_cd.split == "train"]
        scalers[_c] = {"mean": float(_tr.mean()), "std": float(_tr.std())}
        with open(f"ml_pipeline/{_c}/scaler.json", "w") as _fh:
            json.dump(scalers[_c], _fh, indent=2)
    _f, _axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for _ax, (_c, _cd) in zip(_axes, cities.items()):
        for _name in ("train", "val", "test"):
            _z = (_cd.series[_cd.split == _name] - scalers[_c]["mean"]) / scalers[_c]["std"]
            _ax.hist(_z.ravel(), bins=100, range=(-1, 8), density=True, histtype="step", label=_name)
        _ax.set_yscale("log")
        _ax.set_title(f"{_c.upper()}: z-scored counts (train mean/std)")
        _ax.legend()
    guard.fig(8, "scaled_distributions", _f,
              "Counts z-scored with the training mean/std only ("
              + "; ".join(f"{c.upper()} mean {s['mean']:.1f}, std {s['std']:.1f}" for c, s in scalers.items())
              + "); SH validation/test match training closely, while HZ validation (a weekend) has far fewer high-volume intervals, so HZ validation scores will understate weekday difficulty.")
    scalers
    return (scalers,)


@app.cell
def _(cities, np):
    # Persist engineered artefacts for the modelling notebook
    for _c, _cd in cities.items():
        np.savez_compressed(f"ml_pipeline/{_c}/features.npz",
                            series=_cd.series, dates=_cd.dates.values.astype("datetime64[ns]"),
                            split=_cd.split, dow=_cd.dow, workday=_cd.workday, holiday=_cd.holiday,
                            ref_type=_cd.ref_type, ref_week=_cd.ref_week)
    return


if __name__ == "__main__":
    app.run()
