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

    from metro import baselines as bl
    from metro.batches import load_city
    from metro.metrics import mean_over_horizons, per_horizon
    from ml_pipeline import guard
    return ROOT, bl, guard, json, load_city, mean_over_horizons, np, per_horizon, plt


@app.cell
def _(bl, json, load_city, mean_over_horizons, per_horizon):
    # Step 9 - baselines, scored on VALIDATION only (cached: delete the json to refit)
    import os as _os
    if _os.path.exists("ml_pipeline/runs/baselines_val.json"):
        with open("ml_pipeline/runs/baselines_val.json") as _fh:
            baseline_val = json.load(_fh)
    else:
        baseline_val = {}
    for _c in [c for c in ("hz", "sh") if c not in baseline_val]:
        _ct = load_city(_c, "cpu")
        _tr, _va = _ct.windows("train"), _ct.windows("val")
        _true = _ct.targets(_va)
        _table = bl.historical_average_table(_ct)
        _preds = {
            "Persistence": bl.persistence(_ct, _va),
            "Seasonal naive (same-type day)": bl.seasonal(_ct, _va, "ref_type"),
            "Seasonal naive (weekly)": bl.seasonal(_ct, _va, "ref_week"),
            "Historical average": bl.historical_average(_ct, _va, _table),
            "Ridge (lags+refs)": bl.fit_predict_tabular(_ct, _tr, _va, "ridge"),
            "GBDT (lags+refs)": bl.fit_predict_tabular(_ct, _tr, _va, "hgb"),
        }
        baseline_val[_c] = {k: {"per_horizon": per_horizon(v, _true), "mean": mean_over_horizons(per_horizon(v, _true))}
                            for k, v in _preds.items()}
    with open("ml_pipeline/runs/baselines_val.json", "w") as _fh:
        json.dump(baseline_val, _fh, indent=1)
    {c: {k: round(v["mean"]["mae"], 2) for k, v in d.items()} for c, d in baseline_val.items()}
    return (baseline_val,)


@app.cell
def _(baseline_val, guard, np, plt):
    _f, _axes = plt.subplots(1, 2, figsize=(13, 4))
    for _ax, (_c, _d) in zip(_axes, baseline_val.items()):
        _names = list(_d)
        for _i, _n in enumerate(_names):
            _ax.bar(np.arange(4) + (_i - len(_names) / 2) * 0.13, [r["mae"] for r in _d[_n]["per_horizon"]], 0.13, label=_n)
        _ax.set_xticks(range(4))
        _ax.set_xticklabels(["15 min", "30 min", "45 min", "60 min"])
        _ax.set_ylabel("validation MAE (passengers / 15 min)")
        _ax.set_title(f"{_c.upper()} baselines (validation)")
    _axes[0].legend(fontsize=7)
    _best = {c: min(d, key=lambda k: d[k]["mean"]["mae"]) for c, d in baseline_val.items()}
    guard.fig(9, "baselines_val", _f,
              "Validation MAE of the six baselines; the best is "
              + "; ".join(f"{c.upper()}: {b} (MAE {baseline_val[c][b]['mean']['mae']:.2f})" for c, b in _best.items())
              + ". Every deep model must beat this line to justify its cost.")
    return


@app.cell
def _(json, np):
    # Steps 10-11 - collect every training run (validation scores only)
    import glob as _glob
    runs = {}
    for _f in sorted(_glob.glob("ml_pipeline/runs/*_s*.json")):
        _d = json.load(open(_f))
        if not _d.get("finished"):
            continue
        _city, _variant, _seed = _d["config"]["city"], _d["config"]["variant"] + _d["config"].get("tag", ""), _d["config"]["seed"]
        runs.setdefault((_city, _variant), []).append(_d)
    summary = {}
    for (_c, _v), _ds in runs.items():
        _ph = np.array([[h["mae"] for h in d["best_val_per_horizon"]] for d in _ds])
        _m = np.array([[d["best"]["val_mae"], d["best"]["val_rmse"], d["best"]["val_mape"]] for d in _ds])
        summary[(_c, _v)] = {"n": len(_ds), "mae": _m[:, 0].mean(), "mae_std": _m[:, 0].std(), "rmse": _m[:, 1].mean(),
                             "mape": _m[:, 2].mean(), "per_h": _ph.mean(0), "per_h_std": _ph.std(0)}
    {f"{c}/{v}": (s["n"], round(s["mae"], 2), round(s["mae_std"], 2)) for (c, v), s in summary.items()}
    return runs, summary


@app.cell
def _(guard, np, plt, runs):
    # Step 10 figure - validation learning curves, Hangzhou
    _names = {"pvcgn": "PVCGN", "p": "+P", "pt": "PT-PVCGN (+P+T)"}
    _cols = {"pvcgn": "C7", "p": "C0", "pt": "C2"}
    _f, _ax = plt.subplots(figsize=(9, 4.5))
    _ends = {}
    for _v in ("pvcgn", "p", "pt"):
        for _i, _d in enumerate(runs.get(("hz", _v), [])):
            _e = [r["val_mae"] for r in _d["epochs"]]
            _ax.plot(range(1, len(_e) + 1), _e, color=_cols[_v], alpha=0.8, lw=1.2, label=_names[_v] if _i == 0 else None)
            _ends.setdefault(_v, []).append(min(_e[74:]) - min(_e))
    _ax.set_ylim(18, 40)
    _ax.axvline(50, color="grey", ls=":", lw=1)
    _ax.axvline(75, color="grey", ls=":", lw=1)
    _ax.set_xlabel("epoch (learning rate x0.1 at 50 and 75)")
    _ax.set_ylabel("validation MAE")
    _ax.set_title("Hangzhou: validation MAE per epoch, 3 seeds per model")
    _ax.legend()
    _pv = np.mean([min(r["val_mae"] for r in d["epochs"][:75]) - d["best"]["val_mae"] for d in runs[("hz", "pvcgn")]])
    _pt = np.mean([min(r["val_mae"] for r in d["epochs"][:75]) - d["best"]["val_mae"] for d in runs[("hz", "pt")]])
    guard.fig(10, "hz_learning_curves", _f,
              f"PT-PVCGN converges faster and lower than PVCGN on every seed; PVCGN still gained {_pv:.2f} MAE after epoch 75 "
              f"(PT-PVCGN {_pt:.2f}), so PVCGN may be slightly under-trained at 100 epochs - checked before the final test.")
    return


@app.cell
def _(baseline_val, guard, np, plt, summary):
    # Step 12 figure - validation MAE per horizon vs baselines, Hangzhou
    _f, _ax = plt.subplots(figsize=(10, 4.8))
    _h = np.arange(4)
    _bl = ["Historical average", "Seasonal naive (weekly)", "GBDT (lags+refs)", "Ridge (lags+refs)"]
    for _i, _n in enumerate(_bl):
        _ax.plot(_h, [r["mae"] for r in baseline_val["hz"][_n]["per_horizon"]], "--", marker="o", color=f"C{_i+3}", lw=1, label=_n)
    for _v, _lab, _c in (("pvcgn", "PVCGN", "C7"), ("p", "+P", "C0"), ("pt", "PT-PVCGN", "C2")):
        _s = summary[("hz", _v)]
        _ax.errorbar(_h, _s["per_h"], yerr=_s["per_h_std"], marker="s", lw=2.2, capsize=3, color=_c, label=f"{_lab} (mean of {_s['n']})")
    _ax.set_xticks(_h)
    _ax.set_xticklabels(["15 min", "30 min", "45 min", "60 min"])
    _ax.set_ylabel("validation MAE")
    _ax.set_title("Hangzhou validation MAE by horizon")
    _ax.legend(fontsize=8, ncol=2)
    _pv, _pt = summary[("hz", "pvcgn")], summary[("hz", "pt")]
    _ridge = baseline_val["hz"]["Ridge (lags+refs)"]["mean"]["mae"]
    guard.fig(12, "hz_val_by_horizon", _f,
              f"On validation PT-PVCGN averages MAE {_pt['mae']:.2f} +- {_pt['mae_std']:.2f} vs PVCGN {_pv['mae']:.2f} +- {_pv['mae_std']:.2f} "
              f"and the best baseline (ridge) {_ridge:.2f}; its advantage over PVCGN grows with horizon "
              f"({_pv['per_h'][0]-_pt['per_h'][0]:.2f} at 15 min, {_pv['per_h'][3]-_pt['per_h'][3]:.2f} at 60 min).")
    return


@app.cell
def _(guard, np, plt):
    # Step 12 figure - where the gain comes from: validation error by time of day (seed 1 checkpoints)
    import torch as _torch
    from metro.batches import load_city as _load
    from metro.train import CKPT as _CKPT, build_model as _build, predict as _predict
    _ct = _load("hz", "cpu")
    _va = _ct.windows("val")
    _true = _ct.targets(_va)
    _err = {}
    for _v in ("pvcgn", "pt"):
        _m = _build(_ct, _v)
        _m.load_state_dict(_torch.load(_CKPT / f"hz_{_v}_s1.pt", map_location="cpu"))
        _pred = _predict(_m, _ct, _va, 32)
        _err[_v] = np.abs(_pred - _true)                       # [B, 4, N, 2]
    _slot = _va[:, 1][:, None] + np.arange(4, 8)               # target slot per horizon
    _hours = 5.5 + _slot * 0.25
    _f, _axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for _v, _lab, _c in (("pvcgn", "PVCGN", "C7"), ("pt", "PT-PVCGN", "C2")):
        _per = [(_err[_v][_slot == s]).mean() for s in range(4, 73)]
        _axes[0].plot(5.5 + np.arange(4, 73) * 0.25, _per, color=_c, label=_lab)
    _axes[0].set_xlabel("hour of day (target interval)")
    _axes[0].set_ylabel("validation MAE")
    _axes[0].set_title("Error through the day (seed 1, validation = weekend)")
    _axes[0].legend()
    _big = _true > np.quantile(_true, 0.9)
    _axes[1].scatter(_true[_big][::7], (_err["pvcgn"][_big] - _err["pt"][_big])[::7], s=2, alpha=0.3)
    _axes[1].axhline(0, color="grey", lw=1)
    _axes[1].set_xlabel("actual count (top 10% busiest station-intervals)")
    _axes[1].set_ylabel("PVCGN error - PT-PVCGN error")
    _axes[1].set_title("Positive = PT-PVCGN closer to the truth")
    _share = float((_err["pt"] < _err["pvcgn"]).mean())
    guard.fig(12, "hz_val_error_by_time", _f,
              f"PT-PVCGN is closer to the truth on {_share:.0%} of validation targets; its gain is largest in the early morning and "
              "the morning and evening peaks, while for the busiest 10% of station-intervals the difference is mixed (points on both "
              "sides of zero), so the improvement comes mainly from getting the daily shape right rather than from extreme counts.")
    return


@app.cell
def _(baseline_val, guard, np, plt, summary):
    # Step 12 figure - equal training budget on Hangzhou (all models 200-epoch schedule) and Shanghai (all 30 epochs)
    _f, _axes = plt.subplots(1, 2, figsize=(14, 4.8))
    _h = np.arange(4)
    for _ax, _city, _models, _ttl in (
        (_axes[0], "hz", (("pvcgn_e200", "PVCGN", "C7"), ("p_e200", "+P", "C0"), ("pt_e200", "+P+T", "C2")), "Hangzhou, 200-epoch schedule"),
        (_axes[1], "sh", (("pvcgn", "PVCGN", "C7"), ("pt", "+P+T", "C2")), "Shanghai, 30 epochs"),
    ):
        _r = baseline_val[_city]["Ridge (lags+refs)"]["per_horizon"]
        _ax.plot(_h, [r["mae"] for r in _r], "--", marker="o", color="C3", lw=1, label="Ridge (lags+refs)")
        for _v, _lab, _c in _models:
            _s = summary[(_city, _v)]
            _ax.errorbar(_h, _s["per_h"], yerr=_s["per_h_std"], marker="s", lw=2.2, capsize=3, color=_c, label=f"{_lab} (mean of {_s['n']})")
        _ax.set_xticks(_h)
        _ax.set_xticklabels(["15 min", "30 min", "45 min", "60 min"])
        _ax.set_ylabel("validation MAE")
        _ax.set_title(_ttl)
        _ax.legend(fontsize=8)
    _hz = {k: summary[("hz", k)] for k in ("pvcgn_e200", "p_e200", "pt_e200")}
    _sh = {k: summary[("sh", k)] for k in ("pvcgn", "pt")}
    guard.fig(12, "val_equal_budget", _f,
              f"With equal training, Hangzhou validation MAE is PVCGN {_hz['pvcgn_e200']['mae']:.2f}+-{_hz['pvcgn_e200']['mae_std']:.2f}, "
              f"+P {_hz['p_e200']['mae']:.2f}+-{_hz['p_e200']['mae_std']:.2f}, +P+T {_hz['pt_e200']['mae']:.2f}+-{_hz['pt_e200']['mae_std']:.2f}; "
              f"Shanghai PVCGN {_sh['pvcgn']['mae']:.2f} vs +P+T {_sh['pt']['mae']:.2f}. Periodic inputs give the gain; calendar embeddings "
              "add nothing on Hangzhou under the 200-epoch schedule.")
    return


if __name__ == "__main__":
    app.run()
