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
    import torch

    from metro import evaluate as ev
    from metro.batches import load_city
    from metro.metrics import masked_metrics
    from ml_pipeline import guard

    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    return DEVICE, ev, guard, json, load_city, masked_metrics, np, plt


@app.cell
def _(DEVICE, ev, json, load_city):
    # Step 13 - error analysis on VALIDATION (before the test set is touched)
    val_scores = {}
    val_preds = {}
    for _c in ("hz", "sh"):
        _ct = load_city(_c, "cpu")
        _idx = _ct.windows("val")
        _p = {**ev.deep_predictions(_ct, _idx, DEVICE), **ev.baseline_predictions(_ct, _idx)}
        val_preds[_c] = (_ct, _idx, _p)
        val_scores[_c] = ev.score(_p, _ct.targets(_idx), ev.slices(_ct, _idx))
    with open("ml_pipeline/runs/val_results.json", "w") as _fh:
        json.dump(val_scores, _fh, indent=1)
    {c: {m: round(s["all"]["mean"]["mae"], 2) for m, s in d.items()} for c, d in val_scores.items()}
    return val_preds, val_scores


@app.cell
def _(ev, guard, np, plt, val_scores):
    # Step 13 figure - validation MAE by slice: PVCGN vs PT-PVCGN vs best baseline
    _f, _axes = plt.subplots(1, 2, figsize=(14, 4.6))
    _lines = []
    for _ax, _c in zip(_axes, ("hz", "sh")):
        _s = val_scores[_c]
        _sl = [k for k in _s[ev.MAIN_MODEL] if k != "all"]
        _x = np.arange(len(_sl))
        for _i, (_m, _col) in enumerate((("PVCGN", "C7"), ("Ridge (lags+refs)", "C3"), (ev.MAIN_MODEL, "C2"))):
            _ax.bar(_x + (_i - 1) * 0.27, [_s[_m][k]["mean"]["mae"] for k in _sl], 0.27, color=_col, label=_m)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_sl, rotation=20, ha="right", fontsize=8)
        _ax.set_ylabel("validation MAE")
        _ax.set_title({"hz": "Hangzhou (200-epoch schedule)", "sh": "Shanghai (30 epochs)"}[_c])
        _worse = [k for k in _sl if _s[ev.MAIN_MODEL][k]["mean"]["mae"] > _s["PVCGN"][k]["mean"]["mae"]]
        _rel = {k: 1 - _s[ev.MAIN_MODEL][k]["mean"]["mae"] / _s["PVCGN"][k]["mean"]["mae"] for k in _sl}
        _best = max(_rel, key=_rel.get)
        _lines.append(f"{_c.upper()}: PT-PVCGN beats PVCGN on {len(_sl) - len(_worse)} of {len(_sl)} slices"
                      + (f" (worse on: {', '.join(_worse)})" if _worse else "")
                      + f", largest relative gain on {_best} ({_rel[_best]:.0%})")
    _axes[0].legend(fontsize=8)
    guard.fig(13, "val_error_slices", _f,
              "; ".join(_lines) + ". Slices with no validation days (e.g. HZ working days, SH holiday) are absent.")
    return


@app.cell
def _(ev, guard, np, plt, val_preds):
    # Step 13 figure - per-station validation error: where does PT-PVCGN do worse than PVCGN?
    _f, _axes = plt.subplots(1, 2, figsize=(13, 5))
    _txt = []
    for _ax, _c in zip(_axes, ("hz", "sh")):
        _ct, _idx, _p = val_preds[_c]
        _true = _ct.targets(_idx)
        _mask = _true != 0

        def _station_mae(preds):
            e = np.mean([np.abs(p - _true) for p in preds], 0)
            return np.array([e[..., s, :][_mask[..., s, :]].mean() for s in range(_ct.num_nodes)])

        _a, _b = _station_mae(_p["PVCGN"]), _station_mae(_p[ev.MAIN_MODEL])
        _vol = _ct.raw[_ct.split == "train"].mean((0, 1, 3))
        _sc = _ax.scatter(_a, _b, c=np.log10(_vol + 1), s=12, cmap="viridis")
        _lim = [0, max(_a.max(), _b.max()) * 1.05]
        _ax.plot(_lim, _lim, color="grey", lw=1)
        _ax.set_xlabel("PVCGN station MAE")
        _ax.set_ylabel("PT-PVCGN station MAE")
        _ax.set_title(f"{_c.upper()}: one dot per station (colour = log10 mean volume)")
        _txt.append(f"{_c.upper()}: PT-PVCGN better on {(_b < _a).sum()} of {len(_a)} stations, worst relative case "
                    f"station {int(np.argmax(_b / _a))} ({(_b / _a).max():.2f}x PVCGN's error)")
    _f.colorbar(_sc, ax=_axes)
    guard.fig(13, "val_station_errors", _f,
              "; ".join(_txt) + ". Points below the diagonal are stations where PT-PVCGN is more accurate.")
    return


@app.cell
def _(DEVICE, ev, guard, json, load_city, masked_metrics, np):
    # Step 14 - FINAL TEST: one evaluation per city through guard.final_test, all pre-registered models together.
    test_scores = {}
    test_store = {}
    for _c in ("hz", "sh"):
        _ct = load_city(_c, "cpu")
        _idx = _ct.windows("test")
        _frame = ev.frozen_test_frame(_ct)

        def _predict_fn(features, ct=_ct, idx=_idx, city=_c):
            # rows of the frozen frame are ordered (window, horizon, station, direction) - check, then predict
            origins = np.array([np.datetime64(ct.dates[d], "ns") + np.timedelta64(330 + 15 * (w + 3), "m")
                                for d, w in idx]).repeat(4 * ct.num_nodes * 2)
            assert np.array_equal(features["origin_time"].to_numpy(), origins), "test frame / window order mismatch"
            preds = {**ev.deep_predictions(ct, idx, DEVICE), **ev.baseline_predictions(ct, idx)}
            test_store[city] = preds
            np.savez_compressed(f"ml_pipeline/runs/test_predictions_{city}.npz",
                                **{f"{m}__seed{k}": p for m, ps in preds.items() for k, p in enumerate(ps)})
            return np.stack([p.reshape(-1) for p in preds[ev.MAIN_MODEL]], 1)     # [rows, seeds]

        def masked_mae_mean_over_seeds(y_true, y_pred):
            return float(np.mean([masked_metrics(y_pred[:, k], y_true)["mae"] for k in range(y_pred.shape[1])]))

        _state = json.load(open(f"ml_pipeline/{_c}/.guard_state.json"))
        _saved = f"ml_pipeline/runs/test_predictions_{_c}.npz"
        if _state["test_touches"] >= 1:
            # This city's test set was already evaluated once; reuse the predictions saved during that single
            # evaluation (no new test touch, no model changes since).
            _z = np.load(_saved)
            test_store[_c] = {}
            for _k in _z.files:
                _m, _s = _k.rsplit("__seed", 1)
                test_store[_c].setdefault(_m, []).append((int(_s), _z[_k]))
            test_store[_c] = {m: [p for _, p in sorted(v)] for m, v in test_store[_c].items()}
            _primary = None
        else:
            _primary = guard.final_test(_predict_fn, _frame, target="y", metric_fn=masked_mae_mean_over_seeds,
                                        state_dir=f"ml_pipeline/{_c}")
        _true = _ct.targets(_idx)
        assert np.array_equal(_true.reshape(-1), _frame["y"].to_numpy())
        # consistency: the guard's number = masked MAE over all test rows, averaged over seeds
        _check = masked_mae_mean_over_seeds(_true.reshape(-1), np.stack([p.reshape(-1) for p in test_store[_c][ev.MAIN_MODEL]], 1))
        if _primary is not None:
            assert abs(_check - _primary) < 1e-6
        test_scores[_c] = ev.score(test_store[_c], _true, ev.slices(_ct, _idx))
        test_scores[_c]["_guard_primary_masked_mae"] = _check
        test_scores[_c]["PVCGN (paper-reported)"] = {"all": {"per_horizon": ev.PAPER_PVCGN[_c],
                                                             "mean": {k: float(np.mean(v)) for k, v in ev.PAPER_PVCGN[_c].items()}}}
    with open("ml_pipeline/runs/test_results.json", "w") as _fh:
        json.dump(test_scores, _fh, indent=1)
    {c: {m: round(s["all"]["mean"]["mae"], 2) for m, s in d.items() if not m.startswith("_")} for c, d in test_scores.items()}
    return test_scores, test_store


@app.cell
def _(ev, guard, np, plt, test_scores):
    # Step 14 figure - test MAE by horizon, both cities
    _f, _axes = plt.subplots(1, 2, figsize=(14, 4.8))
    _h = np.arange(4)
    _txt = []
    for _ax, _c in zip(_axes, ("hz", "sh")):
        _s = test_scores[_c]
        _ax.plot(_h, _s["PVCGN (paper-reported)"]["all"]["per_horizon"]["mae"], ":", marker="^", color="black", label="PVCGN, paper-reported")
        _ax.plot(_h, _s["Ridge (lags+refs)"]["all"]["per_horizon"]["mae"], "--", marker="o", color="C3", lw=1, label="Ridge (lags+refs)")
        for _m, _col in (("PVCGN", "C7"), ("+P", "C0"), (ev.MAIN_MODEL, "C2")):
            if _m in _s:
                _r = _s[_m]["all"]
                _ax.errorbar(_h, _r["per_horizon"]["mae"], yerr=_r["per_horizon_std"]["mae"], marker="s", lw=2.2, capsize=3,
                             color=_col, label=f"{_m} (ours, {_r['n_seeds']} seeds)" if _m != "PVCGN" else f"PVCGN (re-trained, {_r['n_seeds']} seeds)")
        _ax.set_xticks(_h)
        _ax.set_xticklabels(["15 min", "30 min", "45 min", "60 min"])
        _ax.set_ylabel("test MAE")
        _ax.set_title({"hz": "Hangzhou test", "sh": "Shanghai test"}[_c])
        _ax.legend(fontsize=7)
        _pv, _pt = _s["PVCGN"]["all"]["mean"]["mae"], _s[ev.MAIN_MODEL]["all"]["mean"]["mae"]
        _txt.append(f"{_c.upper()} test MAE: PT-PVCGN {_pt:.2f} vs re-trained PVCGN {_pv:.2f} ({100 * (_pv - _pt) / _pv:.1f}% lower) "
                    f"and paper-reported PVCGN {_s['PVCGN (paper-reported)']['all']['mean']['mae']:.2f}")
    guard.fig(14, "test_by_horizon", _f, "; ".join(_txt) + ". Single test evaluation, all models scored together.")
    return


if __name__ == "__main__":
    app.run()
