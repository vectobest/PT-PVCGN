import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    import os
    os.chdir(ROOT)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from ml_pipeline import guard
    from metro.data import load_graphs, long_table
    return ROOT, guard, load_graphs, long_table, np, pd, plt


@app.cell
def _(long_table, pd):
    # Step 1 - data inspection: both benchmarks as one long table (read-only)
    hz = long_table("hz")
    sh = long_table("sh")
    df = pd.concat([hz, sh], ignore_index=True)
    df.head()
    return df, hz, sh


@app.cell
def _(df, guard):
    profile = guard.profile(df, target="inflow", time_col="timestamp")
    {k: profile[k] for k in ("shape", "duplicates", "leakage_suspects", "traits")}
    return (profile,)


@app.cell
def _(df, guard):
    # Step 2 - the required generic figures
    guard.eda_figures(df, target="inflow", time_col="timestamp")
    return


@app.cell
def _(guard, hz, np, plt, sh):
    # Daily profile: weekday vs weekend network inflow per 15-minute slot
    _f, _axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    _peak_ratio = {}
    for _ax, (_name, _d) in zip(_axes, [("Hangzhou", hz), ("Shanghai", sh)]):
        _tot = _d.groupby(["timestamp", "slot", "day_of_week"], as_index=False)["inflow"].sum()
        _tot["weekend"] = _tot["day_of_week"] >= 5
        _prof = _tot.groupby(["weekend", "slot"])["inflow"].mean().unstack(0)
        _hours = 5.5 + _prof.index.to_numpy() * 0.25
        _ax.plot(_hours, _prof[False], label="weekday")
        _ax.plot(_hours, _prof[True], label="weekend")
        _ax.set_title(f"{_name}: mean network inflow per 15 min")
        _ax.set_xlabel("hour of day")
        _ax.legend()
        _peak_ratio[_name] = _prof[False].max() / _prof[True].max()
    _explanation = (
        "Weekday ridership has sharp morning and evening peaks that weekends lack "
        f"(weekday/weekend peak ratio HZ {_peak_ratio['Hangzhou']:.2f}, SH {_peak_ratio['Shanghai']:.2f}); "
        "the reference model never sees the time of day or day type, so it must infer this regime from one hour of history."
    )
    guard.fig(2, "daily_profile", _f, _explanation)
    return


@app.cell
def _(guard, hz, np, plt, sh):
    # Periodicity: how well does each lag predict the current value (station-level correlation)?
    from metro.data import SLOTS_PER_DAY

    def _lag_corr(_d, n):
        _arr = _d.sort_values(["timestamp", "station"])["inflow"].to_numpy().reshape(-1, SLOTS_PER_DAY, n)
        _cur = _arr[7:, 4:]  # skip the first week and first hour so every lag exists
        _lags = {
            "15 min": _arr[7:, 3:-1],
            "1 hour": _arr[7:, :-4],
            "same slot, 1 day": _arr[6:-1, 4:],
            "same slot, 7 days": _arr[:-7, 4:],
        }
        return {_k: float(np.corrcoef(_cur.ravel(), v.ravel())[0, 1]) for _k, v in _lags.items()}

    _lc = {"Hangzhou": _lag_corr(hz, 80), "Shanghai": _lag_corr(sh, 288)}
    _f, _ax = plt.subplots(figsize=(8, 4))
    _keys = list(_lc["Hangzhou"])
    _xs = np.arange(len(_keys))
    _ax.bar(_xs - 0.2, [_lc["Hangzhou"][_k] for _k in _keys], 0.4, label="Hangzhou")
    _ax.bar(_xs + 0.2, [_lc["Shanghai"][_k] for _k in _keys], 0.4, label="Shanghai")
    _ax.set_xticks(_xs)
    _ax.set_xticklabels(_keys)
    _ax.set_ylim(0.5, 1)
    _ax.set_ylabel("Pearson r with current inflow")
    _ax.set_title("How informative is each look-back?")
    _ax.legend()
    _explanation = (
        f"Same-slot-last-week inflow correlates r={_lc['Hangzhou']['same slot, 7 days']:.3f} (HZ) / "
        f"{_lc['Shanghai']['same slot, 7 days']:.3f} (SH) with the current value, versus "
        f"{_lc['Hangzhou']['1 hour']:.3f} / {_lc['Shanghai']['1 hour']:.3f} for the value one hour earlier - "
        "the 60-minute horizon is exactly where periodic history should help, and PVCGN does not use it."
    )
    guard.fig(2, "periodicity_lags", _f, _explanation)
    return


@app.cell
def _(guard, hz, np, plt):
    # Are inter-station similarities static? Compare correlation structure in AM peak vs PM peak.
    _tr = hz[hz["split_released"] == "train"]

    def _corr_in(_hours):
        _sel = _tr[_tr["timestamp"].dt.hour.isin(_hours)]
        _m = _sel.pivot_table(index="timestamp", columns="station", values="inflow").to_numpy()
        return np.corrcoef(_m.T)

    _am, _pm = _corr_in([7, 8]), _corr_in([17, 18])
    np.fill_diagonal(_am, np.nan)
    np.fill_diagonal(_pm, np.nan)
    _top = 10
    _am_nb = np.argsort(-np.nan_to_num(_am, nan=-9), axis=1)[:, :_top]
    _pm_nb = np.argsort(-np.nan_to_num(_pm, nan=-9), axis=1)[:, :_top]
    _overlap = np.array([len(set(_a) & set(_b)) / _top for _a, _b in zip(_am_nb, _pm_nb)])

    _f, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _iu = np.triu_indices_from(_am, 1)
    _axes[0].scatter(_am[_iu], _pm[_iu], s=2, alpha=0.3)
    _axes[0].plot([-1, 1], [-1, 1], color="grey", lw=1)
    _axes[0].set_xlabel("pair correlation, 07-09h")
    _axes[0].set_ylabel("pair correlation, 17-19h")
    _axes[0].set_title("Hangzhou station-pair inflow correlation (train)")
    _axes[1].hist(_overlap, bins=np.linspace(0, 1, 11))
    _axes[1].set_xlabel("share of top-10 neighbours kept from AM to PM peak")
    _axes[1].set_ylabel("stations")
    _axes[1].set_title("Neighbour stability")
    _explanation = (
        f"Each station keeps on average only {_overlap.mean():.0%} of its 10 most-correlated stations between the "
        "morning and evening peak; PVCGN's similarity/correlation graphs are fixed over the whole day, "
        "so a time-dependent graph is a plausible improvement."
    )
    guard.fig(2, "dynamic_similarity", _f, _explanation)
    return


@app.cell
def _(guard, load_graphs, np, plt):
    # The three released graphs: how much do they _overlap, and how dense are they?
    def _jacc(_a, _b):
        _a, _b = _a > 0, _b > 0
        return (_a & _b).sum() / (_a | _b).sum()

    _f, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _rows = []
    for _ax, city in zip(_axes, ["hz", "sh"]):
        _g = load_graphs(city)
        for _k in _g:
            np.fill_diagonal(_g[_k], 0)
            _ax.hist((_g[_k] > 0).sum(1), bins=30, alpha=0.6, label={"conn": "physical", "sml": "similarity", "cor": "correlation"}[_k])
        _ax.set_title(f"{city.upper()}: out-degree per station")
        _ax.set_xlabel("neighbours")
        _ax.legend()
        _rows.append((city, _jacc(_g["conn"], _g["sml"]), _jacc(_g["conn"], _g["cor"]), _jacc(_g["sml"], _g["cor"])))
    _txt = "; ".join(f"{_c.upper()} P-S {_a:.2f}, P-C {_b:.2f}, S-C {_s:.2f}" for _c, _a, _b, _s in _rows)
    _explanation = (
        f"Edge-set Jaccard overlaps ({_txt}) show the three graphs encode largely different neighbours; "
        "HZ virtual graphs are threshold-based and much denser than SH's top-10 graphs."
    )
    guard.fig(2, "graph_structure", _f, _explanation)
    return


@app.cell
def _(guard, hz, np, plt, sh):
    # Daily totals over the whole period, with the released _split boundaries
    _f, _axes = plt.subplots(2, 1, figsize=(12, 6))
    for _ax, (_name, _d) in zip(_axes, [("Hangzhou", hz), ("Shanghai", sh)]):
        _daily = _d.groupby([_d["timestamp"].dt.date, "split_released"])["inflow"].sum().reset_index()
        for _split, _c in [("train", "C0"), ("val", "C1"), ("test", "C2")]:
            _s = _daily[_daily["split_released"] == _split]
            _ax.bar(_s["timestamp"].astype(str), _s["inflow"] / 1e6, color=_c, label=_split)
        _ax.set_title(f"{_name}: daily inflow (millions), coloured by released split")
        _ax.tick_params(axis="x", rotation=90, labelsize=6)
        _ax.legend()
    _f.tight_layout()
    _explanation = (
        "Weekly cycles dominate both cities; the splits are strictly chronological, the HZ test week is all weekdays "
        "(Mon-Fri), and the SH test period contains the Mid-Autumn Festival holiday (Sep 15-17) - a known shift to check in error analysis."
    )
    guard.fig(2, "daily_totals_splits", _f, _explanation)
    return


@app.cell
def _(guard, hz, np, plt, sh):
    # Noise grows with volume: spread of the same station-slot across weekdays
    _f, _ax = plt.subplots(figsize=(7, 5))
    for _name, _d in [("Hangzhou", hz), ("Shanghai", sh)]:
        _w = _d[(_d["split_released"] == "train") & (_d["day_of_week"] < 5)]
        _st = _w.groupby(["station", "slot"])["inflow"].agg(["mean", "std"])
        _st = _st[_st["mean"] > 1]
        _ax.loglog(_st["mean"], _st["std"], ".", ms=1, alpha=0.3, label=_name)
        _slope = np.polyfit(np.log(_st["mean"]), np.log(_st["std"] + 1e-6), 1)[0]
        _ax.text(0.02, 0.95 - 0.06 * (_name == "Shanghai"), f"{_name}: log-log slope {_slope:.2f}", transform=_ax.transAxes)
    _ax.set_xlabel("mean weekday inflow at station-slot")
    _ax.set_ylabel("std across weekdays")
    _ax.set_title("Day-to-day variability vs volume (train weekdays)")
    _ax.legend(loc="lower right")
    _explanation = (
        "Day-to-day spread grows with volume (heteroscedastic noise), so a single point forecast hides very different "
        "uncertainty at busy vs quiet stations - PVCGN gives no uncertainty estimate."
    )
    guard.fig(2, "heteroscedasticity", _f, _explanation)
    return


if __name__ == "__main__":
    app.run()
