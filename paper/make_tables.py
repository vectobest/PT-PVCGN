"""Generate every results table in the paper from ml_pipeline/runs/*.json (no hand-typed numbers).

    .venv/bin/python paper/make_tables.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "ml_pipeline" / "runs"
OUT = ROOT / "paper" / "tables"
TEST = json.loads((RUNS / "test_results.json").read_text())
VAL = json.loads((RUNS / "val_results.json").read_text())
MAIN = "PT-PVCGN (+P+T)"

ROWS = [  # (key in results, label in table, group)
    ("Persistence", "Persistence", "base"),
    ("Historical average", "Historical average", "base"),
    ("Seasonal naive (same-type day)", "Seasonal naive (same-type day)", "base"),
    ("Seasonal naive (weekly)", "Seasonal naive (weekly)", "base"),
    ("GBDT (lags+refs)", "GBDT (lags + references)", "base"),
    ("Ridge (lags+refs)", "Ridge (lags + references)", "base"),
    ("PVCGN (paper-reported)", "PVCGN, reported in~\\cite{liu2022pvcgn}", "deep"),
    ("PVCGN", "PVCGN, re-trained", "deep"),
    ("+P", "+P (ours, ablation)", "ours"),
    (MAIN, "PT-PVCGN (ours)", "ours"),
]


def fmt(v, metric, std=None):
    s = f"{100 * v:.2f}\\%" if metric == "mape" else f"{v:.2f}"
    if std is not None and std > 0:
        s += f"\\tiny{{$\\pm${100 * std:.2f}}}" if metric == "mape" else f"\\tiny{{$\\pm${std:.2f}}}"
    return s


def main_table(city, caption, label):
    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{l" + "ccc" * 4 + "}", "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{m} min}}" for m in (15, 30, 45, 60)) + " \\\\",
        " ".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(4)),
        "Method & " + " & ".join(["MAE & RMSE & MAPE"] * 4) + " \\\\", "\\midrule",
    ]
    prev = None
    for key, lab, grp in ROWS:
        if key not in TEST[city]:
            continue
        if prev and grp != prev:
            lines.append("\\midrule")
        prev = grp
        r = TEST[city][key]["all"]
        cells = []
        for h in range(4):
            for m in ("mae", "rmse", "mape"):
                v = r["per_horizon"][m][h]
                sd = r.get("per_horizon_std", {}).get(m, [0] * 4)[h] if key != "PVCGN (paper-reported)" else None
                txt = fmt(v, m, sd if (grp != "base") else None)
                if key == MAIN:
                    txt = "\\textbf{" + txt + "}"
                cells.append(txt)
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def ablation_table():
    rows = [("PVCGN", "PVCGN"), ("+P", "+P"), (MAIN, "PT-PVCGN (+P+T)")]
    lines = [
        "\\begin{table}[t]", "\\centering",
        "\\caption{HZMetro ablation: MAE (mean $\\pm$ std over 3 seeds) under the 200-epoch schedule used for all main results and under a shorter 100-epoch schedule. The final model was fixed on validation before the test evaluation.}",
        "\\label{tab:ablation}", "\\footnotesize", "\\begin{tabular}{lcccc}", "\\toprule",
        " & \\multicolumn{2}{c}{200 epochs} & \\multicolumn{2}{c}{100 epochs} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Model & Val & Test & Val & Test \\\\", "\\midrule",
    ]
    for key, lab in rows:
        cells = []
        for k in (key, f"{key} [100 ep]"):
            for src in (VAL, TEST):
                a = src["hz"][k]["all"]
                cells.append(f"{a['mean']['mae']:.2f}\\tiny{{$\\pm${a['std']['mae']:.2f}}}")
        lines.append(f"{lab} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def slices_table():
    cols = [("hz", "rush hours", "HZ rush"), ("hz", "top 1/4 stations", "HZ top-1/4"),
            ("sh", "rush hours", "SH rush"), ("sh", "top 1/4 stations", "SH top-1/4"),
            ("sh", "working days", "SH work"), ("sh", "non-working days", "SH non-work"),
            ("sh", "holiday (Sep 15-17)", "SH holiday"), ("sh", "make-up workday (Sep 18)", "SH make-up")]
    rows = [("Ridge (lags+refs)", "Ridge"), ("PVCGN", "PVCGN (re-trained)"), (MAIN, "PT-PVCGN")]
    lines = [
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Test MAE (mean over 4 horizons and over seeds) on operational slices. Rush hours: 07:30--09:30 and 17:30--19:30. Top-1/4: busiest quarter of stations by training ridership. SH holiday: Mid-Autumn Festival (Sep 15--17, 2016); SH make-up: Sunday Sep 18, 2016, an official working day.}",
        "\\label{tab:slices}", "\\footnotesize", "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
        "Model & " + " & ".join(c[2] for c in cols) + " \\\\", "\\midrule",
    ]
    for key, lab in rows:
        vals = [TEST[c][key][s]["mean"]["mae"] for c, s, _ in cols]
        best = [min(TEST[c][k][s]["mean"]["mae"] for k, _ in rows) for c, s, _ in cols]
        lines.append(f"{lab} & " + " & ".join(("\\textbf{%.2f}" if abs(v - b) < 1e-9 else "%.2f") % v for v, b in zip(vals, best)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def macros():
    """Numbers quoted in the text, as LaTeX macros."""
    t, v = TEST, VAL
    m = lambda src, c, k: src[c][k]["all"]["mean"]["mae"]
    out = {
        "hzPT": m(t, "hz", MAIN), "hzPTstd": t["hz"][MAIN]["all"]["std"]["mae"],
        "hzP": m(t, "hz", "+P"), "hzPV": m(t, "hz", "PVCGN"), "hzPVstd": t["hz"]["PVCGN"]["all"]["std"]["mae"],
        "hzPVpaper": m(t, "hz", "PVCGN (paper-reported)"), "hzRidge": m(t, "hz", "Ridge (lags+refs)"),
        "shPT": m(t, "sh", MAIN), "shPTstd": t["sh"][MAIN]["all"]["std"]["mae"],
        "shPV": m(t, "sh", "PVCGN"), "shPVstd": t["sh"]["PVCGN"]["all"]["std"]["mae"],
        "shPVpaper": m(t, "sh", "PVCGN (paper-reported)"), "shRidge": m(t, "sh", "Ridge (lags+refs)"),
        "hzPVvalHund": m(v, "hz", "PVCGN [100 ep]"), "hzPVval": m(v, "hz", "PVCGN"),
    }
    out["hzGain"] = 100 * (out["hzPV"] - out["hzPT"]) / out["hzPV"]
    out["shGain"] = 100 * (out["shPV"] - out["shPT"]) / out["shPV"]
    out["shGainPaper"] = 100 * (out["shPVpaper"] - out["shPT"]) / out["shPVpaper"]
    h = lambda c, k, i: t[c][k]["all"]["per_horizon"]["mae"][i]
    out["hzGainFifteen"] = 100 * (h("hz", "PVCGN", 0) - h("hz", MAIN, 0)) / h("hz", "PVCGN", 0)
    out["hzGainSixty"] = 100 * (h("hz", "PVCGN", 3) - h("hz", MAIN, 3)) / h("hz", "PVCGN", 3)
    s = lambda c, k, sl: t[c][k][sl]["mean"]["mae"]
    out["shHolPT"], out["shHolPV"] = s("sh", MAIN, "holiday (Sep 15-17)"), s("sh", "PVCGN", "holiday (Sep 15-17)")
    out["shMakePT"], out["shMakePV"] = s("sh", MAIN, "make-up workday (Sep 18)"), s("sh", "PVCGN", "make-up workday (Sep 18)")
    out["shWorkPT"], out["shWorkPV"] = s("sh", MAIN, "working days"), s("sh", "PVCGN", "working days")
    rush = lambda k, i: t["hz"][k]["rush hours"]["per_horizon"]["mae"][i]
    out["hzRushFifteenGap"] = rush("PVCGN", 0) - rush(MAIN, 0)
    out["hzRushSixtyGap"] = rush("PVCGN", 3) - rush(MAIN, 3)
    names = {k: "\\newcommand{\\%s}{%s}" % (k, f"{val:.1f}" if "Gain" in k else f"{val:.2f}") for k, val in out.items()}
    return "\n".join(names.values())


# Test results reported in Liu et al. (T-ITS 2022), Tables II-VII, per horizon: (RMSE, MAE, MAPE %).
# Same splits, same rush-hour and top-1/4-station definitions; quoted for comparison, not re-run by us.
REPORTED = {
    ("sh", "all"): {
        "DCRNN": [(46.02, 24.04, 17.82), (49.90, 25.23, 18.35), (54.92, 26.76, 19.30), (58.83, 28.01, 20.44)],
        "GCRNN": [(46.09, 24.26, 18.06), (50.12, 25.42, 18.73), (54.87, 26.92, 19.81), (58.67, 28.18, 21.07)],
        "Graph-WaveNet": [(46.98, 24.91, 20.05), (51.64, 26.53, 20.38), (58.50, 28.78, 21.99), (65.08, 30.90, 24.36)],
        "PVCGN": [(44.97, 23.29, 16.83), (47.83, 24.16, 17.23), (52.02, 25.33, 17.92), (55.27, 26.29, 18.69)],
    },
    ("hz", "all"): {
        "DCRNN": [(40.39, 23.76, 14.00), (42.57, 25.22, 14.99), (46.26, 26.97, 16.19), (49.35, 28.47, 18.16)],
        "GCRNN": [(40.24, 23.84, 14.08), (41.95, 25.14, 14.86), (45.53, 26.82, 16.05), (50.28, 28.75, 17.89)],
        "Graph-WaveNet": [(40.78, 24.07, 14.27), (42.80, 25.48, 15.23), (45.84, 27.15, 17.34), (49.89, 29.14, 19.37)],
        "PVCGN": [(37.76, 22.68, 13.70), (39.34, 23.33, 13.81), (40.95, 24.22, 14.45), (42.61, 24.93, 15.49)],
    },
    ("sh", "rush hours"): {
        "DCRNN": [(67.50, 37.92, 13.93), (73.07, 40.16, 14.33), (79.42, 41.92, 15.29), (79.98, 41.23, 16.58)],
        "GCRNN": [(66.21, 37.94, 14.07), (73.63, 40.26, 14.44), (79.88, 41.65, 15.29), (81.37, 41.27, 16.62)],
        "Graph-WaveNet": [(68.41, 39.17, 14.14), (78.98, 43.54, 14.79), (87.66, 46.26, 15.95), (90.19, 46.35, 17.66)],
        "PVCGN": [(65.04, 36.46, 13.16), (68.85, 37.77, 13.41), (73.85, 38.84, 14.06), (74.41, 38.12, 15.08)],
    },
    ("hz", "rush hours"): {
        "DCRNN": [(54.17, 35.08, 10.37), (58.27, 37.48, 10.69), (61.83, 37.95, 11.16), (59.52, 36.27, 11.94)],
        "GCRNN": [(55.51, 35.68, 10.36), (57.34, 37.31, 10.54), (59.54, 37.58, 11.16), (58.88, 35.94, 11.93)],
        "Graph-WaveNet": [(56.98, 37.19, 10.84), (59.71, 38.94, 11.04), (59.83, 38.75, 11.83), (59.96, 37.49, 12.35)],
        "PVCGN": [(49.79, 32.63, 9.72), (51.63, 33.30, 9.52), (51.45, 32.73, 9.88), (51.09, 31.43, 10.43)],
    },
    ("sh", "top 1/4 stations"): {
        "DCRNN": [(84.04, 44.98, 13.76), (88.52, 46.80, 14.47), (97.75, 49.84, 16.32), (106.03, 53.30, 17.89)],
        "GCRNN": [(86.09, 45.89, 14.12), (89.89, 47.50, 14.82), (96.81, 50.08, 16.70), (102.93, 52.79, 18.16)],
        "Graph-WaveNet": [(76.93, 43.31, 11.68), (84.23, 46.32, 13.12), (96.35, 50.76, 15.50), (109.26, 55.74, 18.27)],
        "PVCGN": [(74.80, 41.38, 10.62), (79.43, 43.05, 11.46), (87.32, 45.67, 12.48), (93.59, 48.02, 13.61)],
    },
    ("hz", "top 1/4 stations"): {
        "DCRNN": [(66.18, 41.22, 10.72), (69.36, 43.92, 11.51), (75.16, 46.99, 12.74), (79.79, 49.50, 15.00)],
        "GCRNN": [(65.29, 40.93, 10.59), (67.29, 43.07, 11.42), (72.42, 45.93, 12.46), (80.34, 49.40, 14.74)],
        "Graph-WaveNet": [(65.87, 40.72, 10.39), (68.92, 43.21, 11.40), (72.96, 45.71, 13.32), (79.70, 49.77, 16.56)],
        "PVCGN": [(60.56, 38.29, 9.97), (63.77, 39.93, 10.34), (65.99, 41.75, 11.28), (69.25, 43.16, 12.54)],
    },
}
# (column header, source, key): source "ours" = our single test evaluation, "rep" = REPORTED
COMPARISON_COLUMNS = [
    ("HA", "ours", "Historical average"),
    ("SN", "ours", "Seasonal naive (same-type day)"),
    ("Ridge", "ours", "Ridge (lags+refs)"),
    ("GBDT", "ours", "GBDT (lags+refs)"),
    ("DCRNN$^\\dagger$", "rep", "DCRNN"),
    ("GCRNN$^\\dagger$", "rep", "GCRNN"),
    ("Graph-WaveNet$^\\dagger$", "rep", "Graph-WaveNet"),
    ("PVCGN$^\\dagger$", "rep", "PVCGN"),
    ("PVCGN", "ours", "PVCGN"),
    ("+P", "ours", "+P"),
    ("PT-PVCGN", "ours", MAIN),
]


def comparison_table(city, slice_, caption, label):
    """Paper-style table: rows = horizon x (RMSE, MAE, MAPE), columns = methods; best value per row in bold."""
    cols = [c for c in COMPARISON_COLUMNS if c[1] == "rep" or c[2] in TEST[city]]
    rep = REPORTED[(city, slice_)]

    def value(col, h, m):
        _, src, key = col
        if src == "rep":
            return rep[key][h][("rmse", "mae", "mape").index(m)] / (100 if m == "mape" else 1)
        return TEST[city][key][slice_]["per_horizon"][m][h]

    lines = [
        "\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{cc|" + "c" * 4 + "|" + "c" * 4 + "|" + "c" * (len(cols) - 8) + "}", "\\toprule",
        "Time & Metric & " + " & ".join(c[0] for c in cols) + " \\\\", "\\midrule",
    ]
    for h, minutes in enumerate((15, 30, 45, 60)):
        for i, m in enumerate(("rmse", "mae", "mape")):
            vals = [value(c, h, m) for c in cols]
            best = min(vals)
            cells = []
            for v in vals:
                txt = f"{100 * v:.2f}\\%" if m == "mape" else f"{v:.2f}"
                cells.append("\\textbf{" + txt + "}" if abs(v - best) < 5e-5 else txt)
            lead = f"\\multirow{{3}}{{*}}{{{minutes} min}}" if i == 0 else ""
            lines.append(f"{lead} & {m.upper()} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule" if h < 3 else "\\bottomrule")
    lines += ["\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def comparison_data(city, slice_):
    """Same content as comparison_table, as plain data for the slides."""
    cols = [c for c in COMPARISON_COLUMNS if c[1] == "rep" or c[2] in TEST[city]]
    rep = REPORTED[(city, slice_)]
    rows = []
    for h, minutes in enumerate((15, 30, 45, 60)):
        for m in ("rmse", "mae", "mape"):
            vals = [rep[c[2]][h][("rmse", "mae", "mape").index(m)] / (100 if m == "mape" else 1) if c[1] == "rep"
                    else TEST[city][c[2]][slice_]["per_horizon"][m][h] for c in cols]
            best = min(vals)
            rows.append({"time": f"{minutes} min", "metric": m.upper(),
                         "cells": [f"{100 * v:.2f}%" if m == "mape" else f"{v:.2f}" for v in vals],
                         "best": [abs(v - best) < 5e-5 for v in vals]})
    return {"columns": [c[0].replace("$^\\dagger$", "†") for c in cols], "ours": [c[1] == "ours" and c[2] in ("PVCGN", "+P", MAIN) for c in cols], "rows": rows}


COMPARISON_NOTE = ("$^\\dagger$Reported in~\\cite{liu2022pvcgn} on the same test split (not re-run). "
                   "HA: historical average; SN: seasonal naive (same-type reference day). "
                   "Our deep models: mean over seeds.")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    budget = {"hz": "All our deep models use the same 200-epoch schedule (3 seeds).",
              "sh": "Our PVCGN and PT-PVCGN are trained for 30 epochs (2 seeds); +P was not trained on SHMetro; $^\\dagger$models were trained for 200 epochs."}
    name = {"hz": "HZMetro", "sh": "SHMetro"}
    scope = {"all": ("Quantitative comparison on the whole {} test set", "whole"),
             "rush hours": ("Quantitative comparison during rush hours (07:30--09:30 and 17:30--19:30) on the {} test set", "rush"),
             "top 1/4 stations": ("Quantitative comparison on the top 1/4 high-ridership stations of the {} test set", "top")}
    for sl, (title, short) in scope.items():
        for c in ("sh", "hz"):
            (OUT / f"comp_{c}_{short}.tex").write_text(comparison_table(
                c, sl, title.format(name[c]) + ". " + budget[c] + " " + COMPARISON_NOTE, f"tab:{c}-{short}"))
    (OUT / "comparison.json").write_text(json.dumps(
        {f"{c}_{short}": comparison_data(c, sl) for sl, (_, short) in scope.items() for c in ("sh", "hz")}, indent=1))
    (OUT / "ablation.tex").write_text(ablation_table())
    (OUT / "slices.tex").write_text(slices_table())
    (OUT / "numbers.tex").write_text(macros() + "\n")
    print("wrote", sorted(p.name for p in OUT.iterdir()))
