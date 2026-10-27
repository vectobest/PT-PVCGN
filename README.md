# PT-PVCGN: periodicity- and calendar-aware metro ridership forecasting

This project extends **PVCGN** (Liu et al., *Physical-Virtual Collaboration Modeling for Intra- and Inter-Station Metro Ridership Prediction*, IEEE T-ITS 23(4), 2022; [code](https://github.com/HCPLab-SYSU/PVCGN)).

**The research gap.** PVCGN forecasts station inflow and outflow for the next hour from the last hour of observations only. It has no notion of time of day, working day or holiday, and never sees earlier days. In the released benchmarks, ridership at the same time on a comparable earlier day correlates at r ≈ 0.96–0.99 with ridership one hour ahead. The most recent observation reaches only r ≈ 0.66–0.70.

**What we add.** PT-PVCGN is PVCGN plus two switchable parts:
- **P (periodic inputs):** the same time slots on the most recent day of the same type (working or non-working), and on the same weekday one week earlier.
- **T (calendar):** time-of-day, day-of-week and working-day embeddings.

With both switched off, the code is exactly PVCGN.

## Results (test set, evaluated once)

MAE averaged over the 15/30/45/60-minute horizons; lower is better. Our models are the mean ± std over seeds.

| Model | HZMetro | SHMetro |
|---|---|---|
| PVCGN, reported in the original paper (200 epochs) | 23.79 | 24.77 |
| PVCGN, our re-implementation | 23.85 ± 0.10 (200 epochs) | 27.44 ± 0.23 (30 epochs) |
| Ridge regression on lags + reference days | 25.96 | 25.62 |
| +P (ablation) | **21.72 ± 0.09** | – |
| **PT-PVCGN** | 22.06 ± 0.09 | **24.09 ± 0.03** |

- **HZMetro, equal training:** PT-PVCGN has 7.5% lower MAE than PVCGN. The gap grows with the horizon, from 4.4% at 15 minutes to 9.6% at 60 minutes.
- **SHMetro:** after 30 epochs, PT-PVCGN has the lowest MAE of all methods, but not the lowest MAPE.
- **The periodic inputs carry the gain.** The calendar embeddings do not help.
- **Weak on irregular days:** the model is worse than PVCGN on public holidays and make-up working days.

Full per-horizon tables in the original paper's format are in [`paper/tables/`](paper/tables), and the draft paper is [`paper/main.tex`](paper/main.tex).

## Repository layout

| Path | Purpose |
|---|---|
| `metro/data.py` | Loads the released pickles and rebuilds continuous daily series (verified against every released window) |
| `metro/features.py` | Official calendar (holidays, make-up working days) and reference-day selection, using past days only |
| `metro/batches.py` | Builds model batches: last hour, targets, reference-day slots, calendar indices |
| `metro/model.py` | PVCGN in plain PyTorch, plus the `periodic` / `temporal` switches |
| `metro/train.py` | Trains one model on the training days and selects the epoch on validation |
| `metro/baselines.py` | Persistence, seasonal naive, historical average, ridge, gradient boosting |
| `metro/metrics.py` | MAE / RMSE / MAPE exactly as in the original code (zero targets masked) |
| `metro/evaluate.py` | Shared error-analysis and final-test code, with operational slices |
| `ml_pipeline/01_eda.py` … `04_eval.py` | marimo notebooks: analysis → preparation → models → evaluation |
| `ml_pipeline/guard.py` | Safeguards: chronological split, frozen test set, one-time final test |
| `ml_pipeline/PIPELINE.md` | Logbook of every step, decision and review gate |
| `ml_pipeline/figures/`, `ml_pipeline/runs/` | All figures, per-run training logs (JSON), validation/test results, test predictions |
| `tests/test_equivalence.py` | Checks our PVCGN equals the authors' implementation (forward pass, 1e-5) |
| `paper/` | LaTeX draft; `make_tables.py` generates every table and quoted number from `ml_pipeline/runs/` |
| `presentation/` | Slide deck (PPTX + PDF) and the script that builds it |
| `run_queue.sh`, `sh_queue.zsh`, `hz200_queue.zsh` | The training queues that produced the reported runs |

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1. Get the authors' code and data (not redistributed here; see "Data" below)
git clone https://github.com/HCPLab-SYSU/PVCGN.git reference_PVCGN
(cd reference_PVCGN/data && tar xzf data.tar.gz)

# 2. Prepare features (calendar, reference days, scaler, frozen split)
.venv/bin/python ml_pipeline/02_prep.py

# 3. Train, e.g. PT-PVCGN on HZMetro with the 200-epoch schedule
.venv/bin/python -m metro.train --city hz --variant pt --seed 1 --epochs 200 --batch 32 --milestones 100 150 --patience 50 --tag _e200

# 4. Evaluate and regenerate the paper tables
.venv/bin/python ml_pipeline/04_eval.py
.venv/bin/python paper/make_tables.py

# Optional: check the re-implementation against the authors' code
.venv/bin/python tests/test_equivalence.py
```

Model checkpoints are not included: about 150 MB per Hangzhou model and 480 MB per Shanghai model. Variants: `pvcgn`, `p` (+P), `pt` (+P+T).

## Data

The HZMetro and SHMetro benchmarks belong to the PVCGN authors and come from their repository. That repository carries no license, so this repository does not redistribute the data or the feature files derived from it (`ml_pipeline/*/features.npz`). Step 2 above rebuilds them. If you use the data, cite the original paper:

```bibtex
@article{liu2022pvcgn,
  title   = {Physical-Virtual Collaboration Modeling for Intra- and Inter-Station Metro Ridership Prediction},
  author  = {Liu, Lingbo and Chen, Jingwen and Wu, Hefeng and Zhen, Jiajie and Li, Guanbin and Lin, Liang},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {23}, number = {4}, pages = {3377--3391}, year = {2022}
}
```

## Status

This is research code for a minor project. The paper is a draft.
