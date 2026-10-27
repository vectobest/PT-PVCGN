# ML pipeline — extending PVCGN for metro ridership forecasting

Reference: Liu et al., "Physical-Virtual Collaboration Modeling for Intra- and Inter-Station Metro
Ridership Prediction", IEEE T-ITS 23(4), 2022. Code + data: https://github.com/HCPLab-SYSU/PVCGN
(cloned to `reference_PVCGN/`, not committed).

## Status

| # | Step | Status | Result |
|---|------|--------|--------|
| 1 | Data inspection | done | HZ 25 days × 73 slots × 80 stations; SH 92 days × 73 × 288. No NaN, no negatives, 2.4% / 3.6% zeros. Window → day rebuild verified exactly against every released window. No leakage suspects. |
| 2 | EDA | done | 11 figures (below). Strong weekly periodicity, time-varying station similarity, heteroscedastic noise, HZ val = weekend only, SH test has a holiday + a make-up working Sunday. |
| 3 | Problem definition | approved-gate | Contract + research gaps below. |
| 4 | Data cleaning | done | No missing/negative values, 0 closed station-days; 10 (HZ) / 166 (SH) cells >10× train median kept as real surges. No rows removed. Zero targets masked in metrics exactly as reference. |
| 5 | Data engineering | done | Chronological day tensors [D,73,N,2]; official calendar (holidays, make-up workday); reference days = latest earlier same-type day + same weekday 7 days back (fallback). 2 HZ + 2 SH training days without a reference dropped. |
| 6 | Split | done | guard.split (time) on the evaluation table reproduces the released date split exactly (asserted); test fingerprints frozen in ml_pipeline/hz, ml_pipeline/sh. |
| 7 | Feature engineering | done | Periodic reference inputs (same-type day, weekly reference) and calendar indices (slot, day of week, working day). Period-of-day graphs were built then removed 2026-10-23 with the scope change (Gap 1 only); the unused array remains inside features.npz and is ignored. |
| 8 | Preprocessing | done | Global z-score per city fit on train days only (scaler.json). PVCGN ported to plain PyTorch; forward pass equals the authors' code to 1e-5 (tests/test_equivalence.py). |
| 9 | Baseline | done | Val MAE (mean of 4 horizons) HZ / SH: persistence 40.62 / 83.58; seasonal same-type 35.70 / 30.33; seasonal weekly 26.04 / 26.65; historical avg 27.91 / 26.05; ridge 22.68 / 22.38; GBDT 23.75 / 22.61. Ridge on periodic features is the line to beat. |
| 10 | Training | done (HZ), in progress (SH) | HZ: 9 runs (pvcgn/p/pt x 3 seeds, 100 epochs). Best val MAE mean+-std: PVCGN 24.40+-0.36, +P 21.26+-0.05, PT-PVCGN 21.03+-0.04. SH queue running (sh_queue.zsh). |
| 11 | Tuning | done (no extra search) | Epoch selected on validation for every run; no further hyper-parameter search was run (PVCGN keeps the paper config; PT-PVCGN uses the same config so the ablation is controlled). Open question: PVCGN still improved 0.30 MAE after epoch 75, so it may be under-trained at 100 epochs. |
| 12 | Evaluation | done (validation, both cities) | Equal budget. HZ 200-epoch schedule: PVCGN 22.53+-0.15, +P 20.93+-0.04, +P+T 21.20+-0.12 (100-epoch: 24.40 / 21.26 / 21.03). SH 30 epochs: PVCGN 25.40+-0.38, +P+T 20.79+-0.01. Best baseline ridge HZ 22.68 / SH 22.38. Periodic inputs carry the gain; calendar embeddings help only at 100 epochs on HZ. |
| 13 | Error analysis | done | Validation (04_eval.py): PT-PVCGN beats PVCGN on every validation slice and on 68/80 HZ and 287/288 SH stations; worst SH station 1.77x PVCGN's error. Test slices: gains on working days, rush hours, busiest stations; WORSE than PVCGN on the SH holiday and the make-up workday (periodic references from ordinary days mislead it). |
| 14 | Final test | done | One touch per city. Test MAE: HZ PT-PVCGN 22.06+-0.09 vs re-trained PVCGN 23.85+-0.10 (paper-reported 23.79), +P 21.72, ridge 25.96. SH PT-PVCGN 24.09+-0.03 vs re-trained PVCGN (30 ep) 27.44+-0.23 (paper-reported, 200 ep, 24.77), ridge 25.62. |
| 15 | Deployment | skipped (not requested) | Research project; checkpoints + code reproduce every number. |
| 16 | Monitoring plan | done | See 'Monitoring and retraining plan' below. |

## Gates

- Gate A: approved 2026-10-22 (direction: periodic + time-aware graphs; datasets: HZ first, then SH)
- Gate B: approved 2026-10-22 (compute: HZ 100 epochs x 3 seeds all ablations; SH PVCGN vs full, 30 epochs x 2 seeds)
- Gate C: approved 2026-10-24 (main model: PT-PVCGN (+P+T), chosen on validation before the test; single test evaluation of all pre-registered models on both cities)
- Gate D: approved 2026-10-24 (pipeline closed; next: write results into the paper and update the slides)
- Scope change 2026-10-23 (user request): focus on ONE gap - Gap 1, periodic + calendar context. Experiments: pvcgn -> p -> pt (PT-PVCGN = final model); the time-aware graph component (G, Gap 2) is dropped from experiments and left as future work. SH: PVCGN vs PT-PVCGN. Runs already finished are kept unchanged.
- Scope change 2026-10-23 (user request): model training and evaluation on Hangzhou (HZMetro) only; no Shanghai (SHMetro) training runs. SHMetro remains in the step 1-2 data analysis and its prepared/frozen split is left unused.
- Scope change 2026-10-23 (user request, reverses the previous line): after the Hangzhou runs, also train on Shanghai - PVCGN vs PT-PVCGN, 2 seeds, 30 epochs (milestones 20/25, patience 10, batch 8), via sh_queue.zsh.
- Gate C (Hangzhou): was pending - user chose 2026-10-23 to first run a fairness check (PVCGN with the paper's full 200-epoch schedule, 3 seeds, hz200_queue.zsh, after Shanghai); the single Hangzhou test follows, using whichever PVCGN setting is better on validation.
- Fairness check result 2026-10-24: PVCGN at 200 epochs reaches val MAE 22.53+-0.15 (vs 24.40+-0.36 at 100). User chose to give every Hangzhou model the same 200-epoch budget: PT-PVCGN and +P retrained at 200 epochs (3 seeds each) before the single Hangzhou test; the 200-epoch runs are the main Hangzhou results.

## Step 1–2 notes

- Data is the authors' released benchmark (`reference_PVCGN/data/*.pkl`), audited with pickletools
  before loading: numpy arrays only.
- Released splits are chronological and match the paper's Table I exactly.
  HZ: train 01-01..01-18, val 01-19..01-20 (Sat–Sun), test 01-21..01-25 (Mon–Fri).
  SH: train 07-01..08-31, val 09-01..09-09, test 09-10..09-30 (holiday 09-15..09-17; Sun 09-18 was an official working day).
- The reference model ignores `xtime`/`ytime`: it sees only the previous 60 minutes of counts.

## Step 3 — research gaps found (evidence)

1. **No periodic / calendar context.** Same-slot-last-week inflow correlates r≈0.95 with the current
   value vs r≈0.66–0.72 for the value 60 min earlier (fig 02_periodicity_lags), yet PVCGN sees only the
   last hour and no time-of-day/day-type. The authors list periodicity as future work (Sec. VI).
2. **Static virtual graphs.** Similarity/correlation graphs are computed once over the whole training
   period; a station keeps only ~16% of its top-10 correlated stations between AM and PM peaks
   (fig 02_dynamic_similarity).
3. **No uncertainty.** Noise scales with volume (fig 02_heteroscedasticity); PVCGN gives point forecasts only.
4. **Evaluation gaps.** No seasonal-naive or lag-feature boosting baseline (paper's RF/GBDT saw only the
   last hour); single headline run; HZ model selection on a weekend-only validation set; holiday days not analysed.
5. **Cost.** PVCGN inference is ~20× DCRNN (paper Table VIII).

## Step 3 — prediction contract (draft for Gate A)

- **Target:** inflow and outflow passenger counts per station per 15-minute interval, for the next
  4 intervals (15/30/45/60 min ahead). Units: passengers / 15 min.
- **Prediction unit:** (station, forecast origin t); origins every 15 min, 66 per day (benchmark protocol,
  service hours 05:30–23:30).
- **Information available at t:** all stations' counts up to and including t (any past day), calendar
  (time of day, day of week, public-holiday / make-up working-day calendar — known in advance), graphs
  built from the training period only. Not available: any count after t, test-period OD matrices.
- **Splits:** the released chronological splits (identical to the paper so numbers are comparable);
  periodic look-backs may read earlier days' counts (that is history, not leakage).
- **Objective / primary metric:** MAE (also the training loss), model selection on validation only.
  Secondary: RMSE, MAPE (zeros masked, exactly the reference `lib/metrics.py`), per horizon, plus rush
  hours (07:30–09:30, 17:30–19:30) and top-1/4 ridership stations. If uncertainty is added: pinball loss
  and 80% interval coverage.
- **Robustness:** ≥3 seeds per model, mean ± std reported.
- **Constraints:** Apple laptop, 8 GB RAM, MPS GPU; inference for the whole network must stay < 1 s.

## Phase 3 plan (for Gate B)

Model rationale:
- traits: multi-output regression (inflow+outflow, 4 horizons, every station), has_datetime, stations repeated in every split by design (network forecasting, not group generalisation); HZ 1,056 train windows x 80 stations, SH 3,960 x 288; heteroscedastic counts; strong daily/weekly periodicity
- baseline: persistence (last value), seasonal naive (same-type reference day; weekly reference), historical average per station-slot-daytype from train; simple model = ridge regression on lag + reference + calendar features; strong classical = HistGradientBoosting on the same features (one model per horizon x direction)
- candidates: PVCGN (faithful re-implementation, paper config) as the published reference; PT-PVCGN ablations +P (periodic refs), +P+T (calendar embeddings) = final model (G dropped 2026-10-23, user chose a single gap)
- ruled out: random/k-fold splits (temporal data); retuning PVCGN beyond the paper config (it is the published reference); very large hyper-parameter searches (8 GB M1, ~1 h per HZ run); LightGBM (needs a system libomp install; sklearn HGB used instead)
- metric: MAE primary (training loss and val selection, best-val epoch), RMSE and MAPE (zero-masked reference code) secondary, per horizon; rush hours and top-1/4 stations; mean +- std over seeds

Protocol details:
- Deep models: Adam lr 1e-3, eps 1e-4, grad clip 5, L1 loss on de-normalised counts, curriculum learning (k=200), units 256, 2 layers (all as reference config). Batch 32 HZ (paper text). Same training windows for every model (days with a reference).
- Test set: evaluated once at step 14 for the pre-registered set of models above (all baselines + all ablations), so the paper tables come from a single test touch; no selection on test.
- Tuning (step 11): only for PT-PVCGN, on validation, small grid (lr, gate/period-graph top-k) on HZ; PVCGN keeps the paper config.

## Monitoring and retraining plan (step 16)

- Input drift: daily network inflow/outflow totals and the share of zero-count station-intervals versus the training period; alert if a day's total leaves the training range by more than 20% on an ordinary day.
- Calendar: maintain the official holiday / make-up workday list ahead of time; irregular days are where PT-PVCGN is weakest (test error analysis), so flag them and fall back to PVCGN or historical-average forecasts on those days until a holiday-aware variant exists.
- Prediction drift: rolling 7-day MAE per horizon and per station against actual counts; retrain when the rolling MAE exceeds the validation MAE by more than 15% for three consecutive days, or when the network changes (new line or station).
- Retraining reuses this pipeline from step 1: rebuild graphs and reference days on the new training window, keep a later period as validation, freeze a new test period with guard.split, and compare against the previous model before switching.

## Figures

- Figure 01_missingness.png: No column has missing values.
- Figure 02_target_balance.png: inflow ranges 0 to 8.58e+03 with median 118; a mean-predictor baseline is the number to beat.
- Figure 02_distributions.png: Histograms of 4 numeric features; 1 are strongly skewed (outflow), which matters for scaling and outliers.
- Figure 02_correlations.png: Strongest correlation with inflow: outflow (0.47); anything above 0.95 is a leakage suspect.
- Figure 02_temporal_coverage.png: timestamp spans 2016-07-01 to 2019-01-25 over 4 months; a chronological split must hold out the latest period.
- Figure 02_daily_profile.png: Weekday ridership has sharp morning and evening peaks that weekends lack (weekday/weekend peak ratio HZ 2.07, SH 3.47); the reference model never sees the time of day or day type, so it must infer this regime from one hour of history.
- Figure 02_periodicity_lags.png: Same-slot-last-week inflow correlates r=0.950 (HZ) / 0.958 (SH) with the current value, versus 0.724 / 0.663 for the value one hour earlier - the 60-minute horizon is exactly where periodic history should help, and PVCGN does not use it.
- Figure 02_dynamic_similarity.png: Each station keeps on average only 16% of its 10 most-correlated stations between the morning and evening peak; PVCGN's similarity/correlation graphs are fixed over the whole day, so a time-dependent graph is a plausible improvement.
- Figure 02_graph_structure.png: Edge-set Jaccard overlaps (HZ P-S 0.04, P-C 0.08, S-C 0.11; SH P-S 0.03, P-C 0.06, S-C 0.02) show the three graphs encode largely different neighbours; HZ virtual graphs are threshold-based and much denser than SH's top-10 graphs.
- Figure 02_daily_totals_splits.png: Weekly cycles dominate both cities; the splits are strictly chronological, the HZ test week is all weekdays (Mon-Fri), and the SH test period contains the Mid-Autumn Festival holiday (Sep 15-17) - a known shift to check in error analysis.
- Figure 02_heteroscedasticity.png: Day-to-day spread grows with volume (heteroscedastic noise), so a single point forecast hides very different uncertainty at busy vs quiet stations - PVCGN gives no uncertainty estimate.
- Figure 04_cleaning_checks.png: HZ: 0 closed station-days, 10 of 292,000 cells >10x their train median; SH: 0 closed station-days, 166 of 3,868,416 cells >10x their train median. No station was closed on any day, and the rare spikes are plausible surges, so no row is removed; zero targets are masked in MAE/MAPE exactly as in the reference metric code.
- Figure 05_reference_days.png: Every forecast also sees the same slots on the most recent earlier day of the same type (working / non-working) and on the same weekday a week earlier when that day has the same type. Days with no earlier reference (2 HZ, 2 SH, all at the start of training) are dropped.
- Figure 06_split_timeline.png: guard.split reproduces the paper's released date split exactly (rows: HZ 760,320/84,480/211,200; SH 9,427,968/1,368,576/3,193,344 train/val/test); test fingerprints are frozen for the single final evaluation.
- Figure 07_reference_feature_strength.png: On training data the value 60 min ahead correlates HZ r=0.956 with the same-type reference day vs r=0.701 with the last observation; SH r=0.986 with the same-type reference day vs r=0.655 with the last observation - the periodic inputs carry most of the long-horizon signal.
- Figure 08_scaled_distributions.png: Counts z-scored with the training mean/std only (HZ mean 198.5, std 250.9; SH mean 208.5, std 338.4); SH validation/test match training closely, while HZ validation (a weekend) has far fewer high-volume intervals, so HZ validation scores will understate weekday difficulty.
- Figure 09_baselines_val.png: Validation MAE of the six baselines; the best is HZ: Ridge (lags+refs) (MAE 22.68); SH: Ridge (lags+refs) (MAE 22.38). Every deep model must beat this line to justify its cost.
- Figure 10_hz_learning_curves.png: PT-PVCGN converges faster and lower than PVCGN on every seed; PVCGN still gained 0.30 MAE after epoch 75 (PT-PVCGN 0.01), so PVCGN may be slightly under-trained at 100 epochs - checked before the final test.
- Figure 12_hz_val_by_horizon.png: On validation PT-PVCGN averages MAE 21.03 +- 0.04 vs PVCGN 24.40 +- 0.36 and the best baseline (ridge) 22.68; its advantage over PVCGN grows with horizon (2.44 at 15 min, 4.43 at 60 min).
- Figure 12_hz_val_error_by_time.png: PT-PVCGN is closer to the truth on 56% of validation targets; its gain is largest in the early morning and the morning and evening peaks, while for the busiest 10% of station-intervals the difference is mixed (points on both sides of zero), so the improvement comes mainly from getting the daily shape right rather than from extreme counts.
- Figure 12_val_equal_budget.png: With equal training, Hangzhou validation MAE is PVCGN 22.53+-0.15, +P 20.93+-0.04, +P+T 21.20+-0.12; Shanghai PVCGN 25.40 vs +P+T 20.79. Periodic inputs give the gain; calendar embeddings add nothing on Hangzhou under the 200-epoch schedule.
- Figure 13_val_error_slices.png: HZ: PT-PVCGN beats PVCGN on 3 of 3 slices, largest relative gain on rush hours (10%); SH: PT-PVCGN beats PVCGN on 4 of 4 slices, largest relative gain on working days (20%). Slices with no validation days (e.g. HZ working days, SH holiday) are absent.
- Figure 13_val_station_errors.png: HZ: PT-PVCGN better on 68 of 80 stations, worst relative case station 51 (1.10x PVCGN's error); SH: PT-PVCGN better on 287 of 288 stations, worst relative case station 166 (1.77x PVCGN's error). Points below the diagonal are stations where PT-PVCGN is more accurate.
- Step 14 final test: masked_mae_mean_over_seeds=22.0556 (touch 1, 2026-10-24) [Hangzhou]
- Step 14 note 2026-10-24: after the Hangzhou test evaluation, a wrongly specified consistency assert (it compared the all-rows masked MAE with the mean of per-horizon MAEs) stopped the notebook before Shanghai. The Hangzhou test predictions had been saved during that single evaluation (ml_pipeline/runs/test_predictions_hz.npz) and reproduce 22.0556 exactly; they are reused for all Hangzhou tables. No second Hangzhou test touch, no model or selection change. Shanghai is evaluated for the first time in the rerun.
- Step 14 final test: masked_mae_mean_over_seeds=24.0825 (touch 1, 2026-10-24)
- Figure 14_test_by_horizon.png: HZ test MAE: PT-PVCGN 22.06 vs re-trained PVCGN 23.85 (7.5% lower) and paper-reported PVCGN 23.79; SH test MAE: PT-PVCGN 24.09 vs re-trained PVCGN 27.44 (12.2% lower) and paper-reported PVCGN 24.77. Single test evaluation, all models scored together.
- Figure 14_test_slices.png: Test MAE by slice. PT-PVCGN wins on HZ rush hours and busiest stations and on SH working days (24.48 vs PVCGN 29.61), but is worse than PVCGN on the SH holiday (24.49 vs 21.48) and the make-up workday (33.69 vs 29.41): periodic references from ordinary days mislead it on irregular days.
