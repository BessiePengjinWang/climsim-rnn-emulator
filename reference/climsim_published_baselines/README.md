# ClimSim published baselines

`MAE_data.csv`, `RMSE_data.csv`, `R2_data.csv`, `bias_data.csv` hold metrics for ClimSim's constant-prediction (`const`) and multiple linear regression (`mlr`) baselines. They were computed with ClimSim's own evaluation code on ClimSim's own scoring data, not on the two days of data in `data/raw/`. They come from the legacy notebook `archive/legacy_svr_lr_baseline_from_ClimSim_demo.ipynb`.

They are kept as an external reference point. The main comparison in this repo trains its own constant and linear-regression baselines on day 1 and scores them on day 2 with the same code as the emulator, because several of the columns below cannot be compared directly.

## What can and cannot be compared

The conventions below were checked against `climsim_utils/data_utils.py` and `evaluation/ClimSim_metrics.many-trials.v2.py` in [leap-stc/ClimSim](https://github.com/leap-stc/ClimSim) (main branch, 2026-11-16). This repo does not record which version of that code produced the CSV files.

| Columns | Comparable to `results/metrics/`? | Why |
|---|---|---|
| MAE of `NETSW`, `FLWDS`, `SOLS`, `SOLL`, `SOLSD`, `SOLLD` | Yes, approximately | Same native units (W/m^2). ClimSim multiplies every variable by a grid-cell area weight (cell area divided by the mean area) before scoring and this repo does not. A constant and a linear-regression baseline trained on day 1 and scored on day 2 land within 10% of the published values |
| MAE of `PRECC`, `PRECSC` | Yes, after converting | ClimSim converts both to W/m^2 by multiplying by the latent heat of evaporation times the density of water (2.501e6 J/kg x 1e3 kg/m^3 = 2.501e9), so divide the published values by 2.501e9 to get m/s. After converting, the constant baselines trained here are 6% (`PRECC`) and 15% (`PRECSC`) above the published ones, and the linear-regression baselines are 22% and 44% above, which is plausible because the published regression was fit on far more data and inputs |
| `dT/dt`, `dq/dt` (all metrics) | No | ClimSim weights by pressure thickness (dp/g), multiplies by the specific heat or latent heat, applies the area weight, and reports the result in W/m^2, not in K/s and kg/kg/s |
| `RMSE_data.csv` (all rows) | No | ClimSim computes the RMSE of each grid column over time and then averages the per-column values. This repo computes one RMSE over all samples pooled, which is a different quantity |
| `R2_data.csv` (all rows) | No | ClimSim computes R2 for each grid column over time and then averages over columns, so a column whose target barely changes in time gives -inf or a very large negative value (the published `PRECSC` row is -inf, `FLWDS` const is -9.76, `PRECC` const is -3091.78, and the `dq/dt` row is empty). This repo pools all samples, and a constant baseline then scores about 0 |
| `bias_data.csv` | Sign convention only | Both define bias as prediction minus truth, so a positive value means over-prediction. ClimSim averages per grid column and area-weights, so the values are not compared numerically |
