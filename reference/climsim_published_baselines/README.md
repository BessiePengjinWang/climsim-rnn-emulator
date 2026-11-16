# ClimSim published baselines

`MAE_data.csv`, `RMSE_data.csv`, `R2_data.csv`, `bias_data.csv` hold metrics for ClimSim's constant-prediction (`const`) and multiple linear regression (`mlr`) baselines. They were computed with ClimSim's own evaluation pipeline on ClimSim's own scoring data, not on the two days of data in `data/raw/`. They come from the legacy notebook `archive/legacy_svr_lr_baseline_from_ClimSim_demo.ipynb`.

They are kept as an external reference point. The main comparison in this repo trains its own constant and linear-regression baselines on day 1 and scores them on day 2 with the same code as the emulator, because several of the columns below cannot be compared directly.

## What can and cannot be compared

| Columns | Comparable to `results/metrics/`? | Why |
|---|---|---|
| MAE of `NETSW`, `FLWDS`, `SOLS`, `SOLL`, `SOLSD`, `SOLLD` | Yes | Same native units (W/m^2). A constant and a linear-regression baseline trained on day 1 and scored on day 2 land within 10% of the published values |
| `PRECC`, `PRECSC` (all metrics) | No | Published `PRECC` const MAE is 97.28 and `PRECSC` is 10.65, about 2.2e9 to 2.35e9 times the same baselines on day 1 to day 2 (about 4.1e-8 and 4.9e-9 m/s). That factor is close to water density times latent heat of vaporization (1000 kg/m^3 x 2.5e6 J/kg = 2.5e9), which suggests these two rows are in energy units (W/m^2). This was not checked against ClimSim's source. Do not rescale by a guessed factor |
| `dT/dt`, `dq/dt` (all metrics) | No | Reported after ClimSim's pressure- and area-weighted energy conversion, not in K/s and kg/kg/s |
| `R2_data.csv` (all rows) | No | The published constant baseline has R2 of -9.76 for `FLWDS`, -inf for `PRECSC` and -3091.78 for `PRECC`, and the `dq/dt` row is empty. A constant baseline scored with R2 pooled over samples gives about 0, so the aggregation differs |
| `bias_data.csv` | Not checked | The sign convention is not documented. In this repo bias is mean(prediction - truth), so a positive value means over-prediction |
