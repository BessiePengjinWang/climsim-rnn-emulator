# ClimSim RNN – a vertical-sequence emulator for atmospheric physics

<img src="https://leap-stc.github.io/ClimSim/_images/fig_1.png" alt="climsim" width="300"/>

[ClimSim](https://leap-stc.github.io/ClimSim/README.html) is a large hybrid ML-physics dataset from Columbia's [LEAP Center](https://leap.columbia.edu): given one atmospheric column's state, predict the sub-grid physics tendencies a high-resolution simulation would produce ([dataset paper](https://arxiv.org/abs/2306.08754)). This project is my exploration of the emulation problem – an RNN that reads each atmospheric column's 60 vertical levels as a sequence and predicts the tendencies and surface fluxes for that column.

## Result

Trained on one day of ClimSim's low-resolution sample data (~25k column-timesteps after a 90/10 train/val split) and evaluated on a second, entirely held-out day (~27.6k column-timesteps), the RNN has lower MAE and higher R2 than both a constant baseline and a linear-regression baseline on all ten targets. Both baselines are trained on the same data and scored with the same code. Each RNN number is the mean of three seeds with the range in brackets; a seed changes both the network initialization and the 90/10 split.

| variable | MAE, RNN (mean of 3 seeds, range) | MAE, linear regression | MAE, constant | R2, RNN | R2, linear regression |
|---|---|---|---|---|---|
| dT/dt | 1.75e-5 (1.73e-5 to 1.78e-5) | 2.17e-5 | 3.16e-5 | 0.57 | 0.48 |
| dq/dt | 1.03e-8 (1.02e-8 to 1.04e-8) | 1.15e-8 | 1.13e-8 | 0.15 | 0.13 |
| NETSW | 30.2 (29.4 to 30.6) | 69 | 203 | 0.96 | 0.86 |
| FLWDS | 9.51 (9.38 to 9.73) | 12.3 | 62.6 | 0.97 | 0.96 |
| PRECSC | 2.06e-9 (1.99e-9 to 2.14e-9) | 4.31e-9 | 4.91e-9 | 0.63 | 0.28 |
| PRECC | 2.16e-8 (2.04e-8 to 2.22e-8) | 4.51e-8 | 4.14e-8 | 0.66 | 0.33 |
| SOLS | 15.8 (15.4 to 16) | 40 | 86.1 | 0.93 | 0.76 |
| SOLL | 18.4 (18.2 to 18.6) | 43.7 | 93.2 | 0.92 | 0.75 |
| SOLSD | 8.2 (7.92 to 8.45) | 13.7 | 38.3 | 0.90 | 0.80 |
| SOLLD | 7.38 (7.18 to 7.55) | 12.7 | 22.2 | 0.78 | 0.60 |

Compared with linear regression the MAE is 1.1 to 2.5 times lower, and compared with the constant baseline 1.1 to 6.7 times lower. The moisture tendency `dq/dt` is the weakest target: its MAE is 11% below linear regression and its R2 is 0.15 against 0.13. MAE varies by up to about 8% between seeds, so small differences should not be over-read.

R2 for `dT/dt` and `dq/dt` pools all 60 levels together. Scored level by level, R2 runs from about 0.1 to 0.98 for `dT/dt` and from about -0.1 to 0.7 for `dq/dt` (levels where the target is constant are left out), and the mean of the per-level values is 0.50 and 0.23 respectively; see [`results/figures/r2_per_level.png`](results/figures/r2_per_level.png).

ClimSim's own published baseline numbers are kept in [`reference/climsim_published_baselines/`](reference/climsim_published_baselines/). The MAE of the six radiation variables is directly comparable to this repo's, and the baselines trained here land within 10% of those published values; `PRECC` and `PRECSC` become comparable after dividing the published W/m^2 values by 2.501e9. The other published columns use different units or a different aggregation (for example, ClimSim's R2 is averaged per grid column over time rather than pooled); the README in that folder explains which and why, with references to ClimSim's code.

Full numbers: [`results/metrics/rnn_emulator_metrics.csv`](results/metrics/rnn_emulator_metrics.csv) (summary over seeds), [`rnn_emulator_metrics_by_seed.csv`](results/metrics/rnn_emulator_metrics_by_seed.csv) and [`baseline_metrics.csv`](results/metrics/baseline_metrics.csv); plots in [`results/figures/`](results/figures/). See [`notebooks/01_rnn_climate_tendency_emulator.ipynb`](notebooks/01_rnn_climate_tendency_emulator.ipynb) for the full walkthrough.

## Why a vertical-sequence RNN

An earlier version of this notebook (kept in [`archive/legacy_quickstart_RNN_original.ipynb`](archive/legacy_quickstart_RNN_original.ipynb)) treated each *day* as a single training example – 72 timesteps as the sequence axis, with all 384 grid columns × 60 levels flattened into one 47,616-dimensional feature vector per timestep. That yields exactly 2 training examples total and no clear physical justification for that sequence axis.

The redesign here instead treats each `(timestep, grid column)` pair as one sample, and uses the 60 vertical levels as the sequence axis – levels are physically coupled by convection and vertical mixing, so this is the axis with an actual sequential relationship to model. That turns 2 days of data into ~55k samples instead of 2, and is the same framing ClimSim's own baseline emulator uses.

## Limitations

- Two days of sample data. This demonstrates the approach; it isn't tuned or competitive at the scale of ClimSim's full multi-year dataset.
- Minimal feature set: temperature, humidity, surface pressure, and three surface fluxes as input – not ClimSim's full input specification (winds, ozone, additional moisture species, etc.).
- No energy-unit weighting: ClimSim's official leaderboard weights tendencies by pressure-level thickness and grid-cell area before scoring; this repo reports plain per-variable MAE/RMSE/R2/bias in native physical units instead, so its numbers cannot be submitted to that leaderboard or compared with its energy-weighted values.
- Seed variation: MAE moves by up to about 8% between seeds and the held-out set is a single day, which is why the table gives a mean and a range over three seeds.
- Training length: training is capped at 40 epochs and all three seeds reached the cap. A single-seed check with a cap of 200 epochs let early stopping end the run at epoch 62, and held-out MAE then changed by between -6% and +5% across the ten targets, within the spread between seeds, so the cap does not appear to limit the results.
- Per-level standardization: each level's targets are scaled by that level's own standard deviation, so the few near-top levels where `dq/dt` is almost always zero get the same weight in the training loss as physically important levels. This makes the validation loss noisy and seed dependent but contributes almost nothing to the reported metrics. Scaling by one standard deviation per tendency would avoid it.
- The 2-layer LSTM architecture is a reasonable starting point for the vertical-sequence framing, not a tuned or benchmarked one.

## Reproducing this

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Data: place ClimSim's sample NetCDF files under `data/raw/` (not checked into git – see [`data/README.md`](data/README.md) for the expected layout and source). Then:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_rnn_climate_tendency_emulator.ipynb
```

This regenerates the three CSV files under `results/metrics/` and the three figures under `results/figures/`. A fresh clone with a fresh virtual environment, following exactly these steps, installed in under a minute and executed the notebook end to end in about 22 minutes (CPU, Apple Silicon) with Python 3.12.12, TensorFlow 2.21.0, xarray 2026.9.0, numpy 2.5.5, pandas 3.1.1 and scikit-learn 1.9.1.

Neural-network training is not bit-for-bit repeatable, even with fixed seeds, so regenerated numbers will differ slightly from those above. In that clean re-run the three-seed mean MAE moved by at most about 2%, R2 by at most 0.011, and the two baselines were identical; a single run of one seed moved by up to about 6.5%.

The notebook itself is generated from [`scripts/build_notebook.py`](scripts/build_notebook.py) – that script is the source of truth for its content; edit it and re-run rather than hand-editing the `.ipynb`. Running it clears the notebook's stored outputs, so execute the notebook afterwards.

## Repository structure

```
.
├── src/climsim_rnn/    data loading, preprocessing/normalization, model, metrics, baselines
├── notebooks/          the emulator walkthrough (01_rnn_climate_tendency_emulator.ipynb)
├── scripts/            build_notebook.py generates the notebook above
├── data/               raw/ holds the NetCDF input (git-ignored except a .gitkeep, see data/README.md)
├── results/            figures/ and metrics/ produced by the notebook
├── reference/          ClimSim's published baseline metrics, with notes on what is comparable
└── archive/            earlier drafts, kept for reference
```
