# ClimSim RNN — a vertical-sequence emulator for atmospheric physics

<img src="https://leap-stc.github.io/ClimSim/_images/fig_1.png" alt="climsim" width="300"/>

[ClimSim](https://leap-stc.github.io/ClimSim/README.html) is a large hybrid ML-physics dataset from Columbia's [LEAP Center](https://leap.columbia.edu): given one atmospheric column's state, predict the sub-grid physics tendencies a high-resolution simulation would produce ([dataset paper](https://arxiv.org/abs/2306.08754)). This project is my exploration of the emulation problem — an RNN that reads each atmospheric column's 60 vertical levels as a sequence and predicts the tendencies and surface fluxes for that column.

## Result

Trained on one day of ClimSim's low-resolution sample data (~25k column-timesteps after a 90/10 train/val split) and evaluated on a second, entirely held-out day (~27.6k column-timesteps), the model beats ClimSim's own constant-prediction and linear-regression baselines on every one of the eight scalar surface variables it was compared against, most by an order of magnitude or more in MAE:

| variable | MAE (this model) | MAE (ClimSim `const` baseline) | MAE (ClimSim `mlr` baseline) |
|---|---|---|---|
| NETSW | 30.1 | 195.8 | 63.0 |
| FLWDS | 9.4 | 58.4 | 12.3 |
| SOLS | 16.1 | 83.1 | 37.2 |
| SOLL | 18.7 | 89.0 | 40.0 |
| SOLSD | 8.4 | 37.3 | 12.8 |
| SOLLD | 7.6 | 21.6 | 12.0 |
| PRECC | 2.0e-8 | 9.7e-8 | 9.3e-8 |
| PRECSC | 2.1e-9 | 1.1e-8 | 7.5e-9 |

R2 on the temperature tendency (`dT/dt`) is 0.58 and on the moisture tendency (`dq/dt`) is 0.15 — moisture is the harder target, consistent with what ClimSim's own baselines report. Full numbers: [`results/metrics/rnn_emulator_metrics.csv`](results/metrics/rnn_emulator_metrics.csv); training curve and per-variable error plots: [`results/figures/`](results/figures/). See [`notebooks/01_rnn_climate_tendency_emulator.ipynb`](notebooks/01_rnn_climate_tendency_emulator.ipynb) for the full walkthrough, including exactly which numbers are and aren't comparable to the archived baselines.

## Why a vertical-sequence RNN

An earlier version of this notebook (kept in [`archive/legacy_quickstart_RNN_original.ipynb`](archive/legacy_quickstart_RNN_original.ipynb)) treated each *day* as a single training example — 72 timesteps as the sequence axis, with all 384 grid columns × 60 levels flattened into one 47,616-dimensional feature vector per timestep. That yields exactly 2 training examples total and no clear physical justification for that sequence axis.

The redesign here instead treats each `(timestep, grid column)` pair as one sample, and uses the 60 vertical levels as the sequence axis — levels are physically coupled by convection and vertical mixing, so this is the axis with an actual sequential relationship to model. That turns 2 days of data into ~55k samples instead of 2, and is the same framing ClimSim's own baseline emulator uses.

## Reproducing this

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Data: place ClimSim's sample NetCDF files under `data/raw/` (not checked into git — see [`data/README.md`](data/README.md) for the expected layout and source). Then:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/01_rnn_climate_tendency_emulator.ipynb
```

This regenerates `results/metrics/rnn_emulator_metrics.csv` and the figures under `results/figures/`. Ran end-to-end on Python 3.12 / TensorFlow 2.21 / xarray 2026.7 (CPU, Apple Silicon) in about 8 minutes.

The notebook itself is generated from [`scripts/build_notebook.py`](scripts/build_notebook.py) — that script is the source of truth for its content; edit it and re-run rather than hand-editing the `.ipynb`.

## Layout

```
.
├── src/climsim_rnn/    data loading, preprocessing/normalization, model, metrics
├── notebooks/          the emulator walkthrough (01_rnn_climate_tendency_emulator.ipynb)
├── scripts/            build_notebook.py generates the notebook above
├── data/                raw/ (git-ignored NetCDF input, see data/README.md)
├── results/             figures/ and metrics/ produced by the notebook
└── archive/             earlier drafts and ClimSim's own baseline metrics, kept for reference
```

## Limitations

- Two days of sample data. This demonstrates the approach; it isn't tuned or competitive at the scale of ClimSim's full multi-year dataset.
- Minimal feature set: temperature, humidity, surface pressure, and three surface fluxes as input — not ClimSim's full input specification (winds, ozone, additional moisture species, etc.).
- No energy-unit weighting: ClimSim's official leaderboard weights tendencies by pressure-level thickness and grid-cell area before scoring; this repo reports plain per-variable MAE/RMSE/R2/bias in native physical units instead (see the notebook for exactly which comparisons that does and doesn't affect).
- The 2-layer LSTM architecture is a reasonable starting point for the vertical-sequence framing, not a tuned or benchmarked one.
