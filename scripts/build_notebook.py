import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []}


def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else [],
    }


cells = []

cells.append(md(
    "# A vertical-sequence RNN emulator for ClimSim",
    "",
    "ClimSim ([LEAP Center](https://leap.columbia.edu), [dataset paper](https://arxiv.org/abs/2306.08754)) is a "
    "large hybrid ML-physics dataset for climate modeling: given one atmospheric column's state, predict the "
    "sub-grid physics tendencies a high-resolution simulation would produce. This notebook builds a small RNN "
    "emulator for it and evaluates it against a constant-prediction baseline and a linear-regression baseline "
    "trained on the same data.",
    "",
    "**Design note.** An earlier version of this notebook (kept in `archive/legacy_quickstart_RNN_original.ipynb` "
    "for comparison) treated each *day* as a single training example: 72 timesteps as the sequence axis, and all "
    "384 grid columns x 60 levels flattened into one 47,616-dimensional feature vector per timestep. That gives "
    "exactly 2 training examples (one per day) and no clear physical reason for that particular sequence axis.",
    "",
    "ClimSim's own baseline instead predicts one grid column at a time, and there *is* a natural sequence axis "
    "within a column: the 60 vertical levels, which are physically coupled by convection and vertical mixing. "
    "This notebook uses that framing: each `(timestep, column)` pair is one sample, and the RNN reads down the "
    "column level by level. That turns 2 days of data into ~55k samples instead of 2.",
))

cells.append(md("## Setup"))

cells.append(code(
    "from dataclasses import replace",
    "from pathlib import Path",
    "",
    "import matplotlib.pyplot as plt",
    "import numpy as np",
    "import pandas as pd",
    "import tensorflow as tf",
    "from sklearn.model_selection import train_test_split",
    "",
    "from climsim_rnn import (",
    "    LEVEL_TARGET_VARS,",
    "    SCALAR_TARGET_VARS,",
    "    Normalizer,",
    "    build_model,",
    "    build_samples,",
    "    evaluate,",
    "    load_day,",
    "    predict_constant,",
    "    predict_linear_regression,",
    ")",
    "",
    'RAW_DIR = Path("../data/raw")',
    'BASELINE_DIR = Path("../reference/climsim_published_baselines")',
    'FIG_DIR = Path("../results/figures")',
    'METRICS_DIR = Path("../results/metrics")',
    "FIG_DIR.mkdir(parents=True, exist_ok=True)",
    "METRICS_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "SEEDS = [0, 1, 2]",
))

cells.append(md(
    "## Load and reshape",
    "",
    "Day 1 supplies the training/validation split; day 2 is held out entirely and used only for final "
    "evaluation, so the reported metrics reflect generalization to an unseen day rather than an unseen "
    "random subset of the same day.",
))

cells.append(code(
    "ds_in_day1, ds_out_day1 = load_day(RAW_DIR, \"day1\")",
    "ds_in_day2, ds_out_day2 = load_day(RAW_DIR, \"day2\")",
    "print(\"day1 grid:\", dict(ds_in_day1.sizes))",
    "print(\"day2 grid:\", dict(ds_in_day2.sizes))",
))

cells.append(code(
    "day1_samples = build_samples(ds_in_day1, ds_out_day1)",
    "day2_samples = build_samples(ds_in_day2, ds_out_day2)",
    "",
    "print(\"day1 samples:\", day1_samples.level_features.shape[0])",
    "print(\"day2 samples (held-out test):\", day2_samples.level_features.shape[0])",
))

cells.append(md(
    "## One training run",
    "",
    "A run does four things, in this order:",
    "",
    "1. **Split.** A random 90/10 train/validation split of day 1. The seed controls this split and the network "
    "initialization together.",
    "2. **Normalize.** Per-level mean and standard deviation are fit on the training split only and then applied to "
    "validation and to the held-out day 2 -- fitting on anything but train would leak information about data the "
    "model is later evaluated on. The standard deviation is floored at machine epsilon in both directions, so only "
    "exactly-constant levels are affected and the inverse transform undoes the forward transform.",
    "3. **Train.** Adam, equal-weight MSE on the two outputs, at most 40 epochs, early stopping on validation loss "
    "with patience 5.",
    "4. **Evaluate.** Predictions on day 2 are converted back to physical units and scored with `evaluate`.",
    "",
    "The cell below defines these steps as a function so the identical procedure can be repeated for several seeds.",
))

cells.append(code(
    "def select_samples(batch, index):",
    "    return replace(",
    "        batch,",
    "        level_features=batch.level_features[index],",
    "        aux_features=batch.aux_features[index],",
    "        level_targets=batch.level_targets[index],",
    "        scalar_targets=batch.scalar_targets[index],",
    "    )",
    "",
    "",
    "def split_indices(seed):",
    "    return train_test_split(np.arange(day1_samples.level_features.shape[0]), test_size=0.1, random_state=seed)",
    "",
    "",
    "def build_emulator(samples):",
    "    return build_model(",
    "        levels=samples.level_features.shape[1],",
    "        level_input_features=samples.level_features.shape[2],",
    "        aux_input_features=samples.aux_features.shape[1],",
    "        level_target_features=samples.level_targets.shape[2],",
    "        scalar_target_features=samples.scalar_targets.shape[1],",
    "    )",
    "",
    "",
    "def run_seed(seed):",
    "    np.random.seed(seed)",
    "    tf.random.set_seed(seed)",
    "",
    "    train_idx, val_idx = split_indices(seed)",
    "    train_samples = select_samples(day1_samples, train_idx)",
    "    val_samples = select_samples(day1_samples, val_idx)",
    "    print(f\"seed {seed}: train {len(train_idx)}, val {len(val_idx)}, test {day2_samples.level_features.shape[0]}\")",
    "",
    "    normalizer = Normalizer()",
    "    train_n = normalizer.fit_transform(train_samples)",
    "    val_n = normalizer.transform(val_samples)",
    "    test_n = normalizer.transform(day2_samples)",
    "    for name, batch in [(\"train\", train_n), (\"val\", val_n), (\"test\", test_n)]:",
    "        arrays = [batch.level_features, batch.aux_features, batch.level_targets, batch.scalar_targets]",
    "        assert all(np.isfinite(a).all() for a in arrays), f\"non-finite values in {name}\"",
    "",
    "    model = build_emulator(train_samples)",
    "    history = model.fit(",
    "        [train_n.level_features, train_n.aux_features],",
    "        [train_n.level_targets, train_n.scalar_targets],",
    "        validation_data=(",
    "            [val_n.level_features, val_n.aux_features],",
    "            [val_n.level_targets, val_n.scalar_targets],",
    "        ),",
    "        epochs=40,",
    "        batch_size=256,",
    "        callbacks=[tf.keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=5, restore_best_weights=True)],",
    "        verbose=2,",
    "    )",
    "",
    "    pred_level_n, pred_scalar_n = model.predict(",
    "        [test_n.level_features, test_n.aux_features], batch_size=512, verbose=0",
    "    )",
    "    pred_level = normalizer.inverse_transform_level_targets(pred_level_n)",
    "    pred_scalar = normalizer.inverse_transform_scalar_targets(pred_scalar_n)",
    "",
    "    return {",
    "        \"history\": history.history,",
    "        \"epochs_run\": len(history.history[\"loss\"]),",
    "        \"level_pred\": pred_level,",
    "        \"metrics\": evaluate(day2_samples.level_targets, pred_level, day2_samples.scalar_targets, pred_scalar),",
    "    }",
))

cells.append(md("## Model"))

cells.append(code(
    "build_emulator(day1_samples).summary()",
))

cells.append(md(
    "## Train three seeds",
    "",
    "A single training run of a neural network is not exactly repeatable: with the seed fixed, two runs can still "
    "differ slightly. Three seeds are trained here, and the metrics below are reported as the mean and the range "
    "across them.",
))

cells.append(code(
    "runs = {seed: run_seed(seed) for seed in SEEDS}",
    "print(\"epochs run per seed (cap is 40):\", {seed: run[\"epochs_run\"] for seed, run in runs.items()})",
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(7, 4))",
    "for seed, run in runs.items():",
    "    line, = ax.plot(run[\"history\"][\"loss\"], label=f\"train, seed {seed}\")",
    "    ax.plot(run[\"history\"][\"val_loss\"], linestyle=\"--\", color=line.get_color(), label=f\"val, seed {seed}\")",
    "ax.set_xlabel(\"epoch\")",
    "ax.set_ylabel(\"loss (normalized MSE)\")",
    "ax.set_title(\"Training curves\")",
    "ax.legend(ncol=2, fontsize=8)",
    "fig.tight_layout()",
    "fig.savefig(FIG_DIR / \"training_loss.png\", dpi=150)",
    "plt.show()",
))

cells.append(md(
    "## Evaluate on the held-out day",
    "",
    "Metrics are reported in physical units (K/s, kg/kg/s, W/m^2) after inverting the normalization. The two level "
    "variables are pooled over all levels and samples, so their R2 is measured against the pooled variance and is "
    "dominated by the high-variance levels (a per-level view follows below). The eight scalar surface variables are "
    "pooled over samples. Bias is mean(prediction - truth), so a positive value means over-prediction.",
))

cells.append(code(
    "by_seed = pd.concat(",
    "    [run[\"metrics\"].assign(seed=seed) for seed, run in runs.items()], ignore_index=True",
    ")[[\"seed\", \"variable\", \"mae\", \"rmse\", \"r2\", \"bias\"]]",
    "by_seed.to_csv(METRICS_DIR / \"rnn_emulator_metrics_by_seed.csv\", index=False)",
    "",
    "grouped = by_seed.groupby(\"variable\", sort=False)",
    "summary = pd.DataFrame({",
    "    \"mae_mean\": grouped[\"mae\"].mean(),",
    "    \"mae_min\": grouped[\"mae\"].min(),",
    "    \"mae_max\": grouped[\"mae\"].max(),",
    "    \"rmse_mean\": grouped[\"rmse\"].mean(),",
    "    \"r2_mean\": grouped[\"r2\"].mean(),",
    "    \"r2_min\": grouped[\"r2\"].min(),",
    "    \"r2_max\": grouped[\"r2\"].max(),",
    "    \"bias_mean\": grouped[\"bias\"].mean(),",
    "}).reset_index()",
    "summary.to_csv(METRICS_DIR / \"rnn_emulator_metrics.csv\", index=False)",
    "summary",
))

cells.append(md(
    "## Comparison against constant and linear-regression baselines",
    "",
    "Both baselines are fit on the same 90% training split of day 1 as seed 0 and scored on day 2 with the same "
    "`evaluate` function as the RNN, so the three models are compared on identical data. They are deterministic, so "
    "they are computed once.",
    "",
    "- **Constant:** predicts the training mean of every target, level by level, whatever the input.",
    "- **Linear regression:** ordinary least squares from the flattened, normalized inputs (60 levels x 2 variables "
    "plus 4 surface inputs) to all 128 outputs.",
))

cells.append(code(
    "baseline_train = select_samples(day1_samples, split_indices(SEEDS[0])[0])",
    "baseline_predictors = {\"constant\": predict_constant, \"linear_regression\": predict_linear_regression}",
    "",
    "baseline_frames = []",
    "for model_name, predictor in baseline_predictors.items():",
    "    level_pred, scalar_pred = predictor(baseline_train, day2_samples)",
    "    baseline_frames.append(",
    "        evaluate(day2_samples.level_targets, level_pred, day2_samples.scalar_targets, scalar_pred).assign(model=model_name)",
    "    )",
    "baseline_metrics = pd.concat(baseline_frames, ignore_index=True)[[\"model\", \"variable\", \"mae\", \"rmse\", \"r2\", \"bias\"]]",
    "baseline_metrics.to_csv(METRICS_DIR / \"baseline_metrics.csv\", index=False)",
    "",
    "def baseline_column(model_name, metric):",
    "    return baseline_metrics[baseline_metrics[\"model\"] == model_name].set_index(\"variable\")[metric]",
    "",
    "rnn_summary = summary.set_index(\"variable\")",
    "mae_table = pd.DataFrame({",
    "    \"rnn\": rnn_summary[\"mae_mean\"],",
    "    \"linear_regression\": baseline_column(\"linear_regression\", \"mae\"),",
    "    \"constant\": baseline_column(\"constant\", \"mae\"),",
    "})",
    "mae_table[\"rnn / linear_regression\"] = mae_table[\"rnn\"] / mae_table[\"linear_regression\"]",
    "mae_table[\"rnn / constant\"] = mae_table[\"rnn\"] / mae_table[\"constant\"]",
    "print(\"MAE on held-out day 2 (RNN = mean over seeds; ratios below 1 mean the RNN is better)\")",
    "mae_table",
))

cells.append(code(
    "r2_table = pd.DataFrame({",
    "    \"rnn\": rnn_summary[\"r2_mean\"],",
    "    \"rnn_min\": rnn_summary[\"r2_min\"],",
    "    \"rnn_max\": rnn_summary[\"r2_max\"],",
    "    \"linear_regression\": baseline_column(\"linear_regression\", \"r2\"),",
    "    \"constant\": baseline_column(\"constant\", \"r2\"),",
    "})",
    "print(\"R2 on held-out day 2 (higher is better)\")",
    "r2_table",
))

cells.append(code(
    "# Raw MAE isn't comparable across variables here -- dT/dt is O(1e-5) K/s and NETSW is",
    "# O(10) W/m^2, so a shared linear axis makes most bars invisible. Normalizing by each",
    "# variable's own standard deviation (on the same held-out day) gives a scale-free \"fraction",
    "# of natural variability\" that's comparable across variables.",
    "target_std = np.concatenate(",
    "    [day2_samples.level_targets.std(axis=(0, 1)), day2_samples.scalar_targets.std(axis=0)]",
    ")",
    "variables = summary[\"variable\"].to_numpy()",
    "normalized_mae = {",
    "    \"RNN (mean of seeds)\": mae_table.loc[variables, \"rnn\"].to_numpy() / target_std,",
    "    \"linear regression\": mae_table.loc[variables, \"linear_regression\"].to_numpy() / target_std,",
    "    \"constant\": mae_table.loc[variables, \"constant\"].to_numpy() / target_std,",
    "}",
    "",
    "fig, ax = plt.subplots(figsize=(10, 5))",
    "x = np.arange(len(variables))",
    "width = 0.27",
    "for offset, (label, values) in zip((-width, 0, width), normalized_mae.items()):",
    "    ax.bar(x + offset, values, width=width, label=label)",
    "ax.set_xticks(x, variables, rotation=45, ha=\"right\")",
    "ax.set_ylabel(\"MAE / std(target) (lower is better)\")",
    "ax.set_title(\"Normalized error per variable, held-out day 2\")",
    "ax.legend()",
    "fig.tight_layout()",
    "fig.savefig(FIG_DIR / \"normalized_error_per_variable.png\", dpi=150)",
    "plt.show()",
))

cells.append(md(
    "## How R2 varies with height",
    "",
    "The pooled R2 above is a single number per tendency. The plot below scores each of the 60 levels separately "
    "(level 0 is the top of the atmosphere, level 59 is next to the surface). The line is the mean over seeds and the "
    "band is the range. Levels where the held-out target is exactly constant have no defined R2 and are left blank.",
))

cells.append(code(
    "def per_level_r2(level_true, level_pred, target_index):",
    "    truth = level_true[..., target_index].astype(np.float64)",
    "    prediction = level_pred[..., target_index].astype(np.float64)",
    "    residual_sum = ((truth - prediction) ** 2).sum(axis=0)",
    "    total_sum = ((truth - truth.mean(axis=0)) ** 2).sum(axis=0)",
    "    has_variance = total_sum > 0",
    "    return np.where(has_variance, 1 - residual_sum / np.where(has_variance, total_sum, 1.0), np.nan)",
    "",
    "",
    "per_level = {",
    "    name: np.stack([per_level_r2(day2_samples.level_targets, run[\"level_pred\"], index) for run in runs.values()])",
    "    for index, name in enumerate(LEVEL_TARGET_VARS)",
    "}",
    "",
    "fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)",
    "levels = np.arange(day2_samples.level_targets.shape[1])",
    "for ax, name in zip(axes, LEVEL_TARGET_VARS):",
    "    values = per_level[name]",
    "    ax.fill_betweenx(levels, values.min(axis=0), values.max(axis=0), alpha=0.3)",
    "    ax.plot(values.mean(axis=0), levels)",
    "    ax.axvline(0, color=\"gray\", linewidth=0.8)",
    "    ax.set_xlabel(\"R2 at this level\")",
    "    ax.set_title(name)",
    "axes[0].set_ylabel(\"level index (0 = top of atmosphere)\")",
    "axes[0].invert_yaxis()",
    "fig.tight_layout()",
    "fig.savefig(FIG_DIR / \"r2_per_level.png\", dpi=150)",
    "plt.show()",
    "",
    "pd.DataFrame({",
    "    \"pooled_r2 (reported above)\": rnn_summary.loc[LEVEL_TARGET_VARS, \"r2_mean\"],",
    "    \"mean_of_per_level_r2\": {name: float(np.nanmean(per_level[name].mean(axis=0))) for name in LEVEL_TARGET_VARS},",
    "})",
))

cells.append(md(
    "## Sanity check against ClimSim's published baselines",
    "",
    "`../reference/climsim_published_baselines/` holds MAE/RMSE/R2/bias for ClimSim's own constant-prediction "
    "(`const`) and multiple linear regression (`mlr`) baselines, computed with ClimSim's pipeline on ClimSim's own "
    "scoring data. Only part of it is comparable to this notebook (see the README in that folder):",
    "",
    "- The MAE of the six radiation variables is in the same units, so the baselines above should land close to the "
    "published values. The table below checks this.",
    "- `PRECC` and `PRECSC` appear to be in different units, `dT/dt` and `dq/dt` are reported in ClimSim's "
    "energy-weighted units, and the published R2 uses a different aggregation, so none of those are compared.",
))

cells.append(code(
    "name_map = {",
    "    \"$dT/dt$\": \"dT/dt\",",
    "    \"$dq/dt$\": \"dq/dt\",",
    "    **{v.removeprefix(\"cam_out_\"): v for v in SCALAR_TARGET_VARS},",
    "}",
    "published_mae = pd.read_csv(BASELINE_DIR / \"MAE_data.csv\")",
    "published_mae[\"variable\"] = published_mae[\"variable\"].replace(name_map)",
    "published_mae = published_mae.set_index(\"variable\")",
    "",
    "radiation_variables = [\"cam_out_NETSW\", \"cam_out_FLWDS\", \"cam_out_SOLS\", \"cam_out_SOLL\", \"cam_out_SOLSD\", \"cam_out_SOLLD\"]",
    "published_check = pd.DataFrame({",
    "    \"published_const\": published_mae.loc[radiation_variables, \"const\"],",
    "    \"our_constant\": mae_table.loc[radiation_variables, \"constant\"],",
    "    \"published_mlr\": published_mae.loc[radiation_variables, \"mlr\"],",
    "    \"our_linear_regression\": mae_table.loc[radiation_variables, \"linear_regression\"],",
    "})",
    "published_check[\"const_diff_%\"] = 100 * (published_check[\"our_constant\"] / published_check[\"published_const\"] - 1)",
    "published_check[\"mlr_diff_%\"] = 100 * (published_check[\"our_linear_regression\"] / published_check[\"published_mlr\"] - 1)",
    "published_check",
))

cells.append(md(
    "## Limitations and next steps",
    "",
    "- **Two days of data.** Training uses one day (~25k column-timesteps after the train/val split), tested "
    "on a single held-out day. ClimSim's full dataset spans years; results here show the approach works, not "
    "that it's tuned or competitive at scale.",
    "- **Seed variation.** MAE moves by up to about 8% between seeds (range as a share of the mean), and the "
    "held-out day is only one day, so small differences between models should not be over-read. The mean and range "
    "over three seeds are reported for that reason.",
    "- **Training length.** Training is capped at 40 epochs, and the printed epoch counts show whether early "
    "stopping ended a run first. A single-seed check outside this notebook with `epochs=200` let early stopping end "
    "the run at epoch 62; held-out day-2 MAE then changed by between -6% and +5% across the ten variables, which "
    "is within the spread between seeds, so the cap does not appear to be what limits these results. The training "
    "cell's `epochs` argument is the only change needed to repeat that check.",
    "- **Minimal feature set.** Inputs are limited to temperature, humidity, surface pressure, and three "
    "surface fluxes -- the subset the original notebook used. ClimSim's full input specification also "
    "includes winds, ozone, additional moisture species, and more.",
    "- **No energy-unit weighting.** ClimSim's official leaderboard weights tendencies by pressure-level "
    "thickness and grid-cell area to put every variable in a common energy unit before scoring. Metrics here "
    "are plain per-variable MAE/RMSE/R2/bias in each variable's native physical unit, so they cannot be "
    "submitted to ClimSim's leaderboard or compared directly with its energy-weighted numbers.",
    "- **Moisture tendency.** `dq/dt` is the weakest target: the RNN is only modestly better than linear regression "
    "on it. The per-level plot shows R2 reaching about 0.65 only around levels 25 to 32, while it stays below about "
    "0.1 over levels 45 to 54, where the largest share of the held-out variance sits.",
    "- **Per-level standardization.** Each level's targets are scaled by that level's own standard deviation, so "
    "the few levels near the top where `dq/dt` is almost always zero (training standard deviation between about "
    "1e-14 and 1e-10) get the same weight in the training loss as levels that matter physically. Rare spikes there "
    "become very large normalized values (up to about 160 in the seed-0 validation split), which inflates the "
    "validation loss and makes its level differ between seeds. The reported metrics are in physical units, where "
    "those levels contribute almost nothing. Scaling `dT/dt` and `dq/dt` by one standard deviation each, across all "
    "levels, would avoid this and is a possible improvement.",
    "- **Architecture.** A 2-layer LSTM was a reasonable starting point given the vertical-sequence framing; "
    "it hasn't been tuned (units, depth, dropout, learning-rate schedule) or compared against non-recurrent "
    "alternatives (e.g. a plain MLP per level, or attention over levels).",
))

for position, cell in enumerate(cells):
    cell["id"] = f"cell-{position:03d}"

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (climsim-rnn)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/01_rnn_climate_tendency_emulator.ipynb")
out_path.write_text(json.dumps(nb, indent=1))
print("wrote", out_path)
