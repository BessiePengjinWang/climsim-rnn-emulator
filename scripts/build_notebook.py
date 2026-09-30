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
    "emulator for it and evaluates it against ClimSim's own constant-prediction and linear-regression baselines.",
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
    ")",
    "",
    "RAW_DIR = Path(\"../data/raw\")",
    "FIG_DIR = Path(\"../results/figures\")",
    "METRICS_DIR = Path(\"../results/metrics\")",
    "FIG_DIR.mkdir(parents=True, exist_ok=True)",
    "METRICS_DIR.mkdir(parents=True, exist_ok=True)",
    "",
    "SEED = 0",
    "np.random.seed(SEED)",
    "tf.random.set_seed(SEED)",
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

cells.append(code(
    "train_idx, val_idx = train_test_split(",
    "    np.arange(day1_samples.level_features.shape[0]), test_size=0.1, random_state=SEED",
    ")",
    "",
    "from dataclasses import replace",
    "",
    "train_samples = replace(",
    "    day1_samples,",
    "    level_features=day1_samples.level_features[train_idx],",
    "    aux_features=day1_samples.aux_features[train_idx],",
    "    level_targets=day1_samples.level_targets[train_idx],",
    "    scalar_targets=day1_samples.scalar_targets[train_idx],",
    ")",
    "val_samples = replace(",
    "    day1_samples,",
    "    level_features=day1_samples.level_features[val_idx],",
    "    aux_features=day1_samples.aux_features[val_idx],",
    "    level_targets=day1_samples.level_targets[val_idx],",
    "    scalar_targets=day1_samples.scalar_targets[val_idx],",
    ")",
    "print(\"train:\", train_samples.level_features.shape[0], \"val:\", val_samples.level_features.shape[0])",
))

cells.append(md(
    "## Normalize",
    "",
    "Statistics are fit on the training split only, then applied to validation and the held-out day 2 test "
    "set -- fitting on anything but train would leak information about data the model is later evaluated on.",
))

cells.append(code(
    "normalizer = Normalizer()",
    "train_n = normalizer.fit_transform(train_samples)",
    "val_n = normalizer.transform(val_samples)",
    "test_n = normalizer.transform(day2_samples)",
    "",
    "for name, batch in [(\"train\", train_n), (\"val\", val_n), (\"test\", test_n)]:",
    "    arrays = [batch.level_features, batch.aux_features, batch.level_targets, batch.scalar_targets]",
    "    assert all(np.isfinite(a).all() for a in arrays), f\"non-finite values in {name}\"",
    "print(\"all splits finite after normalization\")",
))

cells.append(md("## Model"))

cells.append(code(
    "model = build_model(",
    "    levels=train_samples.level_features.shape[1],",
    "    level_input_features=train_samples.level_features.shape[2],",
    "    aux_input_features=train_samples.aux_features.shape[1],",
    "    level_target_features=train_samples.level_targets.shape[2],",
    "    scalar_target_features=train_samples.scalar_targets.shape[1],",
    ")",
    "model.summary()",
))

cells.append(code(
    "callbacks = [",
    "    tf.keras.callbacks.EarlyStopping(monitor=\"val_loss\", patience=5, restore_best_weights=True),",
    "]",
    "",
    "history = model.fit(",
    "    [train_n.level_features, train_n.aux_features],",
    "    [train_n.level_targets, train_n.scalar_targets],",
    "    validation_data=(",
    "        [val_n.level_features, val_n.aux_features],",
    "        [val_n.level_targets, val_n.scalar_targets],",
    "    ),",
    "    epochs=40,",
    "    batch_size=256,",
    "    callbacks=callbacks,",
    "    verbose=2,",
    ")",
))

cells.append(code(
    "fig, ax = plt.subplots(figsize=(6, 4))",
    "ax.plot(history.history[\"loss\"], label=\"train\")",
    "ax.plot(history.history[\"val_loss\"], label=\"val\")",
    "ax.set_xlabel(\"epoch\")",
    "ax.set_ylabel(\"loss (normalized MSE)\")",
    "ax.set_title(\"Training curve\")",
    "ax.legend()",
    "fig.tight_layout()",
    "fig.savefig(FIG_DIR / \"training_loss.png\", dpi=150)",
    "plt.show()",
))

cells.append(md(
    "## Evaluate on the held-out day",
    "",
    "Metrics are reported in physical units (K/s, kg/kg/s, W/m^2) after inverting the normalization, pooling "
    "over all levels and samples for the two level variables and over samples for the eight scalar surface "
    "variables -- the same grouping ClimSim's own baseline evaluation uses (see `archive/*_data.csv`), so the "
    "two are directly comparable.",
))

cells.append(code(
    "pred_level_n, pred_scalar_n = model.predict(",
    "    [test_n.level_features, test_n.aux_features], batch_size=512, verbose=0",
    ")",
    "pred_level = normalizer.inverse_transform_level_targets(pred_level_n)",
    "pred_scalar = normalizer.inverse_transform_scalar_targets(pred_scalar_n)",
    "",
    "metrics_df = evaluate(day2_samples.level_targets, pred_level, day2_samples.scalar_targets, pred_scalar)",
    "metrics_df.to_csv(METRICS_DIR / \"rnn_emulator_metrics.csv\", index=False)",
    "metrics_df",
))

cells.append(md(
    "## Comparison against ClimSim's constant and linear-regression baselines",
    "",
    "`archive/*_data.csv` holds MAE/RMSE/R2/bias for ClimSim's constant-prediction (`const`) and multiple "
    "linear regression (`mlr`) baselines, computed with ClimSim's official evaluation pipeline on ClimSim's "
    "own (much larger) scoring data split. Two caveats on comparing against them directly:",
    "",
    "- The 8 scalar surface variables (NETSW, FLWDS, PRECSC, PRECC, SOLS, SOLL, SOLSD, SOLLD) are already in "
    "shared, unweighted physical units in both this notebook and the baseline, so MAE/RMSE are directly "
    "comparable there.",
    "- `dT/dt` and `dq/dt` are **not** directly comparable: ClimSim's official pipeline reports these after "
    "an energy-unit weighting (by pressure-level thickness and grid-cell area, converted to W/m^2), while "
    "this notebook reports them in raw K/s and kg/kg/s. The five-orders-of-magnitude gap below is a unit "
    "difference, not a 100,000x accuracy improvement -- treat R2 as the meaningful cross-check for those two "
    "rows instead, since R2 is scale-invariant.",
))

cells.append(code(
    "ARCHIVE_DIR = Path(\"../archive\")",
    "",
    "name_map = {\n"
    "    \"$dT/dt$\": \"dT/dt\",\n"
    "    \"$dq/dt$\": \"dq/dt\",\n"
    "    **{v.removeprefix(\"cam_out_\"): v for v in SCALAR_TARGET_VARS},\n"
    "}",
    "",
    "def load_baseline(metric):",
    "    df = pd.read_csv(ARCHIVE_DIR / f\"{metric}_data.csv\")",
    "    df[\"variable\"] = df[\"variable\"].replace(name_map)",
    "    return df.set_index(\"variable\")[[\"const\", \"mlr\"]].rename(",
    "        columns={\"const\": f\"{metric}_const\", \"mlr\": f\"{metric}_mlr\"}",
    "    )",
    "",
    "baseline = pd.concat([load_baseline(m) for m in [\"MAE\", \"RMSE\", \"R2\", \"bias\"]], axis=1)",
    "comparison = metrics_df.set_index(\"variable\").join(baseline, how=\"left\")",
    "comparison[[\"mae\", \"MAE_const\", \"MAE_mlr\", \"rmse\", \"RMSE_const\", \"RMSE_mlr\"]]",
))

cells.append(code(
    "# Raw MAE/RMSE aren't comparable across variables here -- dT/dt is O(1e-5) K/s and NETSW is",
    "# O(10) W/m^2, so a shared linear axis makes most bars invisible. Normalizing by each",
    "# variable's own standard deviation (on the same held-out day) gives a scale-free \"fraction",
    "# of natural variability explained\" that's comparable across variables and equivalent to 1 - R2.",
    "target_std = np.concatenate(\n"
    "    [day2_samples.level_targets.std(axis=(0, 1)), day2_samples.scalar_targets.std(axis=0)]\n"
    ")",
    "nmae = metrics_df[\"mae\"].to_numpy() / target_std",
    "nrmse = metrics_df[\"rmse\"].to_numpy() / target_std",
    "",
    "fig, ax = plt.subplots(figsize=(9, 5))",
    "x = np.arange(len(metrics_df))",
    "width = 0.35",
    "ax.bar(x - width / 2, nmae, width=width, label=\"MAE / std(target)\")",
    "ax.bar(x + width / 2, nrmse, width=width, label=\"RMSE / std(target)\")",
    "ax.set_xticks(x, metrics_df[\"variable\"], rotation=45, ha=\"right\")",
    "ax.set_ylabel(\"error relative to target std (lower is better)\")",
    "ax.set_title(\"Normalized error per variable, held-out day 2\")",
    "ax.legend()",
    "",
    "fig.tight_layout()",
    "fig.savefig(FIG_DIR / \"normalized_error_per_variable.png\", dpi=150)",
    "plt.show()",
))

cells.append(md(
    "## Limitations and next steps",
    "",
    "- **Two days of data.** Training uses one day (~25k column-timesteps after the train/val split), tested "
    "on a single held-out day. ClimSim's full dataset spans years; results here show the approach works, not "
    "that it's tuned or competitive at scale.",
    "- **Minimal feature set.** Inputs are limited to temperature, humidity, surface pressure, and three "
    "surface fluxes -- the subset the original notebook used. ClimSim's full input specification also "
    "includes winds, ozone, additional moisture species, and more.",
    "- **No energy-unit weighting.** ClimSim's official leaderboard weights tendencies by pressure-level "
    "thickness and grid-cell area to put every variable in a common energy unit before scoring. Metrics here "
    "are plain per-variable MAE/RMSE/R2/bias in each variable's native physical unit, which is enough to "
    "compare against the archived baselines' orders of magnitude but not to submit to ClimSim's leaderboard.",
    "- **Architecture.** A 2-layer LSTM was a reasonable starting point given the vertical-sequence framing; "
    "it hasn't been tuned (units, depth, dropout, learning-rate schedule) or compared against non-recurrent "
    "alternatives (e.g. a plain MLP per level, or attention over levels).",
))

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
