# Archive

Earlier drafts, kept for reference but not part of the verified, reproducible pipeline in this repo.

- `legacy_quickstart_RNN_original.ipynb` — the first version of the RNN notebook, before the rewrite in `notebooks/`. Sample definition was one batch per day (72 timesteps × 47616 flattened features), which does not match how ClimSim's task is normally framed. Kept to show what changed and why.
- `legacy_svr_lr_baseline_from_ClimSim_demo.ipynb`, `legacy_cloud_notebook_from_ClimSim_demo.ipynb` — adaptations of ClimSim's official baseline/demo notebooks. They depend on ClimSim's `data_utils.py`, grid-info file, and precomputed normalization files, none of which are in `data/`, so they are not verified to run here.
- `legacy_huggingface_data_loader_draft.ipynb` — an unfinished draft for streaming ClimSim's full dataset from Hugging Face into local `.npy` splits.
- `legacy_r_quickstart_functions.R` — an R port of the ClimSim quickstart utilities.
- `MAE_data.csv`, `RMSE_data.csv`, `R2_data.csv`, `bias_data.csv` — metrics from the constant-prediction and linear-regression baselines in the legacy SVR/LR notebook, kept as a reference point for the new metrics in `results/metrics/`.
