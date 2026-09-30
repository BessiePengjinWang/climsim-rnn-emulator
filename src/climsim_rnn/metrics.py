import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import LEVEL_TARGET_VARS, SCALAR_TARGET_VARS


def _row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    return {
        "variable": name,
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def evaluate(
    level_true: np.ndarray,
    level_pred: np.ndarray,
    scalar_true: np.ndarray,
    scalar_pred: np.ndarray,
) -> pd.DataFrame:
    """Per-variable MAE/RMSE/R2/bias in physical units.

    Level variables (dT/dt, dq/dt) are pooled across all levels and samples;
    scalar surface variables are pooled across samples only. This mirrors the
    variable groupings ClimSim's own baseline evaluation reports (see
    archive/*_data.csv). The scalar variables are in the same native units in
    both places, so MAE/RMSE are directly comparable; dT/dt and dq/dt are not,
    since ClimSim's official pipeline reports those after an energy-unit
    (pressure/area-weighted, W/m^2) conversion that isn't reproduced here --
    R2 is the safer cross-check for those two.
    """
    rows = [
        _row(name, level_true[..., i], level_pred[..., i]) for i, name in enumerate(LEVEL_TARGET_VARS)
    ]
    rows += [
        _row(name, scalar_true[:, i], scalar_pred[:, i]) for i, name in enumerate(SCALAR_TARGET_VARS)
    ]
    return pd.DataFrame(rows)
