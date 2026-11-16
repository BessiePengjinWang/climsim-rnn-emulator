import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import LEVEL_TARGET_VARS, SCALAR_TARGET_VARS


def _row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch for {name}: truth {y_true.shape} vs prediction {y_pred.shape}")
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

    Level variables (dT/dt, dq/dt) are pooled across all levels and samples, so
    their R2 is measured against the variance of the pooled values and is
    dominated by the high-variance levels; the mean of per-level R2 can differ
    noticeably. Scalar surface variables are pooled across samples only. Bias is
    mean(prediction - truth), so a positive value means over-prediction.

    Only some of ClimSim's published baseline metrics are comparable to these;
    see reference/climsim_published_baselines/README.md for which.
    """
    rows = [
        _row(name, level_true[..., i], level_pred[..., i]) for i, name in enumerate(LEVEL_TARGET_VARS)
    ]
    rows += [
        _row(name, scalar_true[:, i], scalar_pred[:, i]) for i, name in enumerate(SCALAR_TARGET_VARS)
    ]
    return pd.DataFrame(rows)
