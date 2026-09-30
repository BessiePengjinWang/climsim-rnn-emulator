from dataclasses import dataclass, replace

import numpy as np
import xarray as xr

LEVEL_INPUT_VARS = ["state_t", "state_q0001"]
AUX_INPUT_VARS = ["state_ps", "pbuf_SOLIN", "pbuf_LHFLX", "pbuf_SHFLX"]
# Tendencies are derived as (output - input) / dt; these are display labels, not raw variable names.
LEVEL_TARGET_VARS = ["dT/dt", "dq/dt"]
SCALAR_TARGET_VARS = [
    "cam_out_NETSW",
    "cam_out_FLWDS",
    "cam_out_PRECSC",
    "cam_out_PRECC",
    "cam_out_SOLS",
    "cam_out_SOLL",
    "cam_out_SOLSD",
    "cam_out_SOLLD",
]

_LEVEL_STATE_VARS = ["state_t", "state_q0001"]  # source vars for the tendency targets above
EPS = 1e-8


@dataclass
class SampleBatch:
    level_features: np.ndarray  # (N, levels, len(LEVEL_INPUT_VARS))
    aux_features: np.ndarray  # (N, len(AUX_INPUT_VARS))
    level_targets: np.ndarray  # (N, levels, len(LEVEL_TARGET_VARS))
    scalar_targets: np.ndarray  # (N, len(SCALAR_TARGET_VARS))


def build_samples(ds_in: xr.Dataset, ds_out: xr.Dataset, dt: float = 1200.0) -> SampleBatch:
    """Flatten (time, ncol, lev) ClimSim snapshots into one sample per grid column per timestep.

    Each sample's feature/target sequence runs over the 60 vertical levels, which is
    the physically meaningful sequence axis for an RNN here (adjacent levels are
    coupled by convection and vertical mixing) -- as opposed to treating whole days
    or whole columns-at-once as a single sequence.
    """
    time, ncol, levels = ds_in.sizes["time"], ds_in.sizes["ncol"], ds_in.sizes["lev"]
    n = time * ncol

    level_features = np.stack(
        [ds_in[var].transpose("time", "ncol", "lev").values.reshape(n, levels) for var in LEVEL_INPUT_VARS],
        axis=-1,
    )

    aux_features = np.stack(
        [ds_in[var].transpose("time", "ncol").values.reshape(n) for var in AUX_INPUT_VARS],
        axis=-1,
    )

    level_targets = np.stack(
        [
            ((ds_out[var] - ds_in[var]) / dt).transpose("time", "ncol", "lev").values.reshape(n, levels)
            for var in _LEVEL_STATE_VARS
        ],
        axis=-1,
    )

    scalar_targets = np.stack(
        [ds_out[var].transpose("time", "ncol").values.reshape(n) for var in SCALAR_TARGET_VARS],
        axis=-1,
    )

    return SampleBatch(
        level_features.astype(np.float32),
        aux_features.astype(np.float32),
        level_targets.astype(np.float32),
        scalar_targets.astype(np.float32),
    )


@dataclass
class _Stats:
    mean: np.ndarray
    std: np.ndarray


class Normalizer:
    """Fits mean/std on a training SampleBatch, applies it to any batch.

    Standardizing per level (not globally) matters here: temperature and humidity
    have very different scales and variances near the surface vs. near the top of
    the atmosphere. Std is floored at EPS so levels with near-zero variance (e.g.
    humidity at the top of the atmosphere) don't produce inf/NaN after division --
    this is what caused the manual NaN patch-up in the original notebook.
    """

    def fit(self, batch: SampleBatch) -> "Normalizer":
        self._level_input = _Stats(batch.level_features.mean(axis=0), batch.level_features.std(axis=0))
        self._aux_input = _Stats(batch.aux_features.mean(axis=0), batch.aux_features.std(axis=0))
        self._level_target = _Stats(batch.level_targets.mean(axis=0), batch.level_targets.std(axis=0))
        self._scalar_target = _Stats(batch.scalar_targets.mean(axis=0), batch.scalar_targets.std(axis=0))
        return self

    def transform(self, batch: SampleBatch) -> SampleBatch:
        return replace(
            batch,
            level_features=self._standardize(batch.level_features, self._level_input),
            aux_features=self._standardize(batch.aux_features, self._aux_input),
            level_targets=self._standardize(batch.level_targets, self._level_target),
            scalar_targets=self._standardize(batch.scalar_targets, self._scalar_target),
        )

    def fit_transform(self, batch: SampleBatch) -> SampleBatch:
        return self.fit(batch).transform(batch)

    def inverse_transform_level_targets(self, arr: np.ndarray) -> np.ndarray:
        return arr * self._level_target.std + self._level_target.mean

    def inverse_transform_scalar_targets(self, arr: np.ndarray) -> np.ndarray:
        return arr * self._scalar_target.std + self._scalar_target.mean

    @staticmethod
    def _standardize(arr: np.ndarray, stats: _Stats) -> np.ndarray:
        return ((arr - stats.mean) / np.maximum(stats.std, EPS)).astype(np.float32)
