import numpy as np
from sklearn.linear_model import LinearRegression

from .preprocessing import Normalizer, SampleBatch


def predict_constant(train: SampleBatch, test: SampleBatch) -> tuple[np.ndarray, np.ndarray]:
    level_mean = train.level_targets.mean(axis=0)
    scalar_mean = train.scalar_targets.mean(axis=0)
    test_count = test.level_features.shape[0]
    level_prediction = np.broadcast_to(level_mean, (test_count,) + level_mean.shape).copy()
    scalar_prediction = np.broadcast_to(scalar_mean, (test_count, scalar_mean.shape[0])).copy()
    return level_prediction, scalar_prediction


def _flatten_inputs(normalizer: Normalizer, batch: SampleBatch) -> np.ndarray:
    normalized = normalizer.transform(batch)
    flat_levels = normalized.level_features.reshape(len(normalized.level_features), -1)
    return np.concatenate([flat_levels, normalized.aux_features], axis=1).astype(np.float64)


def predict_linear_regression(train: SampleBatch, test: SampleBatch) -> tuple[np.ndarray, np.ndarray]:
    normalizer = Normalizer().fit(train)
    train_inputs = _flatten_inputs(normalizer, train)
    train_targets = np.concatenate(
        [train.level_targets.reshape(len(train_inputs), -1), train.scalar_targets], axis=1
    ).astype(np.float64)

    prediction = LinearRegression().fit(train_inputs, train_targets).predict(_flatten_inputs(normalizer, test))

    level_shape = train.level_targets.shape[1:]
    level_width = int(np.prod(level_shape))
    level_prediction = prediction[:, :level_width].reshape((len(prediction),) + level_shape)
    scalar_prediction = prediction[:, level_width:]
    return level_prediction, scalar_prediction
