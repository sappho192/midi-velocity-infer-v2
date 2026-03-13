from collections import defaultdict
from collections.abc import Iterable, Sequence
from math import sqrt

from mvi_v3.config import BaselineConfig

from .events import DatasetStats, NoteEvent


def fit_dataset_stats(pieces: Sequence[Sequence[NoteEvent]], config: BaselineConfig) -> DatasetStats:
    values: dict[str, list[float]] = defaultdict(list)
    for piece in pieces:
        for event in piece:
            for feature_name in config.continuous_features:
                values[feature_name].append(float(getattr(event, feature_name)))

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for name in config.continuous_features:
        samples = values[name]
        mean = sum(samples) / max(len(samples), 1)
        variance = sum((sample - mean) ** 2 for sample in samples) / max(len(samples), 1)
        means[name] = mean
        stds[name] = sqrt(variance) or 1.0
    return DatasetStats(feature_means=means, feature_stds=stds, velocity_min=0.0, velocity_max=127.0)


def normalize_feature(value: float, name: str, stats: DatasetStats) -> float:
    return (value - stats.feature_means[name]) / stats.feature_stds[name]


def normalize_velocity(value: int | None, stats: DatasetStats) -> float:
    if value is None:
        return 0.0
    return (float(value) - stats.velocity_min) / (stats.velocity_max - stats.velocity_min)


def denormalize_velocity(value: float, stats: DatasetStats) -> float:
    return value * (stats.velocity_max - stats.velocity_min) + stats.velocity_min
