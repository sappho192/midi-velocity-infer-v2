import math
from collections.abc import Sequence

import numpy as np


def compute_regression_metrics(prediction: Sequence[float], target: Sequence[float]) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    normalized_mae = float(mae / 127.0)
    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    return {
        "mae_raw": mae,
        "mae_normalized": normalized_mae,
        "mse": mse,
        "pred_std": pred_std,
        "true_std": true_std,
    }
