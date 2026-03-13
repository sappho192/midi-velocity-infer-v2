from collections.abc import Sequence

import numpy as np


def compute_piece_metrics(
    prediction: Sequence[float],
    target: Sequence[float],
) -> dict[str, float]:
    """Compute per-piece evaluation metrics on raw 0-127 velocity scale.

    Metric set follows He et al. 2025:
    - MAE, MSE: standard regression metrics
    - SD_velo: std of predicted velocities (expressiveness indicator)
    - SD_ratio: pred_std / true_std (1.0 = perfect dynamic range)
    - SD_ae: std of absolute errors (error consistency)
    - CC: Pearson correlation coefficient
    - Recall_10: proportion within ±12.7 (10% of 127) — He2025 primary
    - Recall_5: proportion within ±6.4 (5% of 127) — He2025 strict
    """
    pred = np.asarray(prediction, dtype=np.float32)
    true = np.asarray(target, dtype=np.float32)
    n = len(pred)

    # Regression
    abs_err = np.abs(pred - true)
    mae = float(np.mean(abs_err))
    mse = float(np.mean((pred - true) ** 2))

    # Dynamic range / expressiveness
    pred_std = float(np.std(pred))
    true_std = float(np.std(true))
    sd_ratio = pred_std / true_std if true_std > 0 else 0.0

    # Error consistency (He2025)
    sd_ae = float(np.std(abs_err))

    # Correlation
    if pred_std > 0 and true_std > 0:
        cc = float(np.corrcoef(pred, true)[0, 1])
    else:
        cc = 0.0

    # Recall (He2025): proportion of notes within tolerance
    recall_10 = float(np.mean(abs_err < 12.7))   # 10% of 127
    recall_5 = float(np.mean(abs_err < 6.4))      # 5% of 127

    return {
        "mae": mae,
        "mse": mse,
        "pred_std": pred_std,
        "true_std": true_std,
        "sd_ratio": sd_ratio,
        "sd_ae": sd_ae,
        "cc": cc,
        "recall_10": recall_10,
        "recall_5": recall_5,
        "n_notes": n,
    }


# Keep backward-compatible alias
def compute_regression_metrics(
    prediction: Sequence[float], target: Sequence[float],
) -> dict[str, float]:
    m = compute_piece_metrics(prediction, target)
    return {
        "mae_raw": m["mae"],
        "mae_normalized": m["mae"] / 127.0,
        "mse": m["mse"],
        "pred_std": m["pred_std"],
        "true_std": m["true_std"],
    }


def aggregate_metrics(
    piece_metrics: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute note-weighted and macro (piece-averaged) aggregate metrics."""
    keys = ["mae", "mse", "pred_std", "true_std", "sd_ratio", "sd_ae",
            "cc", "recall_10", "recall_5"]

    # Macro average (mean across pieces)
    n_pieces = len(piece_metrics)
    macro = {}
    for k in keys:
        vals = [m[k] for m in piece_metrics.values()]
        macro[f"macro_{k}"] = float(np.mean(vals))

    # Note-weighted average
    total_notes = sum(m["n_notes"] for m in piece_metrics.values())
    weighted = {}
    for k in keys:
        weighted[f"weighted_{k}"] = float(
            sum(m[k] * m["n_notes"] for m in piece_metrics.values()) / max(total_notes, 1)
        )

    return {
        "n_pieces": n_pieces,
        "n_notes": total_notes,
        **macro,
        **weighted,
    }
