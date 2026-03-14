"""Evaluate v2 ONNX model on MAESTRO test set using He2025 metrics.

Usage:
    uv run python scripts/eval_v2_on_test.py

Requires: onnxruntime (install via `uv pip install onnxruntime`)
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime

from mvi_v3.eval.metrics import aggregate_metrics, compute_piece_metrics

SAMPLE_LENGTH = 4
FEATURE_NUM = 5  # time_diff, note_num, length, note_num_diff, low_octave


def load_v2_stats(stats_path: str | Path) -> dict:
    with open(stats_path) as f:
        return json.load(f)


def load_piece(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV and return (features[N, 5], velocity[N])."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
    # columns: time, time_diff, note_num, length, velocity
    time_diff = data[:, 1]
    note_num = data[:, 2]
    length = data[:, 3]
    velocity = data[:, 4]

    # Derive note_num_diff
    note_num_diff = np.zeros(len(note_num), dtype=np.float32)
    note_num_diff[1:] = note_num[1:] - note_num[:-1]

    # Derive low_octave
    low_octave = (note_num < 72).astype(np.float32)

    # Stack features: time_diff, note_num, length, note_num_diff, low_octave
    features = np.column_stack([time_diff, note_num, length, note_num_diff, low_octave])
    return features, velocity


def normalize_features(features: np.ndarray, stats: dict) -> np.ndarray:
    """Min-max normalize features (low_octave is NOT normalized)."""
    normed = features.copy()
    # time_diff
    normed[:, 0] = (normed[:, 0] - stats["train_time_diff_min"]) / (
        stats["train_time_diff_max"] - stats["train_time_diff_min"]
    )
    # note_num
    normed[:, 1] = (normed[:, 1] - stats["note_num_min"]) / (
        stats["note_num_max"] - stats["note_num_min"]
    )
    # length
    normed[:, 2] = (normed[:, 2] - stats["length_min"]) / (
        stats["length_max"] - stats["length_min"]
    )
    # note_num_diff
    normed[:, 3] = (normed[:, 3] - stats["note_num_diff_min"]) / (
        stats["note_num_diff_max"] - stats["note_num_diff_min"]
    )
    # low_octave (col 4) is not normalized
    return normed


def make_windows(data: np.ndarray, window_size: int) -> list[np.ndarray]:
    """Non-overlapping windows with zero-padding for the last window."""
    windows = []
    for i in range(0, len(data), window_size):
        chunk = data[i : i + window_size]
        if len(chunk) < window_size:
            pad = np.zeros((window_size - len(chunk), chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])
        windows.append(chunk)
    return windows


def main():
    # Use existing ONNX model and stats from demo repo
    demo_models = Path("/home/tikim/repo/midi-velocity-infer-demo/proj/mvi2demo/Tools/models")
    model_path = demo_models / "mvi-v2-2023-07-20_13-00_56-h4-e5-mse_cosine_loss-alpha0.15-m0.60-LSTM-luong_attention-MAESTRO.onnx"
    stats_path = demo_models / "dataset32-MAESTRO-len4.json"
    test_dir = Path("/home/tikim/dataset/maestro/maestro-raw/maestro-midi/test")
    output_dir = Path("runs/v2_test")

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not stats_path.exists():
        raise FileNotFoundError(f"Stats not found: {stats_path}. Run generate_v2_stats.py first.")

    stats = load_v2_stats(stats_path)
    print(f"Loaded stats from {stats_path}")

    session = onnxruntime.InferenceSession(str(model_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Loaded ONNX model from {model_path}")

    csv_files = sorted(test_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} test pieces")

    piece_metrics = {}
    for csv_path in csv_files:
        piece_id = csv_path.stem
        features, velocity = load_piece(csv_path)
        n_notes = len(velocity)

        normed = normalize_features(features, stats)
        windows = make_windows(normed, SAMPLE_LENGTH)

        # Stack windows into batch: [n_windows, window_size, features]
        batch = np.array(windows, dtype=np.float32)

        # Run inference
        output = session.run([output_name], {input_name: batch})
        predictions = np.array(output).reshape(-1)

        # Trim to piece length (remove padding from last window)
        predictions = predictions[:n_notes]

        # Denormalize: pred * 127, round, clip
        predictions = predictions * stats["velocity_max"]
        predictions = np.round(predictions)
        predictions = np.clip(predictions, 0, stats["velocity_max"])

        metrics = compute_piece_metrics(predictions, velocity)
        piece_metrics[piece_id] = metrics

    agg = aggregate_metrics(piece_metrics)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "aggregate": agg,
        "per_piece": piece_metrics,
    }
    output_path = output_dir / "metrics.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    print(f"n_pieces: {agg['n_pieces']}, n_notes: {agg['n_notes']}")
    print(f"  MAE:        {agg['weighted_mae']:.2f}  (macro {agg['macro_mae']:.2f})")
    print(f"  MSE:        {agg['weighted_mse']:.2f}  (macro {agg['macro_mse']:.2f})")
    print(f"  SD_ratio:   {agg['weighted_sd_ratio']:.1%}")
    print(f"  SD_ae:      {agg['weighted_sd_ae']:.2f}")
    print(f"  CC:         {agg['weighted_cc']:.4f}")
    print(f"  Recall(10%): {agg['weighted_recall_10']:.1%}")
    print(f"  Recall(5%):  {agg['weighted_recall_5']:.1%}")


if __name__ == "__main__":
    main()
