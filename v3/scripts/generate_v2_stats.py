"""Regenerate v2 normalization statistics from MAESTRO train split.

Usage:
    uv run python scripts/generate_v2_stats.py
"""

import json
from pathlib import Path

import numpy as np


def main():
    train_dir = Path("/home/tikim/dataset/maestro/maestro-raw/maestro-midi/train")
    csv_files = sorted(train_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} train CSV files")

    all_time_diff = []
    all_note_num_diff = []
    all_length = []

    for csv_path in csv_files:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
        # columns: time, time_diff, note_num, length, velocity
        time_diff = data[:, 1]
        note_num = data[:, 2]
        length = data[:, 3]

        # Derive note_num_diff: diff[0]=0, diff[i] = note_num[i] - note_num[i-1]
        note_num_diff = np.zeros(len(note_num), dtype=np.float32)
        note_num_diff[1:] = note_num[1:] - note_num[:-1]

        all_time_diff.append(time_diff)
        all_note_num_diff.append(note_num_diff)
        all_length.append(length)

    all_time_diff = np.concatenate(all_time_diff)
    all_note_num_diff = np.concatenate(all_note_num_diff)
    all_length = np.concatenate(all_length)

    stats = {
        "train_time_diff_min": float(np.min(all_time_diff)),
        "train_time_diff_max": float(np.max(all_time_diff)),
        "note_num_min": 0,
        "note_num_max": 127,
        "note_num_diff_min": float(np.min(all_note_num_diff)),
        "note_num_diff_max": float(np.max(all_note_num_diff)),
        "length_min": float(np.min(all_length)),
        "length_max": float(np.max(all_length)),
        "velocity_min": 0,
        "velocity_max": 127,
    }

    output_dir = Path("runs/v2_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "v2_stats.json"

    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to {output_path}")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
