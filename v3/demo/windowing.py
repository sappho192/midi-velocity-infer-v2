"""Windowing, window feature extraction, and reconstruction for the demo.

Ported from mvi_v3.data.windowing, mvi_v3.data.preset_features,
and mvi_v3.eval.reconstruct. Self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from features import CONTINUOUS_FEATURES, Stats, normalize_feature
from midi_io import NoteEvent

WINDOW_SIZE = 256
STRIDE = 128
N_REGISTER_BUCKETS = 4
N_CONTINUOUS = 6


@dataclass(slots=True)
class Window:
    start_note_index: int
    true_length: int
    pitch: np.ndarray          # [256] int64
    register_bucket: np.ndarray  # [256] int64
    continuous: np.ndarray     # [256, 6] float32 (z-normalized)
    padding_mask: np.ndarray   # [256] bool
    global_note_indices: np.ndarray  # [256] int64


def build_windows(events: list[NoteEvent], stats: Stats) -> list[Window]:
    """Build sliding windows from a single piece's note events."""
    windows: list[Window] = []
    n = len(events)

    starts = list(range(0, n, STRIDE))
    for start in starts:
        chunk = events[start : start + WINDOW_SIZE]
        if not chunk:
            continue

        true_length = len(chunk)
        pad_count = WINDOW_SIZE - true_length

        # Arrays for valid notes
        pitches = [e.pitch for e in chunk]
        registers = [e.register_bucket for e in chunk]
        indices = [e.note_index for e in chunk]
        cont_rows = [
            [normalize_feature(getattr(e, name), name, stats) for name in CONTINUOUS_FEATURES]
            for e in chunk
        ]

        # Pad
        pitches.extend([0] * pad_count)
        registers.extend([0] * pad_count)
        indices.extend([-1] * pad_count)
        cont_rows.extend([[0.0] * N_CONTINUOUS] * pad_count)

        windows.append(Window(
            start_note_index=start,
            true_length=true_length,
            pitch=np.array(pitches, dtype=np.int64),
            register_bucket=np.array(registers, dtype=np.int64),
            continuous=np.array(cont_rows, dtype=np.float32),
            padding_mask=np.array(
                [False] * true_length + [True] * pad_count, dtype=bool
            ),
            global_note_indices=np.array(indices, dtype=np.int64),
        ))

        if start + WINDOW_SIZE >= n:
            break

    return windows


def extract_window_features(window: Window) -> np.ndarray:
    """Extract 21-dim aggregate feature vector for control MLP input.

    Groups: pitch stats (4), continuous stats (12), fill ratio (1),
    register distribution (4).
    """
    valid = ~window.padding_mask
    n_valid = int(valid.sum())

    # Pitch stats (4)
    if n_valid > 0:
        vp = window.pitch[valid].astype(np.float32) / 127.0
        pitch_feats = [vp.mean(), vp.std() if n_valid > 1 else 0.0, vp.min(), vp.max()]
    else:
        pitch_feats = [0.0, 0.0, 0.0, 0.0]

    # Continuous stats (12): mean + std for each of 6 features
    if n_valid > 0:
        vc = window.continuous[valid]
        cont_mean = vc.mean(axis=0)
        cont_std = vc.std(axis=0) if n_valid > 1 else np.zeros(N_CONTINUOUS, dtype=np.float32)
    else:
        cont_mean = np.zeros(N_CONTINUOUS, dtype=np.float32)
        cont_std = np.zeros(N_CONTINUOUS, dtype=np.float32)

    # Fill ratio (1)
    fill = n_valid / WINDOW_SIZE

    # Register distribution (4)
    if n_valid > 0:
        vr = window.register_bucket[valid]
        reg_dist = np.array([
            float((vr == b).sum()) / n_valid for b in range(N_REGISTER_BUCKETS)
        ], dtype=np.float32)
    else:
        reg_dist = np.zeros(N_REGISTER_BUCKETS, dtype=np.float32)

    return np.concatenate([
        np.array(pitch_feats, dtype=np.float32),
        cont_mean.astype(np.float32),
        cont_std.astype(np.float32),
        np.array([fill], dtype=np.float32),
        reg_dist,
    ])


def reconstruct_center_priority(
    windows: list[Window],
    predictions: list[np.ndarray],
) -> list[float]:
    """Reconstruct note-level predictions from overlapping windows.

    For each note, picks the prediction from the window where the note
    is closest to center. Returns a flat list indexed by global note index.
    """
    best: dict[int, tuple[float, int, float]] = {}

    for window_order, (window, pred) in enumerate(zip(windows, predictions)):
        center = (len(pred) - 1) / 2.0
        for local_idx, note_idx in enumerate(window.global_note_indices.tolist()):
            if note_idx < 0 or window.padding_mask[local_idx]:
                continue
            score = (abs(local_idx - center), window_order)
            current = best.get(note_idx)
            if current is None or score < current[:2]:
                best[note_idx] = (score[0], score[1], float(pred[local_idx]))

    return [best[idx][2] for idx in sorted(best)]
