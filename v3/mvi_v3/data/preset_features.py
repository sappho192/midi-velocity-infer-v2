"""Window-level aggregate feature extraction for preset classification."""

from __future__ import annotations

import numpy as np

from .events import WindowRecord


# Register bucket boundaries match config default: (48, 60, 72) → 4 buckets
_N_REGISTER_BUCKETS = 4
# continuous_features order: duration_sec, ioi_next_sec, delta_pitch_prev,
# delta_pitch_next, local_note_density, same_onset_chord_size
_N_CONTINUOUS = 6


def extract_window_features(window: WindowRecord, window_size: int = 256) -> np.ndarray:
    """Extract 21-dimensional aggregate feature vector from a window.

    Groups:
        Pitch stats (4): mean, std, min, max (÷127 normalized)
        Continuous stats (12): 6 features × (mean, std) across valid notes
        Fill ratio (1): true_length / window_size
        Register distribution (4): proportion in each of 4 buckets

    Args:
        window: A WindowRecord with pitch, continuous, padding_mask, etc.
        window_size: Window size for fill ratio computation.

    Returns:
        np.ndarray of shape (21,), dtype float32.
    """
    valid_mask = ~window.padding_mask
    n_valid = int(valid_mask.sum())

    # --- Pitch stats (4) ---
    if n_valid > 0:
        valid_pitches = window.pitch[valid_mask].astype(np.float32) / 127.0
        pitch_mean = valid_pitches.mean()
        pitch_std = valid_pitches.std() if n_valid > 1 else 0.0
        pitch_min = valid_pitches.min()
        pitch_max = valid_pitches.max()
    else:
        pitch_mean = pitch_std = pitch_min = pitch_max = 0.0

    # --- Continuous stats (12) ---
    # continuous shape: [window_size, n_continuous], already z-normalized
    if n_valid > 0:
        valid_cont = window.continuous[valid_mask]  # [n_valid, 6]
        cont_mean = valid_cont.mean(axis=0)  # [6]
        cont_std = valid_cont.std(axis=0) if n_valid > 1 else np.zeros(_N_CONTINUOUS, dtype=np.float32)
    else:
        cont_mean = np.zeros(_N_CONTINUOUS, dtype=np.float32)
        cont_std = np.zeros(_N_CONTINUOUS, dtype=np.float32)

    # --- Fill ratio (1) ---
    fill_ratio = n_valid / window_size

    # --- Register distribution (4) ---
    if n_valid > 0:
        valid_registers = window.register_bucket[valid_mask]
        reg_dist = np.zeros(_N_REGISTER_BUCKETS, dtype=np.float32)
        for bucket in range(_N_REGISTER_BUCKETS):
            reg_dist[bucket] = (valid_registers == bucket).sum() / n_valid
    else:
        reg_dist = np.zeros(_N_REGISTER_BUCKETS, dtype=np.float32)

    # Assemble 21-dim vector
    features = np.concatenate([
        np.array([pitch_mean, pitch_std, pitch_min, pitch_max], dtype=np.float32),
        cont_mean.astype(np.float32),
        cont_std.astype(np.float32),
        np.array([fill_ratio], dtype=np.float32),
        reg_dist,
    ])
    return features
