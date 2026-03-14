from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from mvi_v3.config import BaselineConfig

from .events import DatasetStats, NoteEvent, PredictionRecord, WindowRecord
from .normalize import normalize_feature, normalize_velocity


def _compute_oracle_controls(raw_velocities: np.ndarray) -> np.ndarray:
    """Compute oracle control parameters from raw velocities (0-127).

    Returns [expressiveness, dynamics_center] as raw (un-normalized) values.
    - expressiveness = std(velocity)
    - dynamics_center = mean(velocity)
    """
    if len(raw_velocities) < 2:
        return np.array([0.0, float(raw_velocities.mean()) if len(raw_velocities) else 64.0], dtype=np.float32)
    return np.array([
        float(raw_velocities.std()),
        float(raw_velocities.mean()),
    ], dtype=np.float32)


def fit_oracle_stats(windows: Sequence[WindowRecord]) -> tuple[list[float], list[float]]:
    """Fit min-max normalization stats for oracle controls from training windows.

    Returns (mins, maxs) lists, one value per control dimension.
    """
    all_controls = np.stack([w.oracle_controls for w in windows if w.oracle_controls is not None])
    mins = all_controls.min(axis=0).tolist()
    maxs = all_controls.max(axis=0).tolist()
    # Avoid division by zero
    for i in range(len(mins)):
        if maxs[i] - mins[i] < 1e-8:
            maxs[i] = mins[i] + 1.0
    return mins, maxs


def normalize_oracle_controls(
    windows: Sequence[WindowRecord],
    mins: list[float],
    maxs: list[float],
) -> None:
    """Normalize oracle controls to [0, 1] in-place using min-max scaling."""
    mins_arr = np.array(mins, dtype=np.float32)
    maxs_arr = np.array(maxs, dtype=np.float32)
    ranges = maxs_arr - mins_arr
    for window in windows:
        if window.oracle_controls is not None:
            window.oracle_controls = np.clip(
                (window.oracle_controls - mins_arr) / ranges, 0.0, 1.0
            ).astype(np.float32)


def build_windows(
    pieces: Sequence[Sequence[NoteEvent]],
    stats: DatasetStats,
    config: BaselineConfig,
) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    feature_names = config.continuous_features
    enable_controls = config.enable_controls
    for piece in pieces:
        if not piece:
            continue
        piece_id = piece[0].piece_id
        starts = list(range(0, len(piece), config.stride))
        for start in starts:
            chunk = list(piece[start : start + config.window_size])
            if not chunk:
                continue
            true_length = len(chunk)
            if true_length < config.window_size:
                pad_count = config.window_size - true_length
                for pad_index in range(pad_count):
                    chunk.append(
                        NoteEvent(
                            piece_id=piece_id,
                            note_index=start + true_length + pad_index,
                            pitch=0,
                            onset_sec=0.0,
                            offset_sec=0.0,
                            velocity=None,
                        )
                    )

            pitch = np.array([event.pitch for event in chunk], dtype=np.int64)
            register = np.array([event.register_bucket for event in chunk], dtype=np.int64)
            continuous = np.array(
                [
                    [normalize_feature(float(getattr(event, name)), name, stats) for name in feature_names]
                    for event in chunk
                ],
                dtype=np.float32,
            )
            targets = np.array([normalize_velocity(event.velocity, stats) for event in chunk], dtype=np.float32)
            padding_mask = np.array([False] * true_length + [True] * (config.window_size - true_length), dtype=bool)
            global_indices = np.array(
                [event.note_index if idx < true_length else -1 for idx, event in enumerate(chunk)],
                dtype=np.int64,
            )

            # Oracle controls from raw velocities of valid notes
            oracle = None
            if enable_controls:
                raw_velocities = np.array(
                    [event.velocity for event in chunk[:true_length] if event.velocity is not None],
                    dtype=np.float32,
                )
                oracle = _compute_oracle_controls(raw_velocities)

            windows.append(
                WindowRecord(
                    piece_id=piece_id,
                    start_note_index=start,
                    true_length=true_length,
                    pitch=pitch,
                    register_bucket=register,
                    continuous=continuous,
                    target_velocity=targets,
                    padding_mask=padding_mask,
                    global_note_indices=global_indices,
                    oracle_controls=oracle,
                )
            )
            if start + config.window_size >= len(piece):
                break
    return windows


def reconstruct_piece_predictions(
    windows: Sequence[WindowRecord],
    predictions: Sequence[np.ndarray],
) -> dict[str, list[float]]:
    ranked: dict[str, dict[int, tuple[float, int, float]]] = defaultdict(dict)
    for window_order, (window, pred) in enumerate(zip(windows, predictions, strict=True)):
        center = (len(pred) - 1) / 2.0
        for local_index, global_index in enumerate(window.global_note_indices.tolist()):
            if global_index < 0 or window.padding_mask[local_index]:
                continue
            distance = abs(local_index - center)
            current = ranked[window.piece_id].get(global_index)
            candidate = (distance, window_order, float(pred[local_index]))
            if current is None or candidate[:2] < current[:2]:
                ranked[window.piece_id][global_index] = candidate

    reconstructed: dict[str, list[float]] = {}
    for piece_id, per_note in ranked.items():
        reconstructed[piece_id] = [per_note[index][2] for index in sorted(per_note)]
    return reconstructed
