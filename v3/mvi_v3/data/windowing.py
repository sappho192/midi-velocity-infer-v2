from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from mvi_v3.config import BaselineConfig

from .events import NoteEvent, PredictionRecord, WindowRecord
from .normalize import normalize_feature, normalize_velocity
from .events import DatasetStats


def build_windows(
    pieces: Sequence[Sequence[NoteEvent]],
    stats: DatasetStats,
    config: BaselineConfig,
) -> list[WindowRecord]:
    windows: list[WindowRecord] = []
    feature_names = config.continuous_features
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
