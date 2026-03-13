from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class NoteEvent:
    piece_id: str
    note_index: int
    pitch: int
    onset_sec: float
    offset_sec: float
    velocity: int | None
    duration_sec: float = 0.0
    ioi_next_sec: float = 0.0
    delta_pitch_prev: float = 0.0
    delta_pitch_next: float = 0.0
    same_onset_chord_size: float = 1.0
    local_note_density: float = 0.0
    register_bucket: int = 0


@dataclass(slots=True)
class WindowRecord:
    piece_id: str
    start_note_index: int
    true_length: int
    pitch: np.ndarray
    register_bucket: np.ndarray
    continuous: np.ndarray
    target_velocity: np.ndarray
    padding_mask: np.ndarray
    global_note_indices: np.ndarray


@dataclass(slots=True)
class DatasetStats:
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)
    velocity_min: float = 0.0
    velocity_max: float = 127.0


@dataclass(slots=True)
class PredictionRecord:
    piece_id: str
    global_note_index: int
    local_index: int
    window_start_index: int
    prediction_norm: float
