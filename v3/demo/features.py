"""Feature computation and normalization for the demo.

Ported from mvi_v3.data.features and mvi_v3.data.normalize.
Self-contained, no mvi_v3 imports needed.
"""

from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path

from midi_io import NoteEvent


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Stats:
    feature_means: dict[str, float] = field(default_factory=dict)
    feature_stds: dict[str, float] = field(default_factory=dict)
    velocity_min: float = 0.0
    velocity_max: float = 127.0
    oracle_mins: list[float] = field(default_factory=list)
    oracle_maxs: list[float] = field(default_factory=list)


def load_stats(path: str | Path) -> Stats:
    with open(path) as f:
        d = json.load(f)
    return Stats(
        feature_means=d["feature_means"],
        feature_stds=d["feature_stds"],
        velocity_min=d.get("velocity_min", 0.0),
        velocity_max=d.get("velocity_max", 127.0),
        oracle_mins=d.get("oracle_mins", []),
        oracle_maxs=d.get("oracle_maxs", []),
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_feature(value: float, name: str, stats: Stats) -> float:
    return (value - stats.feature_means[name]) / stats.feature_stds[name]


def normalize_velocity(value: int | None, stats: Stats) -> float:
    if value is None:
        return 0.0
    return (float(value) - stats.velocity_min) / (stats.velocity_max - stats.velocity_min)


def denormalize_velocity(value: float, stats: Stats) -> float:
    return value * (stats.velocity_max - stats.velocity_min) + stats.velocity_min


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGISTER_BOUNDARIES = (48, 60, 72)
ONSET_TOLERANCE_SEC = 0.03
LOCAL_DENSITY_NEIGHBOR_K = 8

# The models were trained with time_scale=1.0 on CSV data where time values
# are in centiseconds (raw tick-like units). pretty_midi returns real seconds.
# To match the training scale: multiply real seconds by 100.
TIME_SCALE = 100.0

CONTINUOUS_FEATURES = (
    "duration_sec",
    "ioi_next_sec",
    "delta_pitch_prev",
    "delta_pitch_next",
    "local_note_density",
    "same_onset_chord_size",
)


# ---------------------------------------------------------------------------
# Derived features
# ---------------------------------------------------------------------------

def register_bucket(pitch: int) -> int:
    if pitch < REGISTER_BOUNDARIES[0]:
        return 0
    if pitch < REGISTER_BOUNDARIES[1]:
        return 1
    if pitch < REGISTER_BOUNDARIES[2]:
        return 2
    return 3


def add_derived_features(events: list[NoteEvent]) -> list[NoteEvent]:
    """Compute derived features for each note event in-place and return.

    TIME_SCALE converts real seconds from pretty_midi to the units
    the model was trained on (centisecond-scale, matching time_scale=1.0
    applied during CSV loading in training).
    """
    if not events:
        return events

    n = len(events)

    # Scale onset/offset for feature computation (matches training pipeline
    # where time_scale is applied at CSV load time)
    scaled_onsets = [e.onset_sec * TIME_SCALE for e in events]
    scaled_offsets = [e.offset_sec * TIME_SCALE for e in events]

    for idx, event in enumerate(events):
        prev = events[idx - 1] if idx > 0 else None
        nxt = events[idx + 1] if idx + 1 < n else None

        # Duration (in scaled time units)
        event.duration_sec = max(scaled_offsets[idx] - scaled_onsets[idx], 0.0)

        # IOI (in scaled time units)
        event.ioi_next_sec = (
            scaled_onsets[idx + 1] - scaled_onsets[idx] if nxt is not None else 0.0
        )

        # Delta pitch (no time scaling needed)
        event.delta_pitch_prev = float(event.pitch - prev.pitch) if prev is not None else 0.0
        event.delta_pitch_next = float(nxt.pitch - event.pitch) if nxt is not None else 0.0

        # Local note density (uses scaled time spans)
        left = max(0, idx - LOCAL_DENSITY_NEIGHBOR_K)
        right = min(n - 1, idx + LOCAL_DENSITY_NEIGHBOR_K)
        span = scaled_onsets[right] - scaled_onsets[left]
        count = right - left + 1
        event.local_note_density = count / span if span > 0 else float(count)

        # Chord size (uses real-second tolerance)
        lo = bisect_left(scaled_onsets, scaled_onsets[idx] - ONSET_TOLERANCE_SEC * TIME_SCALE)
        hi = bisect_right(scaled_onsets, scaled_onsets[idx] + ONSET_TOLERANCE_SEC * TIME_SCALE)
        event.same_onset_chord_size = float(hi - lo)

        # Register
        event.register_bucket = register_bucket(event.pitch)

    return events
