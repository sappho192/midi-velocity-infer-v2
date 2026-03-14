from bisect import bisect_left, bisect_right
from collections.abc import Sequence

from mvi_v3.config import BaselineConfig

from .events import NoteEvent


def sort_and_reindex(events: Sequence[NoteEvent]) -> list[NoteEvent]:
    ordered = sorted(events, key=lambda event: (event.onset_sec, event.pitch, event.note_index))
    reindexed: list[NoteEvent] = []
    for idx, event in enumerate(ordered):
        reindexed.append(
            NoteEvent(
                piece_id=event.piece_id,
                note_index=idx,
                pitch=event.pitch,
                onset_sec=event.onset_sec,
                offset_sec=event.offset_sec,
                velocity=event.velocity,
            )
        )
    return reindexed


def register_bucket(pitch: int, boundaries: tuple[int, int, int]) -> int:
    if pitch < boundaries[0]:
        return 0
    if pitch < boundaries[1]:
        return 1
    if pitch < boundaries[2]:
        return 2
    return 3


def add_derived_features(events: Sequence[NoteEvent], config: BaselineConfig) -> list[NoteEvent]:
    if not events:
        return []

    result: list[NoteEvent] = []
    n_events = len(events)
    tol = config.onset_tolerance_sec
    neighbor_k = config.local_density_neighbor_k

    # Pre-extract onset times for O(log n) chord_size via bisect
    onsets = [e.onset_sec for e in events]

    for idx, event in enumerate(events):
        prev_event = events[idx - 1] if idx > 0 else None
        next_event = events[idx + 1] if idx + 1 < n_events else None
        left = max(0, idx - neighbor_k)
        right = min(n_events - 1, idx + neighbor_k)
        span = events[right].onset_sec - events[left].onset_sec
        count = right - left + 1
        density = count / span if span > 0 else float(count)
        # O(log n) chord size using sorted onset times + bisect
        lo = bisect_left(onsets, event.onset_sec - tol)
        hi = bisect_right(onsets, event.onset_sec + tol)
        chord_size = hi - lo

        result.append(
            NoteEvent(
                piece_id=event.piece_id,
                note_index=event.note_index,
                pitch=event.pitch,
                onset_sec=event.onset_sec,
                offset_sec=event.offset_sec,
                velocity=event.velocity,
                duration_sec=max(event.offset_sec - event.onset_sec, 0.0),
                ioi_next_sec=0.0 if next_event is None else next_event.onset_sec - event.onset_sec,
                delta_pitch_prev=0.0 if prev_event is None else event.pitch - prev_event.pitch,
                delta_pitch_next=0.0 if next_event is None else next_event.pitch - event.pitch,
                same_onset_chord_size=float(chord_size),
                local_note_density=float(density),
                register_bucket=register_bucket(event.pitch, config.register_boundaries),
            )
        )
    return result
