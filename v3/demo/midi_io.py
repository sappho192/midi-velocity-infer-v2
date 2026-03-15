"""MIDI file parsing and output for the demo.

Converts .mid files to internal NoteEvent format and writes back
with replaced velocities while preserving all metadata.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import pretty_midi


@dataclass(slots=True)
class NoteEvent:
    note_index: int
    pitch: int
    onset_sec: float
    offset_sec: float
    velocity: int | None = None
    # Derived features (populated by features.py)
    duration_sec: float = 0.0
    ioi_next_sec: float = 0.0
    delta_pitch_prev: float = 0.0
    delta_pitch_next: float = 0.0
    same_onset_chord_size: float = 1.0
    local_note_density: float = 0.0
    register_bucket: int = 0


def parse_midi(path: str) -> tuple[list[NoteEvent], pretty_midi.PrettyMIDI]:
    """Parse a MIDI file into NoteEvent list + original PrettyMIDI object.

    All instruments are flattened into a single note list sorted by
    (onset_sec, pitch). The original PrettyMIDI is retained for
    metadata preservation when writing output.

    Returns:
        (events, pm) where events are sorted and re-indexed.
    """
    pm = pretty_midi.PrettyMIDI(path)

    raw_notes: list[tuple[float, float, int, int, int, int]] = []
    for inst_idx, instrument in enumerate(pm.instruments):
        for note_idx, note in enumerate(instrument.notes):
            raw_notes.append((
                note.start,
                note.end,
                note.pitch,
                note.velocity,
                inst_idx,
                note_idx,
            ))

    # Sort by (onset, pitch)
    raw_notes.sort(key=lambda x: (x[0], x[2]))

    events = []
    for idx, (start, end, pitch, velocity, _, _) in enumerate(raw_notes):
        events.append(NoteEvent(
            note_index=idx,
            pitch=pitch,
            onset_sec=start,
            offset_sec=end,
            velocity=velocity,
        ))

    return events, pm


def write_midi_with_velocities(
    original_pm: pretty_midi.PrettyMIDI,
    velocities: list[int],
    output_path: str,
) -> None:
    """Write a MIDI file with replaced velocities, preserving all metadata.

    Notes are matched by the same sort order used in parse_midi.
    """
    pm = copy.deepcopy(original_pm)

    # Flatten and sort notes in the same order as parse_midi
    all_notes: list[tuple[float, int, pretty_midi.Note]] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            all_notes.append((note.start, note.pitch, note))

    all_notes.sort(key=lambda x: (x[0], x[1]))

    for idx, (_, _, note) in enumerate(all_notes):
        if idx < len(velocities):
            note.velocity = max(1, min(127, velocities[idx]))

    pm.write(output_path)


def is_likely_piano(pm: pretty_midi.PrettyMIDI) -> bool:
    """Heuristic check if the MIDI is likely piano music."""
    for inst in pm.instruments:
        if inst.is_drum:
            return False
        # Piano programs are 0-7
        if inst.program > 7:
            return False
    return True
