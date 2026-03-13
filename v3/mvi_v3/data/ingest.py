import csv
from pathlib import Path

from .events import NoteEvent


LEGACY_REQUIRED_COLUMNS = {"time", "length", "note_num", "velocity"}
CANONICAL_REQUIRED_COLUMNS = {"piece_id", "note_index", "pitch", "onset_sec", "offset_sec", "velocity"}


def _parse_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _parse_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def load_piece_csv(path: str | Path, time_scale: float = 1.0) -> list[NoteEvent]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return []

    columns = set(rows[0].keys())
    piece_id = path.stem
    events: list[NoteEvent] = []

    if CANONICAL_REQUIRED_COLUMNS.issubset(columns):
        for row in rows:
            velocity = row["velocity"]
            events.append(
                NoteEvent(
                    piece_id=row.get("piece_id") or piece_id,
                    note_index=_parse_int(row, "note_index"),
                    pitch=_parse_int(row, "pitch"),
                    onset_sec=_parse_float(row, "onset_sec"),
                    offset_sec=_parse_float(row, "offset_sec"),
                    velocity=None if velocity == "" else int(float(velocity)),
                )
            )
        return events

    if LEGACY_REQUIRED_COLUMNS.issubset(columns):
        for i, row in enumerate(rows):
            onset_sec = _parse_float(row, "time") * time_scale
            duration_sec = _parse_float(row, "length") * time_scale
            events.append(
                NoteEvent(
                    piece_id=piece_id,
                    note_index=i,
                    pitch=_parse_int(row, "note_num"),
                    onset_sec=onset_sec,
                    offset_sec=onset_sec + duration_sec,
                    velocity=_parse_int(row, "velocity"),
                )
            )
        return events

    raise ValueError(f"Unsupported CSV schema for {path}")


def load_piece_directory(path: str | Path, time_scale: float = 1.0) -> list[list[NoteEvent]]:
    path = Path(path)
    pieces: list[list[NoteEvent]] = []
    for csv_path in sorted(path.glob("*.csv")):
        piece = load_piece_csv(csv_path, time_scale=time_scale)
        if piece:
            pieces.append(piece)
    return pieces
