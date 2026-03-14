"""Convert GiantMIDI-Piano MIDI files to canonical CSV format.

Usage:
    uv run python scripts/convert_giantmidi_to_csv.py \
        --midi-dir /path/to/GiantMIDI-PIano/midis \
        --metadata /path/to/full_music_pieces_youtube_similarity_pianosoloprob_split.csv \
        --output-dir /path/to/GiantMIDI-PIano/csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)


def midi_to_csv_rows(midi_path: Path, piece_id: str) -> list[dict[str, str]]:
    """Parse a MIDI file and return canonical CSV rows."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[tuple[float, float, int, int]] = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    # Sort by onset, then pitch
    notes.sort(key=lambda n: (n[0], n[2]))
    rows = []
    for idx, (onset, offset, pitch, velocity) in enumerate(notes):
        rows.append({
            "piece_id": piece_id,
            "note_index": str(idx),
            "pitch": str(pitch),
            "onset_sec": f"{onset:.6f}",
            "offset_sec": f"{offset:.6f}",
            "velocity": str(velocity),
        })
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["piece_id", "note_index", "pitch", "onset_sec", "offset_sec", "velocity"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _convert_one(args: tuple[Path, Path, str]) -> tuple[str, bool, str]:
    """Convert a single MIDI file. Returns (piece_id, success, message)."""
    midi_path, output_path, piece_id = args
    try:
        rows = midi_to_csv_rows(midi_path, piece_id)
        if not rows:
            return piece_id, False, "no notes found"
        write_csv(rows, output_path)
        return piece_id, True, f"{len(rows)} notes"
    except Exception as e:
        return piece_id, False, str(e)


def parse_metadata(metadata_path: Path) -> dict[str, dict[str, str]]:
    """Parse the GiantMIDI metadata CSV.

    Returns a dict mapping audio_name -> {split, giant_midi_piano, ...}.
    """
    entries: dict[str, dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("giant_midi_piano") != "1":
                continue
            audio_name = row.get("audio_name", "").strip()
            split = row.get("split", "").strip()
            if audio_name and split:
                entries[audio_name] = {"split": split}
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GiantMIDI-Piano MIDI to CSV")
    parser.add_argument("--midi-dir", required=True, help="Directory containing .mid files")
    parser.add_argument("--metadata", required=True, help="Path to metadata TSV")
    parser.add_argument("--output-dir", required=True, help="Output directory for CSV splits")
    args = parser.parse_args()

    midi_dir = Path(args.midi_dir)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse metadata
    logger.info("Parsing metadata from %s", metadata_path)
    entries = parse_metadata(metadata_path)
    logger.info("Found %d pieces with giant_midi_piano=1", len(entries))

    # Count splits
    split_counts: dict[str, int] = {}
    for info in entries.values():
        s = info["split"]
        split_counts[s] = split_counts.get(s, 0) + 1
    for s, c in sorted(split_counts.items()):
        logger.info("  %s: %d pieces", s, c)

    # Build conversion tasks
    tasks: list[tuple[Path, Path, str]] = []
    missing = 0
    for audio_name, info in entries.items():
        midi_path = midi_dir / f"{audio_name}.mid"
        if not midi_path.exists():
            missing += 1
            continue
        split = info["split"]
        # Use audio_name as piece_id but sanitize for filename
        safe_name = audio_name.replace("/", "_").replace("\\", "_")
        output_path = output_dir / split / f"{safe_name}.csv"
        tasks.append((midi_path, output_path, audio_name))

    if missing:
        logger.warning("%d MIDI files not found in %s", missing, midi_dir)
    logger.info("Converting %d MIDI files...", len(tasks))

    # Parallel conversion
    workers = min(len(tasks), os.cpu_count() or 1)
    success_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_convert_one, task): task[2] for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            piece_id, ok, msg = future.result()
            if ok:
                success_count += 1
            else:
                fail_count += 1
                logger.warning("FAILED %s: %s", piece_id, msg)
            if i % 500 == 0:
                logger.info("  progress: %d/%d", i, len(tasks))

    logger.info("Done: %d succeeded, %d failed", success_count, fail_count)

    # Report output
    for split_name in sorted(split_counts.keys()):
        split_dir = output_dir / split_name
        if split_dir.exists():
            n = len(list(split_dir.glob("*.csv")))
            logger.info("  %s: %d CSV files", split_name, n)


if __name__ == "__main__":
    main()
