"""Audio synthesis via FluidSynth for MIDI preview.

Outputs FLAC (compression level 5) to reduce file size significantly
compared to WAV (~10x smaller).
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def synthesize_audio(
    midi_path: str,
    sf2_path: str,
    output_path: str | None = None,
    sample_rate: int = 44100,
) -> str | None:
    """Synthesize MIDI to FLAC using FluidSynth.

    Args:
        midi_path: Path to input .mid file.
        sf2_path: Path to SoundFont .sf2 file.
        output_path: Path for output .flac. If None, uses a temp file.
        sample_rate: Audio sample rate.

    Returns:
        Path to FLAC file, or None if synthesis failed.
    """
    if not Path(sf2_path).exists():
        return None

    if output_path is None:
        output_path = tempfile.mktemp(suffix=".flac")
    elif not output_path.endswith(".flac"):
        output_path = str(Path(output_path).with_suffix(".flac"))

    if shutil.which("fluidsynth"):
        audio = _synth_fluidsynth_cli(midi_path, sf2_path, sample_rate)
    else:
        audio = _synth_pretty_midi(midi_path, sf2_path, sample_rate)

    if audio is None:
        return None

    # Normalize to [-0.9, 0.9] to avoid clipping
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9

    # Write FLAC with compression level 5
    sf.write(output_path, audio, sample_rate, format="FLAC", subtype="PCM_16")
    return output_path


def _synth_fluidsynth_cli(
    midi_path: str, sf2_path: str, sample_rate: int
) -> np.ndarray | None:
    """Synthesize via FluidSynth CLI to a temp WAV, then load as numpy."""
    tmp_wav = tempfile.mktemp(suffix=".wav")
    try:
        subprocess.run(
            [
                "fluidsynth", "-ni",
                sf2_path, midi_path,
                "-F", tmp_wav,
                "-r", str(sample_rate),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        audio, _ = sf.read(tmp_wav, dtype="float32")
        return audio
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        Path(tmp_wav).unlink(missing_ok=True)


def _synth_pretty_midi(
    midi_path: str, sf2_path: str, sample_rate: int
) -> np.ndarray | None:
    """Fallback: synthesize using pretty_midi (requires fluidsynth library)."""
    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(midi_path)
        return pm.fluidsynth(fs=sample_rate, sf2_path=sf2_path).astype(np.float32)
    except Exception:
        return None
