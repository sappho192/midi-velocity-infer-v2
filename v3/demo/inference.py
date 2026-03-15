"""ONNX-based inference pipeline for the demo.

Handles the full flow: MIDI parsing -> feature extraction -> windowing ->
control prediction -> model inference -> reconstruction -> output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort

from features import Stats, add_derived_features, denormalize_velocity, load_stats
from midi_io import NoteEvent, is_likely_piano, parse_midi, write_midi_with_velocities
from windowing import (
    Window,
    build_windows,
    extract_window_features,
    reconstruct_center_priority,
)

MODEL_NAMES = {"regression", "classification", "stochastic"}


@dataclass(slots=True)
class InferenceResult:
    velocities: list[int]
    notes: list[dict]  # For piano roll: {pitch, onset, offset, velocity}
    warning: str | None = None


class VelocityInferenceEngine:
    """Loads ONNX models and runs the full inference pipeline."""

    def __init__(self, models_dir: str | Path) -> None:
        self.models_dir = Path(models_dir)
        self.stats = load_stats(self.models_dir / "stats.json")
        self._sessions: dict[str, ort.InferenceSession] = {}

        # Pre-load control MLP (tiny, always needed)
        mlp_path = self.models_dir / "control_mlp.onnx"
        if mlp_path.exists():
            self._mlp_session = ort.InferenceSession(
                str(mlp_path),
                providers=["CPUExecutionProvider"],
            )
        else:
            self._mlp_session = None

    def _get_session(self, model_name: str) -> ort.InferenceSession:
        if model_name not in self._sessions:
            path = self.models_dir / f"{model_name}.onnx"
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            self._sessions[model_name] = ort.InferenceSession(
                str(path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._sessions[model_name]

    def infer(
        self,
        midi_path: str,
        model_name: str = "regression",
        ctrl_expressiveness: float | None = None,
        ctrl_dynamics: float | None = None,
        temperature: float = 1.0,
        progress_fn: Callable[[int, int], None] | None = None,
    ) -> InferenceResult:
        """Run full inference pipeline.

        Args:
            midi_path: Path to .mid file.
            model_name: One of "regression", "classification", "stochastic".
            ctrl_expressiveness: Manual expressiveness [0,1] or None for auto.
            ctrl_dynamics: Manual dynamics center [0,1] or None for auto.
            temperature: Sampling temperature for stochastic model (0=deterministic).
            progress_fn: Optional callback(current_window, total_windows).

        Returns:
            InferenceResult with velocities and piano roll data.
        """
        # 1. Parse MIDI
        events, pm = parse_midi(midi_path)
        if not events:
            return InferenceResult(velocities=[], notes=[], warning="No notes found in MIDI file.")

        warning = None
        if not is_likely_piano(pm):
            warning = "This model is optimized for piano MIDI. Results for other instruments may vary."

        # 2. Derive features
        events = add_derived_features(events)

        # 3-4. Build windows (normalization happens inside)
        windows = build_windows(events, self.stats)
        if not windows:
            return InferenceResult(velocities=[], notes=[], warning="No windows generated.")

        # 5. Control prediction
        use_auto = ctrl_expressiveness is None and ctrl_dynamics is None
        controls_per_window = self._predict_controls(windows, use_auto, ctrl_expressiveness, ctrl_dynamics)

        # 6. ONNX inference
        session = self._get_session(model_name)
        predictions = []
        total = len(windows)

        for i, (window, ctrl) in enumerate(zip(windows, controls_per_window)):
            pred = self._infer_window(session, window, ctrl, model_name, temperature)
            predictions.append(pred)
            if progress_fn:
                progress_fn(i + 1, total)

        # 7. Reconstruct
        norm_velocities = reconstruct_center_priority(windows, predictions)

        # 8. Denormalize and clamp
        raw_velocities = [
            max(1, min(127, int(round(denormalize_velocity(v, self.stats)))))
            for v in norm_velocities
        ]

        # Build piano roll data
        notes_data = []
        for event, vel in zip(events, raw_velocities):
            notes_data.append({
                "pitch": event.pitch,
                "onset": event.onset_sec,
                "offset": event.offset_sec,
                "velocity": vel,
            })

        return InferenceResult(
            velocities=raw_velocities,
            notes=notes_data,
            warning=warning,
        )

    def _predict_controls(
        self,
        windows: list[Window],
        use_auto: bool,
        ctrl_expr: float | None,
        ctrl_dyn: float | None,
    ) -> list[np.ndarray]:
        """Predict or assign control parameters for each window."""
        if not use_auto:
            # Manual override: same value for all windows
            ctrl = np.array(
                [ctrl_expr or 0.5, ctrl_dyn or 0.5], dtype=np.float32
            )
            return [ctrl] * len(windows)

        # Auto: use MLP
        if self._mlp_session is None:
            # Fallback to default
            ctrl = np.array([0.5, 0.5], dtype=np.float32)
            return [ctrl] * len(windows)

        controls = []
        for window in windows:
            feats = extract_window_features(window)
            feats_input = feats.reshape(1, -1).astype(np.float32)
            out = self._mlp_session.run(None, {"features": feats_input})
            controls.append(out[0][0])  # [2]

        return controls

    def _infer_window(
        self,
        session: ort.InferenceSession,
        window: Window,
        control: np.ndarray,
        model_name: str,
        temperature: float,
    ) -> np.ndarray:
        """Run ONNX inference for a single window."""
        feeds = {
            "pitch": window.pitch.reshape(1, -1),
            "register_bucket": window.register_bucket.reshape(1, -1),
            "continuous": window.continuous.reshape(1, 256, 6),
            "padding_mask": window.padding_mask.reshape(1, -1),
            "control_params": control.reshape(1, -1).astype(np.float32),
        }

        outputs = session.run(None, feeds)

        if model_name == "regression":
            return outputs[0][0]  # [256]

        elif model_name == "classification":
            logits = outputs[0][0]  # [256, 128]
            # Softmax + expectation decode
            probs = _softmax(logits, axis=-1)
            bin_centers = np.arange(128, dtype=np.float32) / 127.0
            return (probs * bin_centers).sum(axis=-1)  # [256]

        elif model_name == "stochastic":
            mu = outputs[0][0]         # [256]
            log_sigma = outputs[1][0]  # [256]
            if temperature <= 0:
                return mu
            sigma = np.exp(log_sigma)
            eps = np.random.randn(*mu.shape).astype(np.float32)
            return mu + temperature * sigma * eps

        else:
            raise ValueError(f"Unknown model: {model_name}")


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
