"""Gradio demo for MIDI Velocity Inference.

Left panel: upload, model selection, control sliders, generate button.
Right panel: piano roll visualization, audio preview, MIDI download.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import gradio as gr

from audio import synthesize_audio
from inference import VelocityInferenceEngine
from midi_io import write_midi_with_velocities, parse_midi
from piano_roll import render_piano_roll

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).resolve().parent
MODELS_DIR = DEMO_DIR / "models"
SF2_PATH = DEMO_DIR / "assets" / "Nice-Steinway-Lite-v3.0.sf2"

# ---------------------------------------------------------------------------
# Model name mapping
# ---------------------------------------------------------------------------

MODEL_CHOICES = [
    "Balanced (Regression)",
    "Precise (Classification)",
    "Creative (Stochastic)",
]

MODEL_MAP = {
    "Balanced (Regression)": "regression",
    "Precise (Classification)": "classification",
    "Creative (Stochastic)": "stochastic",
}

# ---------------------------------------------------------------------------
# Engine (lazy init)
# ---------------------------------------------------------------------------

engine: VelocityInferenceEngine | None = None


def get_engine() -> VelocityInferenceEngine:
    global engine
    if engine is None:
        engine = VelocityInferenceEngine(MODELS_DIR)
    return engine


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def on_model_change(model_label: str):
    """Show/hide temperature slider based on model selection."""
    is_stochastic = "Stochastic" in model_label
    return gr.update(visible=is_stochastic)


def on_auto_change(is_auto: bool):
    """Enable/disable control sliders when auto mode changes."""
    return gr.update(interactive=not is_auto), gr.update(interactive=not is_auto)


def generate(
    midi_file,
    model_label: str,
    auto_control: bool,
    ctrl_dynamics: float,
    ctrl_expressiveness: float,
    temperature: float,
    progress=gr.Progress(),
):
    """Run inference and return results."""
    if midi_file is None:
        gr.Warning("Please upload a MIDI file first.")
        return None, None, None

    eng = get_engine()
    model_name = MODEL_MAP[model_label]

    # Determine control mode
    ctrl_expr = None if auto_control else ctrl_expressiveness
    ctrl_dyn = None if auto_control else ctrl_dynamics

    # Progress callback
    def progress_fn(current: int, total: int):
        progress(current / total, desc=f"Processing window {current}/{total}")

    result = eng.infer(
        midi_path=midi_file,
        model_name=model_name,
        ctrl_expressiveness=ctrl_expr,
        ctrl_dynamics=ctrl_dyn,
        temperature=temperature,
        progress_fn=progress_fn,
    )

    if result.warning:
        gr.Warning(result.warning)

    if not result.velocities:
        gr.Warning("No predictions generated.")
        return None, None, None

    # Piano roll figure
    pr_fig = render_piano_roll(result.notes, height=480)

    # Write output MIDI
    _, pm = parse_midi(midi_file)
    output_midi = tempfile.mktemp(suffix=".mid")
    write_midi_with_velocities(pm, result.velocities, output_midi)

    # Synthesize audio
    audio_path = None
    if SF2_PATH.exists():
        audio_path = synthesize_audio(output_midi, str(SF2_PATH))

    return pr_fig, audio_path, output_midi


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def create_app() -> gr.Blocks:
    with gr.Blocks(title="MIDI Velocity Inference") as app:
        gr.Markdown("# MIDI Velocity Inference Demo")
        gr.Markdown(
            "Upload a piano MIDI file and generate expressive velocity predictions "
            "using transformer-based models."
        )

        with gr.Row():
            # --- Left panel ---
            with gr.Column(scale=1, min_width=280):
                midi_upload = gr.File(
                    label="Upload MIDI",
                    file_types=[".mid", ".midi"],
                    type="filepath",
                )

                model_selector = gr.Radio(
                    choices=MODEL_CHOICES,
                    value=MODEL_CHOICES[0],
                    label="Model",
                )

                gr.Markdown("### Controls")
                auto_control = gr.Checkbox(
                    value=True,
                    label="Auto (MLP predicts per-window)",
                )
                ctrl_dynamics = gr.Slider(
                    0.0, 1.0, 0.5,
                    step=0.01,
                    label="Overall Dynamics (Soft \u2194 Loud)",
                    interactive=False,
                )
                ctrl_expressiveness = gr.Slider(
                    0.0, 1.0, 0.5,
                    step=0.01,
                    label="Dynamic Range (Narrow \u2194 Wide)",
                    interactive=False,
                )
                temperature = gr.Slider(
                    0.0, 2.0, 1.0,
                    step=0.1,
                    label="Temperature (Creative mode only)",
                    visible=False,
                )

                generate_btn = gr.Button("Generate", variant="primary", size="lg")

            # --- Right panel ---
            with gr.Column(scale=3):
                piano_roll_output = gr.Plot(label="Piano Roll")
                with gr.Row():
                    audio_output = gr.Audio(label="Preview", type="filepath")
                    midi_output = gr.File(label="Download MIDI")

        # --- Events ---
        model_selector.change(
            on_model_change, inputs=[model_selector], outputs=[temperature]
        )
        auto_control.change(
            on_auto_change,
            inputs=[auto_control],
            outputs=[ctrl_dynamics, ctrl_expressiveness],
        )
        generate_btn.click(
            generate,
            inputs=[
                midi_upload,
                model_selector,
                auto_control,
                ctrl_dynamics,
                ctrl_expressiveness,
                temperature,
            ],
            outputs=[piano_roll_output, audio_output, midi_output],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=True)
