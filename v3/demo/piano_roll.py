"""Interactive piano roll visualization using Plotly.

Renders:
1. Top: Piano roll with color-coded velocity (green->red, FL Studio style)
2. Bottom: Velocity bars per note

Uses efficient single-trace rendering for thousands of notes.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def _velocity_colors(velocities: list[int]) -> list[str]:
    """FL Studio style: green (low) -> red (high)."""
    colors = []
    for v in velocities:
        t = max(0.0, min(1.0, v / 127))
        r = int(34 + t * (239 - 34))
        g = int(197 - t * (197 - 68))
        b = int(94 - t * (94 - 68))
        colors.append(f"rgb({r},{g},{b})")
    return colors


def render_piano_roll(notes: list[dict], height: int = 600) -> go.Figure | None:
    """Create an interactive Plotly piano roll figure.

    Args:
        notes: List of {pitch, onset, offset, velocity} dicts.
        height: Total figure height in pixels.

    Returns:
        Plotly Figure or None if no notes.
    """
    if not notes:
        return None

    pitches = [n["pitch"] for n in notes]
    onsets = [n["onset"] for n in notes]
    offsets = [n["offset"] for n in notes]
    velocities = [n["velocity"] for n in notes]

    min_pitch = max(0, min(pitches) - 2)
    max_pitch = min(127, max(pitches) + 2)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    # --- Piano roll (top): horizontal bars ---
    colors = _velocity_colors(velocities)
    durations = [off - on for on, off in zip(onsets, offsets)]

    # Use Bar with horizontal base for note rectangles
    fig.add_trace(
        go.Bar(
            x=durations,
            y=pitches,
            base=onsets,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            width=0.7,
            customdata=np.column_stack([velocities, onsets, offsets]),
            hovertemplate=(
                "%{y} | vel: %{customdata[0]}<br>"
                "%{customdata[1]:.2f}s - %{customdata[2]:.2f}s"
                "<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # --- Velocity bars (bottom) ---
    fig.add_trace(
        go.Bar(
            x=onsets,
            y=velocities,
            marker=dict(color=colors, line=dict(width=0)),
            width=max(0.01, (max(offsets) - min(onsets)) / len(notes) * 0.3),
            hovertemplate="vel: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Layout ---
    fig.update_layout(
        height=height,
        margin=dict(l=60, r=20, t=10, b=40),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        dragmode="pan",
        bargap=0,
        barmode="overlay",
    )

    # Piano roll y-axis
    tick_step = max(1, (max_pitch - min_pitch) // 12)
    tick_pitches = list(range(min_pitch, max_pitch + 1, tick_step))
    fig.update_yaxes(
        range=[min_pitch - 1, max_pitch + 1],
        tickvals=tick_pitches,
        ticktext=[_note_name(p) for p in tick_pitches],
        gridcolor="#2a2a5a",
        color="#999",
        row=1, col=1,
    )

    # Velocity y-axis
    fig.update_yaxes(
        range=[0, 130],
        tickvals=[0, 32, 64, 96, 127],
        gridcolor="#2a2a5a",
        color="#999",
        title_text="vel",
        row=2, col=1,
    )

    # X-axes
    fig.update_xaxes(gridcolor="#2a2a5a", color="#999", row=1, col=1)
    fig.update_xaxes(
        gridcolor="#2a2a5a",
        color="#999",
        title_text="Time (sec)",
        row=2, col=1,
    )

    return fig
