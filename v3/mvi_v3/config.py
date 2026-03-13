from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class BaselineConfig:
    # Data pipeline
    window_size: int = 256
    stride: int = 128
    onset_tolerance_sec: float = 0.03
    local_density_neighbor_k: int = 8
    register_boundaries: tuple[int, int, int] = (48, 60, 72)
    time_scale: float = 1.0
    continuous_features: tuple[str, ...] = field(
        default_factory=lambda: (
            "duration_sec",
            "ioi_next_sec",
            "delta_pitch_prev",
            "delta_pitch_next",
            "local_note_density",
            "same_onset_chord_size",
        )
    )

    # Model architecture
    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    num_layers: int = 4
    dropout: float = 0.1

    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 256
    epochs: int = 100
    seed: int = 42
    max_grad_norm: float = 1.0
    warmup_fraction: float = 0.05
    patience: int = 10
    huber_delta: float = 1.0
    velocity_weight_beta: float = 3.0
    gradient_accumulation_steps: int = 1
    ema_decay: float = 0.999
    ema_enabled: bool = True

    # Checkpoint / resume
    resume_from: str | None = None

    # Output head
    head_type: str = "regression"  # "regression", "classification", "stochastic"
    num_velocity_bins: int = 128
    label_smoothing: float = 0.1

    # Controllable velocity inference
    enable_controls: bool = False
    control_dims: int = 3
    stochastic_head: bool = False  # deprecated, use head_type="stochastic"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
