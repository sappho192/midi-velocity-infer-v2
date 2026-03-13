from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class BaselineConfig:
    window_size: int = 256
    stride: int = 128
    onset_tolerance_sec: float = 0.03
    local_density_neighbor_k: int = 8
    register_boundaries: tuple[int, int, int] = (48, 60, 72)
    d_model: int = 256
    n_heads: int = 8
    ffn_dim: int = 1024
    num_layers: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-4
    batch_size: int = 16
    epochs: int = 10
    seed: int = 42
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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
