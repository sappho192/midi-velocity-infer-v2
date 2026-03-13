from __future__ import annotations

import torch


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Uses warmup: decay = min(target_decay, 1 - 1/step) so that early
    updates weigh recent parameters more heavily.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {
            k: v.clone().detach() for k, v in model.state_dict().items()
        }
        self.step_count = 0

    def get_decay(self) -> float:
        self.step_count += 1
        return min(self.decay, 1 - 1 / self.step_count)

    def update(self, model: torch.nn.Module) -> None:
        decay = self.get_decay()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k].lerp_(v, 1 - decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = state["decay"]
        self.shadow = state["shadow"]
        self.step_count = state["step_count"]

    def apply_to(self, model: torch.nn.Module) -> None:
        """Load EMA shadow weights into the model."""
        model.load_state_dict(self.shadow)
