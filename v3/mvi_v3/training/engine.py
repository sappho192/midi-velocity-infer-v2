from collections.abc import Iterable

import torch

from .losses import masked_l1_loss


def run_epoch(
    model: torch.nn.Module,
    dataloader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_steps = 0

    for batch in dataloader:
        pitch = batch["pitch"].to(device)
        register_bucket = batch["register_bucket"].to(device)
        continuous = batch["continuous"].to(device)
        target = batch["target_velocity"].to(device)
        padding_mask = batch["padding_mask"].to(device)

        with torch.set_grad_enabled(training):
            prediction = model(pitch, register_bucket, continuous, padding_mask)
            loss = masked_l1_loss(prediction, target, padding_mask)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_steps += 1

    return total_loss / max(total_steps, 1)
