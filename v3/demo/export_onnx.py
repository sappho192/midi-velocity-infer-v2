"""One-time ONNX export script for demo models.

Converts 3 PyTorch velocity models + 1 control MLP to ONNX format.
Requires torch and mvi_v3 (training codebase). Not needed at demo runtime.

Usage:
    uv run python export_onnx.py \
        --reg-ckpt ../runs/ssl_finetune/best.pt \
        --cls-ckpt ../runs_ablation/cls_head/best.pt \
        --stoch-ckpt ../runs_ablation/stoch_head/best.pt \
        --mlp-ckpt ../runs/ssl_finetune/control_mlp/control_mlp.pt \
        --stats ../runs/ssl_finetune/stats.json \
        --output-dir models
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

# Add v3/ to path so mvi_v3 is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mvi_v3.config import BaselineConfig
from mvi_v3.models.control_predictor import ControlPredictorMLP
from mvi_v3.models.transformer import TransformerVelocityModel


# ---------------------------------------------------------------------------
# Wrapper modules for clean ONNX export
# ---------------------------------------------------------------------------

class RegressionWrapper(nn.Module):
    """Wraps TransformerVelocityModel for regression ONNX export."""

    def __init__(self, model: TransformerVelocityModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
        padding_mask: torch.Tensor,
        control_params: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(pitch, register_bucket, continuous, padding_mask, control_params)


class ClassificationWrapper(nn.Module):
    """Wraps TransformerVelocityModel for classification ONNX export.
    Returns raw logits [1, 256, 128]. Softmax+expectation done in numpy.
    """

    def __init__(self, model: TransformerVelocityModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
        padding_mask: torch.Tensor,
        control_params: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(pitch, register_bucket, continuous, padding_mask, control_params)


class StochasticWrapper(nn.Module):
    """Wraps TransformerVelocityModel for stochastic ONNX export.
    Returns (mu, log_sigma) as two separate outputs.
    """

    def __init__(self, model: TransformerVelocityModel) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        pitch: torch.Tensor,
        register_bucket: torch.Tensor,
        continuous: torch.Tensor,
        padding_mask: torch.Tensor,
        control_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(pitch, register_bucket, continuous, padding_mask, control_params)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str, device: str = "cpu") -> TransformerVelocityModel:
    """Load model from checkpoint with EMA weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt["config"]

    config = BaselineConfig(
        head_type=cfg_dict["head_type"],
        enable_controls=cfg_dict.get("enable_controls", False),
        control_dims=cfg_dict.get("control_dims", 2),
        d_model=cfg_dict.get("d_model", 256),
        n_heads=cfg_dict.get("n_heads", 8),
        ffn_dim=cfg_dict.get("ffn_dim", 1024),
        num_layers=cfg_dict.get("num_layers", 4),
        dropout=cfg_dict.get("dropout", 0.1),
        num_velocity_bins=cfg_dict.get("num_velocity_bins", 128),
        embedding_dropout=cfg_dict.get("embedding_dropout", 0.0),
        mask_ratio=0.0,  # Force no aux head for export
        stochastic_head=cfg_dict.get("stochastic_head", False),
    )
    model = TransformerVelocityModel(config)

    # Prefer EMA shadow weights
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"]:
        state_dict = ckpt["ema_state_dict"]["shadow"]
    else:
        state_dict = ckpt["model_state_dict"]

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def make_dummy_inputs(device: str = "cpu") -> dict[str, torch.Tensor]:
    """Create dummy inputs for ONNX tracing (batch=1, seq_len=256)."""
    return {
        "pitch": torch.randint(0, 128, (1, 256), dtype=torch.long, device=device),
        "register_bucket": torch.randint(0, 4, (1, 256), dtype=torch.long, device=device),
        "continuous": torch.randn(1, 256, 6, dtype=torch.float32, device=device),
        "padding_mask": torch.zeros(1, 256, dtype=torch.bool, device=device),
        "control_params": torch.rand(1, 2, dtype=torch.float32, device=device),
    }


def export_velocity_model(
    checkpoint_path: str,
    output_path: str,
    head_type: str,
) -> None:
    """Export a velocity model to ONNX."""
    print(f"Exporting {head_type} model from {checkpoint_path}...")
    model = load_model(checkpoint_path)

    wrapper_cls = {
        "regression": RegressionWrapper,
        "classification": ClassificationWrapper,
        "stochastic": StochasticWrapper,
    }[head_type]
    wrapper = wrapper_cls(model)
    wrapper.eval()

    dummy = make_dummy_inputs()
    input_tuple = (
        dummy["pitch"],
        dummy["register_bucket"],
        dummy["continuous"],
        dummy["padding_mask"],
        dummy["control_params"],
    )

    input_names = ["pitch", "register_bucket", "continuous", "padding_mask", "control_params"]

    if head_type == "stochastic":
        output_names = ["mu", "log_sigma"]
    elif head_type == "classification":
        output_names = ["logits"]
    else:
        output_names = ["velocity"]

    torch.onnx.export(
        wrapper,
        input_tuple,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  Saved to {output_path}")

    # Validate
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    feeds = {k: v.numpy() for k, v in dummy.items()}
    outputs = sess.run(None, feeds)
    for name, arr in zip(output_names, outputs):
        print(f"  Output '{name}': shape={arr.shape}, dtype={arr.dtype}")


def export_control_mlp(mlp_path: str, output_path: str) -> None:
    """Export control predictor MLP to ONNX."""
    print(f"Exporting control MLP from {mlp_path}...")
    mlp_ckpt = torch.load(mlp_path, map_location="cpu", weights_only=False)

    mlp = ControlPredictorMLP(**mlp_ckpt["config"])
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()

    dummy_features = torch.randn(1, 21, dtype=torch.float32)

    torch.onnx.export(
        mlp,
        dummy_features,
        output_path,
        input_names=["features"],
        output_names=["controls"],
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Saved to {output_path}")

    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"features": dummy_features.numpy()})
    print(f"  Output 'controls': shape={out[0].shape}, dtype={out[0].dtype}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export velocity models to ONNX")
    parser.add_argument("--reg-ckpt", required=True, help="Regression model checkpoint")
    parser.add_argument("--cls-ckpt", required=True, help="Classification model checkpoint")
    parser.add_argument("--stoch-ckpt", required=True, help="Stochastic model checkpoint")
    parser.add_argument("--mlp-ckpt", required=True, help="Control MLP checkpoint")
    parser.add_argument("--stats", required=True, help="stats.json path")
    parser.add_argument("--output-dir", default="models", help="Output directory")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    export_velocity_model(args.reg_ckpt, str(out / "regression.onnx"), "regression")
    export_velocity_model(args.cls_ckpt, str(out / "classification.onnx"), "classification")
    export_velocity_model(args.stoch_ckpt, str(out / "stochastic.onnx"), "stochastic")
    export_control_mlp(args.mlp_ckpt, str(out / "control_mlp.onnx"))

    # Copy stats.json
    shutil.copy2(args.stats, str(out / "stats.json"))
    print(f"\nCopied stats.json to {out / 'stats.json'}")
    print("\nDone! All ONNX models exported.")


if __name__ == "__main__":
    main()
