"""GPU-ready PyTorch ports of the crop-boundary segmentation models.

Each network takes an NCHW float tensor and returns logits of shape
``(N, num_classes, H, W)``. ``num_classes`` defaults to 2, matching
``cropboundarycode/boundarydatargb/class_dict.csv`` (boundary and field).
"""

from __future__ import annotations

from pathlib import Path

import torch

from models.catalog import MODEL_FACTORIES

__all__ = ["available_models", "build_model", "load_model", "save_model"]


def available_models() -> list[str]:
    return list(MODEL_FACTORIES)


def _device(device: str | torch.device | None) -> torch.device:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def build_model(name: str, num_classes: int = 2, device: str | torch.device | None = None) -> torch.nn.Module:
    """Build one converted model and move it to the GPU when CUDA is available."""
    if name not in MODEL_FACTORIES:
        known = ", ".join(MODEL_FACTORIES)
        raise KeyError(f"Unknown model '{name}'. Available models: {known}")
    model = MODEL_FACTORIES[name](num_classes)
    model.name = name
    model.num_classes = num_classes
    return model.to(_device(device))


def save_model(model: torch.nn.Module, path: str | Path) -> Path:
    """Save a state dict that can be loaded back with ``load_model``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = getattr(model, "name", None)
    num_classes = getattr(model, "num_classes", None)
    if name is None or num_classes is None:
        raise ValueError("Model is missing name or num_classes. Build it with build_model().")
    torch.save(
        {"name": name, "num_classes": num_classes, "state_dict": model.state_dict()},
        path,
    )
    return path


def load_model(path: str | Path, device: str | torch.device | None = None) -> torch.nn.Module:
    """Load a checkpoint written by ``save_model`` onto the requested device."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["name"], num_classes=checkpoint["num_classes"], device="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(_device(device))
