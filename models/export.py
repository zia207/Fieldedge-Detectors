"""Build every converted model on the GPU and save its weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import available_models, build_model, save_model


def export_all(output_dir: str | Path = "models/weights", num_classes: int = 2, image_size: int = 64) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = []
    for name in available_models():
        model = build_model(name, num_classes=num_classes, device=device)
        model.eval()
        sample = torch.randn(1, 3, image_size, image_size, device=device)
        with torch.inference_mode():
            logits = model(sample)
        if logits.shape != (1, num_classes, image_size, image_size):
            raise RuntimeError(f"{name} returned {tuple(logits.shape)}")
        path = save_model(model, output_dir / f"{name}.pt")
        params = sum(parameter.numel() for parameter in model.parameters())
        print(f"{name:24}  {params/1e6:7.2f}M params  {tuple(logits.shape)}  {device.type}  {path}")
        saved.append(path)
        del model, sample, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Export every segmentation model as a PyTorch checkpoint.")
    parser.add_argument("--output-dir", default="models/weights")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()
    export_all(args.output_dir, args.num_classes, args.image_size)


if __name__ == "__main__":
    main()
