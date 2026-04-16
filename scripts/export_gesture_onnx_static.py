#!/usr/bin/env python3
"""Export gesture classifier checkpoint (.pth) to static-shape ONNX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export gesture classifier to static ONNX.")
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to checkpoint (.pth), e.g. training_runs/run_xxx/best_model.pth",
    )
    parser.add_argument(
        "--output_onnx",
        type=Path,
        default=Path("artifacts/gesture_classifier_static.onnx"),
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=0,
        help="Override image size. 0 uses checkpoint image_size.",
    )
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")

    device = torch.device(args.device)
    checkpoint = torch.load(args.model_path, map_location=device)

    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise RuntimeError("Checkpoint missing valid class_names")

    ckpt_image_size = int(checkpoint.get("image_size", 160))
    image_size = args.image_size if args.image_size > 0 else ckpt_image_size

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    args.output_onnx.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(args.output_onnx),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        opset_version=args.opset,
    )

    meta = {
        "source_checkpoint": str(args.model_path),
        "output_onnx": str(args.output_onnx),
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_size": image_size,
        "opset": args.opset,
    }
    meta_path = args.output_onnx.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("[DONE] Gesture ONNX export finished")
    print(f"[DONE] onnx : {args.output_onnx}")
    print(f"[DONE] meta : {meta_path}")


if __name__ == "__main__":
    main()
