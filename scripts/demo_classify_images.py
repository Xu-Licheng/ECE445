#!/usr/bin/env python3
"""Demo inference script for classifying images into class folders.

Workflow:
1) Load a specified .pth checkpoint from training.
2) Create output subfolders under demo directory (one folder per class).
3) Classify images directly under demo directory.
4) Copy or move each image to the predicted class folder.
5) Save prediction details to CSV.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify images in demo folder using a trained gesture model."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pth), e.g. training_runs/run_xxx/best_model.pth",
    )
    parser.add_argument(
        "--demo_dir",
        type=Path,
        default=Path("test_demo"),
        help="Demo folder that stores images to classify and output class subfolders.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Move files into predicted class folders. Default behavior is copy.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional CSV path. Default: <demo_dir>/predictions.csv",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested by --device, but current PyTorch cannot use CUDA. "
            "Install a CUDA-enabled torch build or run with --device cpu/auto."
        )
    return device


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing key: model_state_dict")

    class_names = checkpoint.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise KeyError("Checkpoint missing valid class_names")

    image_size = int(checkpoint.get("image_size", 160))

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, image_size


def make_eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def list_demo_images(demo_dir: Path) -> list[Path]:
    if not demo_dir.exists():
        return []

    images = [
        path
        for path in demo_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images)


def ensure_class_dirs(demo_dir: Path, class_names: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    demo_dir.mkdir(parents=True, exist_ok=True)

    for class_name in class_names:
        class_dir = demo_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        mapping[class_name] = class_dir
    return mapping


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def predict_single(
    model: nn.Module,
    transform: transforms.Compose,
    image_path: Path,
    device: torch.device,
) -> tuple[int, float]:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return int(pred_idx.item()), float(confidence.item())


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    model, class_names, image_size = load_checkpoint(args.model_path, device)
    transform = make_eval_transform(image_size)

    class_dirs = ensure_class_dirs(args.demo_dir, class_names)
    image_paths = list_demo_images(args.demo_dir)

    if not image_paths:
        print(f"[INFO] No images found under: {args.demo_dir}")
        print("[INFO] Put images directly into the demo root folder, then rerun.")
        print("[INFO] Class output folders are ready:")
        for class_name in class_names:
            print(f"  - {class_dirs[class_name]}")
        return

    output_csv = args.output_csv or (args.demo_dir / "predictions.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_count: dict[str, int] = {name: 0 for name in class_names}

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_name", "pred_class", "confidence", "source_path", "output_path"])

        for image_path in image_paths:
            pred_idx, confidence = predict_single(
                model=model,
                transform=transform,
                image_path=image_path,
                device=device,
            )
            pred_class = class_names[pred_idx]
            destination = unique_destination(class_dirs[pred_class] / image_path.name)

            if args.move_files:
                shutil.move(str(image_path), str(destination))
            else:
                shutil.copy2(str(image_path), str(destination))

            summary_count[pred_class] += 1
            writer.writerow(
                [
                    image_path.name,
                    pred_class,
                    f"{confidence:.6f}",
                    str(image_path),
                    str(destination),
                ]
            )

            print(f"[PRED] {image_path.name} -> {pred_class} ({confidence:.4f})")

    print("\n[DONE] Demo inference finished")
    print(f"Model   : {args.model_path}")
    print(f"Device  : {device}")
    print(f"Demo dir: {args.demo_dir}")
    print(f"CSV     : {output_csv}")
    print("[SUMMARY]")
    for class_name in class_names:
        print(f"- {class_name}: {summary_count[class_name]}")


if __name__ == "__main__":
    main()
