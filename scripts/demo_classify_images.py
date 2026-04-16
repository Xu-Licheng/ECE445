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

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from hand_box_detector import HandBoxDetector


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
    parser.add_argument(
        "--hand_detector",
        type=str,
        default="auto",
        choices=["auto", "yolo", "mediapipe", "skin"],
        help="Hand detector backend used before gesture classification.",
    )
    parser.add_argument(
        "--hand_det_model_path",
        type=Path,
        default=None,
        help="Path to trained YOLO hand detector (.pt/.onnx).",
    )
    parser.add_argument(
        "--hand_det_imgsz",
        type=int,
        default=320,
        help="Input size for YOLO hand detector.",
    )
    parser.add_argument(
        "--hand_det_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO hand detector.",
    )
    parser.add_argument(
        "--hand_det_iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for YOLO hand detector.",
    )
    parser.add_argument(
        "--box_expand_ratio",
        type=float,
        default=1.25,
        help="Expand detected hand box by this ratio before cropping.",
    )
    parser.add_argument(
        "--hand_min_area_ratio",
        type=float,
        default=0.01,
        help="Minimum hand area ratio for skin fallback detector.",
    )
    parser.add_argument(
        "--require_hand_box",
        action="store_true",
        help="Skip image if no hand box is detected.",
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


def image_bgr_to_tensor(
    image_bgr: "cv2.typing.MatLike",
    transform: transforms.Compose,
) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    return transform(image).unsqueeze(0)


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
    image_bgr: "cv2.typing.MatLike",
    device: torch.device,
) -> tuple[int, float]:
    tensor = image_bgr_to_tensor(image_bgr, transform).to(device)

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
    hand_detector = HandBoxDetector(
        detector=args.hand_detector,
        yolo_model_path=args.hand_det_model_path,
        yolo_imgsz=args.hand_det_imgsz,
        yolo_conf=args.hand_det_conf,
        yolo_iou=args.hand_det_iou,
        box_expand_ratio=args.box_expand_ratio,
        min_area_ratio=args.hand_min_area_ratio,
    )

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
    skipped_no_hand = 0

    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "image_name",
                "pred_class",
                "confidence",
                "hand_box_found",
                "box_x1",
                "box_y1",
                "box_x2",
                "box_y2",
                "detector",
                "source_path",
                "output_path",
            ]
        )

        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[WARN] Cannot read image: {image_path}")
                continue

            box_result = hand_detector.detect(frame)
            hand_box = box_result.box

            if hand_box is None and args.require_hand_box:
                skipped_no_hand += 1
                print(f"[SKIP] No hand box: {image_path.name}")
                continue

            if hand_box is None:
                roi = frame
                box_vals = ("", "", "", "")
            else:
                x1, y1, x2, y2 = hand_box
                roi = frame[y1:y2, x1:x2]
                box_vals = (x1, y1, x2, y2)

            pred_idx, confidence = predict_single(
                model=model,
                transform=transform,
                image_bgr=roi,
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
                    bool(hand_box is not None),
                    box_vals[0],
                    box_vals[1],
                    box_vals[2],
                    box_vals[3],
                    box_result.method,
                    str(image_path),
                    str(destination),
                ]
            )

            if hand_box is None:
                print(f"[PRED] {image_path.name} -> {pred_class} ({confidence:.4f}) [full image]")
            else:
                print(f"[PRED] {image_path.name} -> {pred_class} ({confidence:.4f}) [hand box]")

    print("\n[DONE] Demo inference finished")
    print(f"Model   : {args.model_path}")
    print(f"Device  : {device}")
    print(f"Detector: {hand_detector.active_detector}")
    print(f"Demo dir: {args.demo_dir}")
    print(f"CSV     : {output_csv}")
    print("[SUMMARY]")
    for class_name in class_names:
        print(f"- {class_name}: {summary_count[class_name]}")
    if skipped_no_hand > 0:
        print(f"- skipped(no hand box): {skipped_no_hand}")


if __name__ == "__main__":
    main()
