#!/usr/bin/env python3
"""Run real-time gesture classification from webcam frames.

Usage example:
python scripts/live_camera_gesture_demo.py \
    --model_path training_runs/run_xxx/best_model.pth \
    --device cuda:0
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time gesture classifier demo using webcam."
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pth), e.g. training_runs/run_xxx/best_model.pth",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--camera_index",
        type=int,
        default=0,
        help="Webcam index for OpenCV VideoCapture.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="Gesture Live Demo",
        help="Display window title.",
    )
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=5,
        help="Number of recent frames to smooth prediction probabilities.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.0,
        help="If confidence is below this threshold, label as uncertain.",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror camera image horizontally for a selfie-like preview.",
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


def frame_to_tensor(frame_bgr: "cv2.typing.MatLike", transform: transforms.Compose) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return transform(image).unsqueeze(0)


def predict_probs(
    model: nn.Module,
    frame_bgr: "cv2.typing.MatLike",
    transform: transforms.Compose,
    device: torch.device,
) -> torch.Tensor:
    tensor = frame_to_tensor(frame_bgr, transform).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.squeeze(0).detach().cpu()


def draw_overlay(
    frame_bgr: "cv2.typing.MatLike",
    label: str,
    confidence: float,
    fps: float,
    device: torch.device,
) -> "cv2.typing.MatLike":
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    bar_h = 90
    cv2.rectangle(frame, (0, 0), (w, bar_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        f"Pred: {label}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Conf: {confidence:.3f}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | Device: {device}",
        (12, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    if args.smoothing_window < 1:
        raise ValueError("--smoothing_window must be >= 1")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min_confidence must be in [0, 1]")

    device = resolve_device(args.device)
    model, class_names, image_size = load_checkpoint(args.model_path, device)
    transform = make_eval_transform(image_size)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {args.camera_index}. "
            "Try another index such as 1 or 2."
        )

    print("[INFO] Live demo started")
    print(f"[INFO] Model : {args.model_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Classes: {class_names}")
    print("[INFO] Press 'q' or ESC to quit")

    prob_history: deque[torch.Tensor] = deque(maxlen=args.smoothing_window)
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera frame read failed, stopping demo.")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            probs = predict_probs(model, frame, transform, device)
            prob_history.append(probs)

            smooth_probs = torch.stack(list(prob_history), dim=0).mean(dim=0)
            pred_idx = int(torch.argmax(smooth_probs).item())
            confidence = float(smooth_probs[pred_idx].item())

            if confidence < args.min_confidence:
                label = "uncertain"
            else:
                label = class_names[pred_idx]

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            vis_frame = draw_overlay(frame, label, confidence, fps, device)
            cv2.imshow(args.window_name, vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[DONE] Live demo exited")


if __name__ == "__main__":
    main()
