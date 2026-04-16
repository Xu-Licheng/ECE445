#!/usr/bin/env python3
"""Run real-time gesture classification from webcam frames.

Usage example:
python scripts/live_camera_gesture_demo.py \
    --gesture_model_path training_runs/run_gesture/best_model.pth \
    --hand_det_model_path training_runs_hand_det/run_hand_det/weights/best.pt \
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

from hand_box_detector import HandBoxDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time dual-model demo: hand detector + gesture classifier."
    )
    parser.add_argument(
        "--gesture_model_path",
        type=Path,
        default=None,
        help="Gesture classifier checkpoint (.pth). If omitted, auto-pick latest from training_runs.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Deprecated alias of --gesture_model_path for backward compatibility.",
    )
    parser.add_argument(
        "--hand_det_model_path",
        type=Path,
        default=None,
        help="YOLO hand detector model path (.pt/.onnx). If omitted, auto-pick latest best.pt under training_runs_hand_det.",
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
        "--no_hand_label",
        type=str,
        default="no_hand",
        help="Overlay label when no hand box is detected.",
    )
    return parser.parse_args()


def resolve_gesture_model_path(args: argparse.Namespace) -> Path:
    if args.gesture_model_path is not None:
        return args.gesture_model_path
    if args.model_path is not None:
        return args.model_path

    candidates = sorted(Path("training_runs").glob("*/best_model.pth"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "Cannot find gesture model automatically. "
            "Pass --gesture_model_path explicitly or train gesture model first."
        )
    return candidates[-1]


def resolve_hand_det_model_path(args: argparse.Namespace) -> Path:
    if args.hand_det_model_path is not None:
        return args.hand_det_model_path

    candidates = sorted(
        Path("training_runs_hand_det").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            "Cannot find hand detector model automatically. "
            "Pass --hand_det_model_path explicitly or train hand detector first."
        )
    return candidates[-1]


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
    detector_name: str,
    hand_box_found: bool,
) -> "cv2.typing.MatLike":
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]

    bar_h = 112
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
    cv2.putText(
        frame,
        f"Box: {'yes' if hand_box_found else 'no'} | Detector: {detector_name}",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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

    gesture_model_path = resolve_gesture_model_path(args)
    hand_det_model_path = resolve_hand_det_model_path(args)

    device = resolve_device(args.device)
    model, class_names, image_size = load_checkpoint(gesture_model_path, device)
    transform = make_eval_transform(image_size)
    hand_detector = HandBoxDetector(
        detector="yolo",
        yolo_model_path=hand_det_model_path,
        yolo_imgsz=args.hand_det_imgsz,
        yolo_conf=args.hand_det_conf,
        yolo_iou=args.hand_det_iou,
        box_expand_ratio=args.box_expand_ratio,
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {args.camera_index}. "
            "Try another index such as 1 or 2."
        )

    print("[INFO] Live demo started")
    print(f"[INFO] Gesture model: {gesture_model_path}")
    print(f"[INFO] Hand model   : {hand_det_model_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Detector: {hand_detector.active_detector}")
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

            box_result = hand_detector.detect(frame)
            hand_box = box_result.box

            if hand_box is None:
                prob_history.clear()
                label = args.no_hand_label
                confidence = 0.0
            else:
                x1, y1, x2, y2 = hand_box
                hand_roi = frame[y1:y2, x1:x2]
                probs = predict_probs(model, hand_roi, transform, device)
                prob_history.append(probs)

                smooth_probs = torch.stack(list(prob_history), dim=0).mean(dim=0)
                pred_idx = int(torch.argmax(smooth_probs).item())
                confidence = float(smooth_probs[pred_idx].item())

                if confidence < args.min_confidence:
                    label = "uncertain"
                else:
                    label = class_names[pred_idx]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    f"hand box ({box_result.method})",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            vis_frame = draw_overlay(
                frame,
                label,
                confidence,
                fps,
                device,
                hand_detector.active_detector,
                hand_box is not None,
            )
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
