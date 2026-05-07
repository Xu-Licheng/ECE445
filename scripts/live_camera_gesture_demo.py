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

from gesture_debounce import GestureDebounceConfig, GestureDebouncer
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
        default=0.6,
        help="Minimum top-1 confidence required to enter a new gesture output.",
    )
    parser.add_argument(
        "--hold_confidence",
        type=float,
        default=None,
        help="Lower hysteresis threshold for keeping the current gesture. Default: min_confidence - 0.10.",
    )
    parser.add_argument(
        "--min_top_margin",
        type=float,
        default=0.10,
        help="Minimum probability gap between top-1 and top-2 classes.",
    )
    parser.add_argument(
        "--stable_frames",
        type=int,
        default=3,
        help="Require the same valid top-1 class for this many frames before output.",
    )
    parser.add_argument(
        "--min_response_seconds",
        type=float,
        default=0.5,
        help="Minimum time between output label changes.",
    )
    parser.add_argument(
        "--hold_last_seconds",
        type=float,
        default=0.3,
        help="Keep the last valid gesture for this long when the current prediction is uncertain.",
    )
    parser.add_argument(
        "--no_hand_timeout_seconds",
        type=float,
        default=0.2,
        help="Keep the last output during short hand-box dropouts before showing no_hand_label.",
    )
    parser.add_argument(
        "--default_label",
        type=str,
        default="default",
        help="Fallback label for low confidence when uncertain_label is not one of the trained classes.",
    )
    parser.add_argument(
        "--uncertain_label",
        type=str,
        default="uncertain",
        help="Use this label for low-confidence output if it exists in the checkpoint class list.",
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


def resolve_hold_confidence(args: argparse.Namespace) -> float:
    if args.hold_confidence is not None:
        return args.hold_confidence
    return max(0.0, args.min_confidence - 0.10)


def resolve_low_confidence_label(
    class_names: list[str],
    uncertain_label: str,
    default_label: str,
) -> str:
    if uncertain_label in class_names:
        return uncertain_label
    return default_label


def validate_debounce_args(args: argparse.Namespace) -> None:
    hold_confidence = resolve_hold_confidence(args)
    if args.smoothing_window < 1:
        raise ValueError("--smoothing_window must be >= 1")
    if not (0.0 <= args.min_confidence <= 1.0):
        raise ValueError("--min_confidence must be in [0, 1]")
    if not (0.0 <= hold_confidence <= 1.0):
        raise ValueError("--hold_confidence must be in [0, 1]")
    if hold_confidence > args.min_confidence:
        raise ValueError("--hold_confidence must be <= --min_confidence")
    if not (0.0 <= args.min_top_margin <= 1.0):
        raise ValueError("--min_top_margin must be in [0, 1]")
    if args.stable_frames < 1:
        raise ValueError("--stable_frames must be >= 1")
    if args.min_response_seconds < 0.0:
        raise ValueError("--min_response_seconds must be >= 0")
    if args.hold_last_seconds < 0.0:
        raise ValueError("--hold_last_seconds must be >= 0")
    if args.no_hand_timeout_seconds < 0.0:
        raise ValueError("--no_hand_timeout_seconds must be >= 0")
    if not args.default_label:
        raise ValueError("--default_label must be non-empty")
    if not args.no_hand_label:
        raise ValueError("--no_hand_label must be non-empty")
    if not args.uncertain_label:
        raise ValueError("--uncertain_label must be non-empty")


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


def top_two_prediction(
    probs: torch.Tensor,
    class_names: list[str],
) -> tuple[str, float, float]:
    top_count = min(2, int(probs.numel()))
    top_values, top_indices = torch.topk(probs, k=top_count)

    pred_idx = int(top_indices[0].item())
    confidence = float(top_values[0].item())
    second_confidence = float(top_values[1].item()) if top_count > 1 else 0.0
    return class_names[pred_idx], confidence, second_confidence


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
    validate_debounce_args(args)

    gesture_model_path = resolve_gesture_model_path(args)
    hand_det_model_path = resolve_hand_det_model_path(args)

    device = resolve_device(args.device)
    model, class_names, image_size = load_checkpoint(gesture_model_path, device)
    hold_confidence = resolve_hold_confidence(args)
    low_confidence_label = resolve_low_confidence_label(
        class_names=class_names,
        uncertain_label=args.uncertain_label,
        default_label=args.default_label,
    )
    debouncer = GestureDebouncer(
        GestureDebounceConfig(
            min_confidence=args.min_confidence,
            hold_confidence=hold_confidence,
            min_top_margin=args.min_top_margin,
            stable_frames=args.stable_frames,
            min_response_seconds=args.min_response_seconds,
            hold_last_seconds=args.hold_last_seconds,
            no_hand_timeout_seconds=args.no_hand_timeout_seconds,
            low_confidence_label=low_confidence_label,
            no_hand_label=args.no_hand_label,
        )
    )
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
    print(
        "[INFO] Debounce: "
        f"min_conf={args.min_confidence:.2f}, "
        f"hold_conf={hold_confidence:.2f}, "
        f"top_margin={args.min_top_margin:.2f}, "
        f"stable_frames={args.stable_frames}, "
        f"min_response={args.min_response_seconds:.2f}s, "
        f"hold_last={args.hold_last_seconds:.2f}s, "
        f"no_hand_timeout={args.no_hand_timeout_seconds:.2f}s, "
        f"low_conf_label={low_confidence_label}"
    )
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
                now = time.perf_counter()
                debounce_result = debouncer.update(
                    raw_label=None,
                    raw_confidence=0.0,
                    second_confidence=0.0,
                    hand_detected=False,
                    now=now,
                )
                if debounce_result.hand_missing_timed_out:
                    prob_history.clear()
            else:
                x1, y1, x2, y2 = hand_box
                hand_roi = frame[y1:y2, x1:x2]
                probs = predict_probs(model, hand_roi, transform, device)
                prob_history.append(probs)

                smooth_probs = torch.stack(list(prob_history), dim=0).mean(dim=0)
                raw_label, raw_confidence, second_confidence = top_two_prediction(
                    smooth_probs,
                    class_names,
                )
                now = time.perf_counter()
                debounce_result = debouncer.update(
                    raw_label=raw_label,
                    raw_confidence=raw_confidence,
                    second_confidence=second_confidence,
                    hand_detected=True,
                    now=now,
                )

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

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            vis_frame = draw_overlay(
                frame,
                debounce_result.label,
                debounce_result.confidence,
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
