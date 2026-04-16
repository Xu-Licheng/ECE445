#!/usr/bin/env python3
"""Automatic hand-box labeling with GroundingDINO.

This script generates YOLO-format hand detection labels from images.
It is designed as a high-quality pseudo-label stage before manual review.

Pipeline per image:
1) Detect hands with GroundingDINO text prompt.
2) Optionally detect faces and suppress overlapping hand boxes.
3) Apply confidence threshold, area filtering, and NMS.
4) Save image copy and YOLO labels for downstream detector training.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.ops import nms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-label hand boxes using GroundingDINO and export YOLO labels."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dataset/raw_frames"),
        help="Root directory containing input images.",
    )
    parser.add_argument(
        "--output_images_dir",
        type=Path,
        default=Path("dataset/hand_det/images_all"),
        help="Output image root copied for detector training.",
    )
    parser.add_argument(
        "--output_labels_dir",
        type=Path,
        default=Path("dataset/hand_det/labels_all"),
        help="Output YOLO label root.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("dataset/hand_det/autolabel_summary.csv"),
        help="Summary CSV output path.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face model id for GroundingDINO.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="hand . palm . fingers .",
        help="Text prompt for hand detection.",
    )
    parser.add_argument(
        "--box_threshold",
        type=float,
        default=0.30,
        help="Box confidence threshold for prompt detection.",
    )
    parser.add_argument(
        "--text_threshold",
        type=float,
        default=0.25,
        help="Text confidence threshold for prompt detection.",
    )
    parser.add_argument(
        "--suppress_faces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress hand boxes that overlap detected face boxes.",
    )
    parser.add_argument(
        "--face_prompt",
        type=str,
        default="face . human face .",
        help="Prompt used for face detection in suppression step.",
    )
    parser.add_argument(
        "--face_box_threshold",
        type=float,
        default=0.35,
        help="Face detection box threshold.",
    )
    parser.add_argument(
        "--face_text_threshold",
        type=float,
        default=0.25,
        help="Face detection text threshold.",
    )
    parser.add_argument(
        "--face_iou_threshold",
        type=float,
        default=0.35,
        help="Discard hand boxes with IoU >= this threshold against face boxes.",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold for hand boxes.",
    )
    parser.add_argument(
        "--max_hands",
        type=int,
        default=2,
        help="Maximum number of kept hand boxes per image.",
    )
    parser.add_argument(
        "--box_expand_ratio",
        type=float,
        default=1.10,
        help="Expand each box by this ratio before label export.",
    )
    parser.add_argument(
        "--min_box_area_ratio",
        type=float,
        default=0.002,
        help="Drop boxes smaller than this image-area ratio.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in current torch build.")
    return device


def collect_images(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def detect_boxes(
    image: Image.Image,
    prompt: str,
    processor: Any,
    model: Any,
    device: torch.device,
    box_threshold: float,
    text_threshold: float,
) -> list[Box]:
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]], device=device)
    post_fn = processor.post_process_grounded_object_detection
    post_params = inspect.signature(post_fn).parameters

    post_kwargs: dict[str, Any] = {
        "text_threshold": text_threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in post_params:
        post_kwargs["box_threshold"] = box_threshold
    elif "threshold" in post_params:
        post_kwargs["threshold"] = box_threshold
    else:
        raise RuntimeError(
            "Unsupported GroundingDINO processor API: missing threshold argument"
        )

    results = post_fn(outputs, inputs["input_ids"], **post_kwargs)[0]

    boxes: list[Box] = []
    for xyxy, score in zip(results["boxes"], results["scores"]):
        x1, y1, x2, y2 = [float(v) for v in xyxy.tolist()]
        boxes.append(Box(x1=x1, y1=y1, x2=x2, y2=y2, score=float(score.item())))
    return boxes


def iou(a: Box, b: Box) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def suppress_face_overlap(hand_boxes: list[Box], face_boxes: list[Box], iou_threshold: float) -> list[Box]:
    if not face_boxes:
        return hand_boxes

    kept: list[Box] = []
    for hand in hand_boxes:
        overlaps_face = any(iou(hand, face) >= iou_threshold for face in face_boxes)
        if not overlaps_face:
            kept.append(hand)
    return kept


def apply_nms(boxes: list[Box], iou_threshold: float) -> list[Box]:
    if not boxes:
        return []

    xyxy = torch.tensor([[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32)
    scores = torch.tensor([b.score for b in boxes], dtype=torch.float32)
    keep_idx = nms(xyxy, scores, iou_threshold=iou_threshold).tolist()
    return [boxes[idx] for idx in keep_idx]


def expand_box(box: Box, width: int, height: int, ratio: float) -> Box:
    w = max(1.0, box.x2 - box.x1)
    h = max(1.0, box.y2 - box.y1)
    cx = box.x1 + w / 2.0
    cy = box.y1 + h / 2.0

    nw = w * ratio
    nh = h * ratio

    x1 = max(0.0, cx - nw / 2.0)
    y1 = max(0.0, cy - nh / 2.0)
    x2 = min(float(width), cx + nw / 2.0)
    y2 = min(float(height), cy + nh / 2.0)
    return Box(x1=x1, y1=y1, x2=x2, y2=y2, score=box.score)


def to_yolo_line(box: Box, width: int, height: int) -> str:
    bw = max(1e-6, box.x2 - box.x1)
    bh = max(1e-6, box.y2 - box.y1)
    cx = box.x1 + bw / 2.0
    cy = box.y1 + bh / 2.0

    return (
        "0 "
        f"{cx / width:.6f} "
        f"{cy / height:.6f} "
        f"{bw / width:.6f} "
        f"{bh / height:.6f}"
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers is required for auto labeling. "
            "Install it manually with: pip install transformers accelerate"
        ) from exc

    images = collect_images(args.input_dir)
    if not images:
        raise RuntimeError(f"No images found in: {args.input_dir}")

    print(f"[INFO] Found images: {len(images)}")
    print(f"[INFO] Loading model: {args.model_id}")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id)
    model.to(device)
    model.eval()

    kept_total = 0
    face_suppressed_total = 0

    ensure_parent(args.summary_csv)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "image_rel_path",
                "num_hands_raw",
                "num_hands_after_face_suppress",
                "num_hands_final",
                "status",
            ]
        )

        for idx, image_path in enumerate(images, start=1):
            rel = image_path.relative_to(args.input_dir)
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            hand_boxes = detect_boxes(
                image=image,
                prompt=args.prompt,
                processor=processor,
                model=model,
                device=device,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
            )
            num_raw = len(hand_boxes)

            num_after_face = num_raw
            if args.suppress_faces and hand_boxes:
                face_boxes = detect_boxes(
                    image=image,
                    prompt=args.face_prompt,
                    processor=processor,
                    model=model,
                    device=device,
                    box_threshold=args.face_box_threshold,
                    text_threshold=args.face_text_threshold,
                )
                hand_boxes_after_face = suppress_face_overlap(
                    hand_boxes,
                    face_boxes,
                    iou_threshold=args.face_iou_threshold,
                )
                face_suppressed_total += max(0, len(hand_boxes) - len(hand_boxes_after_face))
                hand_boxes = hand_boxes_after_face
                num_after_face = len(hand_boxes)

            hand_boxes = [
                b
                for b in hand_boxes
                if (b.x2 - b.x1) * (b.y2 - b.y1) >= args.min_box_area_ratio * float(width * height)
            ]

            hand_boxes = apply_nms(hand_boxes, iou_threshold=args.nms_iou)
            hand_boxes = sorted(hand_boxes, key=lambda b: b.score, reverse=True)[: args.max_hands]
            hand_boxes = [expand_box(b, width=width, height=height, ratio=args.box_expand_ratio) for b in hand_boxes]

            out_image_path = args.output_images_dir / rel
            out_label_path = args.output_labels_dir / rel.with_suffix(".txt")
            ensure_parent(out_image_path)
            ensure_parent(out_label_path)
            shutil.copy2(image_path, out_image_path)

            with out_label_path.open("w", encoding="utf-8") as label_file:
                for b in hand_boxes:
                    label_file.write(to_yolo_line(b, width=width, height=height) + "\n")

            kept_total += len(hand_boxes)
            status = "ok" if hand_boxes else "no_hand"
            writer.writerow([str(rel).replace("\\", "/"), num_raw, num_after_face, len(hand_boxes), status])

            if idx % 100 == 0 or idx == len(images):
                print(f"[INFO] Processed {idx}/{len(images)}")

    print("[DONE] Auto labeling finished")
    print(f"[DONE] images_out: {args.output_images_dir}")
    print(f"[DONE] labels_out: {args.output_labels_dir}")
    print(f"[DONE] summary: {args.summary_csv}")
    print(f"[DONE] total_kept_boxes: {kept_total}")
    print(f"[DONE] face_suppressed_boxes: {face_suppressed_total}")


if __name__ == "__main__":
    main()
