#!/usr/bin/env python3
"""Build gesture classification dataset by cropping hand ROIs with a trained hand detector.

Input layout:
    dataset/raw_frames/<class_name>/*.jpg

Output layout:
    dataset/gesture_crops/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2

from hand_box_detector import HandBoxDetector


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop hand ROIs from class images using a trained hand detector."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dataset/raw_frames"),
        help="Input dataset directory with class subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/gesture_crops"),
        help="Output dataset directory for cropped hand images.",
    )
    parser.add_argument(
        "--hand_det_model_path",
        type=Path,
        required=True,
        help="Path to trained hand detector model (.pt/.onnx).",
    )
    parser.add_argument(
        "--detector",
        type=str,
        choices=["yolo", "auto"],
        default="yolo",
        help="Hand detector backend. Recommended: yolo.",
    )
    parser.add_argument("--hand_det_imgsz", type=int, default=320)
    parser.add_argument("--hand_det_conf", type=float, default=0.25)
    parser.add_argument("--hand_det_iou", type=float, default=0.45)
    parser.add_argument("--box_expand_ratio", type=float, default=1.25)
    parser.add_argument(
        "--require_hand_box",
        action="store_true",
        help="Skip image if no hand box is detected.",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=Path("dataset/gesture_crops/crop_summary.csv"),
        help="Path to output summary CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir before building cropped dataset.",
    )
    return parser.parse_args()


def collect_class_dirs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")
    class_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {input_dir}")
    return class_dirs


def collect_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    if not args.hand_det_model_path.exists():
        raise FileNotFoundError(f"Hand detector model not found: {args.hand_det_model_path}")

    class_dirs = collect_class_dirs(args.input_dir)
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(args.summary_csv)

    detector = HandBoxDetector(
        detector=args.detector,
        yolo_model_path=args.hand_det_model_path,
        yolo_imgsz=args.hand_det_imgsz,
        yolo_conf=args.hand_det_conf,
        yolo_iou=args.hand_det_iou,
        box_expand_ratio=args.box_expand_ratio,
    )

    total_images = 0
    total_saved = 0
    total_skipped = 0

    with args.summary_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "class_name",
                "image_rel_path",
                "hand_box_found",
                "detector_method",
                "detector_score",
                "output_rel_path",
                "status",
            ]
        )

        for class_dir in class_dirs:
            class_name = class_dir.name
            image_paths = collect_images(class_dir)
            if not image_paths:
                print(f"[WARN] No images in class folder: {class_dir}")
                continue

            class_saved = 0
            class_skipped = 0
            out_class_dir = args.output_dir / class_name
            out_class_dir.mkdir(parents=True, exist_ok=True)

            for image_path in image_paths:
                total_images += 1
                rel_path = image_path.relative_to(args.input_dir)

                frame = cv2.imread(str(image_path))
                if frame is None:
                    class_skipped += 1
                    total_skipped += 1
                    writer.writerow(
                        [
                            class_name,
                            str(rel_path).replace("\\", "/"),
                            False,
                            detector.active_detector,
                            "",
                            "",
                            "read_failed",
                        ]
                    )
                    continue

                box_result = detector.detect(frame)
                hand_box = box_result.box

                if hand_box is None and args.require_hand_box:
                    class_skipped += 1
                    total_skipped += 1
                    writer.writerow(
                        [
                            class_name,
                            str(rel_path).replace("\\", "/"),
                            False,
                            box_result.method,
                            f"{box_result.score:.6f}",
                            "",
                            "no_hand",
                        ]
                    )
                    continue

                if hand_box is None:
                    roi = frame
                else:
                    x1, y1, x2, y2 = hand_box
                    roi = frame[y1:y2, x1:x2]

                out_image_path = out_class_dir / image_path.name
                ensure_parent(out_image_path)
                write_ok = cv2.imwrite(str(out_image_path), roi)
                if not write_ok:
                    class_skipped += 1
                    total_skipped += 1
                    writer.writerow(
                        [
                            class_name,
                            str(rel_path).replace("\\", "/"),
                            hand_box is not None,
                            box_result.method,
                            f"{box_result.score:.6f}",
                            str(out_image_path.relative_to(args.output_dir)).replace("\\", "/"),
                            "write_failed",
                        ]
                    )
                    continue

                class_saved += 1
                total_saved += 1
                writer.writerow(
                    [
                        class_name,
                        str(rel_path).replace("\\", "/"),
                        hand_box is not None,
                        box_result.method,
                        f"{box_result.score:.6f}",
                        str(out_image_path.relative_to(args.output_dir)).replace("\\", "/"),
                        "ok",
                    ]
                )

            print(
                f"[INFO] class '{class_name}': "
                f"input={len(image_paths)}, saved={class_saved}, skipped={class_skipped}"
            )

    print("[DONE] Gesture crop dataset build finished")
    print(f"[DONE] input_images  : {total_images}")
    print(f"[DONE] saved_images  : {total_saved}")
    print(f"[DONE] skipped_images: {total_skipped}")
    print(f"[DONE] output_dir    : {args.output_dir}")
    print(f"[DONE] summary_csv   : {args.summary_csv}")


if __name__ == "__main__":
    main()
