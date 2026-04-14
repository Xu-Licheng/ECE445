#!/usr/bin/env python3
"""Build an image dataset by extracting frames from gesture videos.

Each video file under --video_dir is treated as one class. Frames are written to
--output_dir/<class_name>/ and metadata is saved to CSV files.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from videos to build an image classification dataset."
    )
    parser.add_argument(
        "--video_dir",
        type=Path,
        default=Path("video"),
        help="Directory containing source videos. One video equals one class.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/raw_frames"),
        help="Directory where extracted images are saved.",
    )
    parser.add_argument(
        "--sample_every_n_frames",
        type=int,
        default=1,
        help="Extract one image every N frames. Use 1 to save every frame.",
    )
    parser.add_argument(
        "--max_frames_per_class",
        type=int,
        default=0,
        help="Optional cap for saved frames per class. 0 means no cap.",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=0,
        help="Optional resize width. Must be used together with --resize_height.",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=0,
        help="Optional resize height. Must be used together with --resize_width.",
    )
    parser.add_argument(
        "--image_ext",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image extension.",
    )
    parser.add_argument(
        "--jpg_quality",
        type=int,
        default=95,
        help="JPEG quality (1-100). Only applies when --image_ext=jpg.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir before extraction.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.sample_every_n_frames < 1:
        raise ValueError("--sample_every_n_frames must be >= 1")
    if args.max_frames_per_class < 0:
        raise ValueError("--max_frames_per_class must be >= 0")
    if (args.resize_width > 0) != (args.resize_height > 0):
        raise ValueError("--resize_width and --resize_height must be set together")
    if not (1 <= args.jpg_quality <= 100):
        raise ValueError("--jpg_quality must be between 1 and 100")


def sanitize_label(name: str) -> str:
    label = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    label = re.sub(r"_+", "_", label).strip("_")
    return label or "class"


def list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    videos = [
        path
        for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def main() -> None:
    args = parse_args()
    validate_args(args)

    video_files = list_videos(args.video_dir)
    if not video_files:
        raise RuntimeError(f"No supported videos found in: {args.video_dir}")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "frames_manifest.csv"
    summary_path = args.output_dir / "extract_summary.csv"

    total_saved = 0
    summary_rows: list[tuple[str, str, int]] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "class_name",
                "video_file",
                "frame_index",
                "timestamp_sec",
                "image_path",
            ]
        )

        for video_path in video_files:
            class_name = sanitize_label(video_path.stem)
            class_dir = args.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"[WARN] Cannot open video: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0

            frame_index = 0
            saved_count = 0

            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_index % args.sample_every_n_frames == 0:
                    if args.resize_width > 0 and args.resize_height > 0:
                        frame = cv2.resize(
                            frame,
                            (args.resize_width, args.resize_height),
                            interpolation=cv2.INTER_AREA,
                        )

                    image_name = f"{class_name}_{saved_count:06d}.{args.image_ext}"
                    image_path = class_dir / image_name

                    if args.image_ext == "jpg":
                        write_ok = cv2.imwrite(
                            str(image_path),
                            frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpg_quality],
                        )
                    else:
                        write_ok = cv2.imwrite(str(image_path), frame)

                    if write_ok:
                        timestamp_sec = frame_index / fps
                        relative_image_path = image_path.relative_to(args.output_dir)
                        writer.writerow(
                            [
                                class_name,
                                video_path.name,
                                frame_index,
                                f"{timestamp_sec:.6f}",
                                str(relative_image_path).replace("\\", "/"),
                            ]
                        )
                        saved_count += 1
                        total_saved += 1
                    else:
                        print(f"[WARN] Failed to write image: {image_path}")

                    if (
                        args.max_frames_per_class > 0
                        and saved_count >= args.max_frames_per_class
                    ):
                        break

                frame_index += 1

            cap.release()

            summary_rows.append((class_name, video_path.name, saved_count))
            print(
                f"[INFO] {video_path.name} -> class '{class_name}', "
                f"saved {saved_count} images"
            )

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["class_name", "video_file", "saved_images"])
        writer.writerows(summary_rows)

    print(f"[DONE] Dataset extraction finished. Total images: {total_saved}")
    print(f"[DONE] Manifest: {manifest_path}")
    print(f"[DONE] Summary : {summary_path}")


if __name__ == "__main__":
    main()
