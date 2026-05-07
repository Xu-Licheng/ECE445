#!/usr/bin/env python3
"""Build an image dataset by extracting frames from gesture videos.

Preferred layout:
    --video_dir/<class_name>/*.mp4

All videos under the same class folder are merged into one class in the output
dataset. For backward compatibility, a flat layout where each root video file
is one class is also supported.
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
        help=(
            "Video root directory. Preferred: video/<class_name>/*.mp4 "
            "(one folder per class, multiple videos allowed). "
            "Backward-compatible flat layout is also supported."
        ),
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


def is_supported_video(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS


def list_videos(video_dir: Path) -> list[Path]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    videos = [path for path in video_dir.iterdir() if is_supported_video(path)]
    return sorted(videos)


def list_videos_recursive(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_supported_video(path))


def ensure_unique_labels(raw_names: list[str], source_kind: str) -> None:
    label_to_raw: dict[str, list[str]] = {}
    for raw in raw_names:
        label = sanitize_label(raw)
        label_to_raw.setdefault(label, []).append(raw)

    duplicated = {label: names for label, names in label_to_raw.items() if len(names) > 1}
    if duplicated:
        details = "; ".join(
            f"{label} <- {', '.join(names)}" for label, names in sorted(duplicated.items())
        )
        raise ValueError(
            f"Detected duplicated class labels after sanitization for {source_kind}: {details}"
        )


def discover_class_sources(video_dir: Path) -> tuple[list[tuple[str, list[Path]]], str]:
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    class_dirs = sorted(path for path in video_dir.iterdir() if path.is_dir())
    folder_sources: list[tuple[str, list[Path]]] = []
    for class_dir in class_dirs:
        videos = list_videos_recursive(class_dir)
        if videos:
            folder_sources.append((class_dir.name, videos))

    if folder_sources:
        ensure_unique_labels([name for name, _ in folder_sources], "class folders")
        class_sources = [
            (sanitize_label(class_name), videos)
            for class_name, videos in sorted(folder_sources, key=lambda item: item[0].lower())
        ]
        return class_sources, "folder"

    flat_videos = list_videos(video_dir)
    if flat_videos:
        ensure_unique_labels([video.stem for video in flat_videos], "video filenames")
        class_sources = [(sanitize_label(video.stem), [video]) for video in flat_videos]
        return class_sources, "flat"

    raise RuntimeError(
        "No supported videos found. Expected either "
        "video/<class_name>/*.mp4 (preferred) or flat videos directly under video/."
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    class_sources, source_mode = discover_class_sources(args.video_dir)
    if source_mode == "folder":
        print("[INFO] Using folder-based class discovery: video/<class_name>/*.mp4")
    else:
        print("[INFO] Using flat class discovery (backward compatibility mode).")

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "frames_manifest.csv"
    summary_path = args.output_dir / "extract_summary.csv"

    total_saved = 0
    class_saved_counts: dict[str, int] = {class_name: 0 for class_name, _ in class_sources}
    summary_rows: list[tuple[str, str, int]] = []

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(
            [
                "class_name",
                "video_rel_path",
                "frame_index",
                "timestamp_sec",
                "image_path",
            ]
        )

        for class_name, class_videos in class_sources:
            class_dir = args.output_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for video_path in class_videos:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[WARN] Cannot open video: {video_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 30.0

                frame_index = 0
                saved_from_video = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    if frame_index % args.sample_every_n_frames == 0:
                        if (
                            args.max_frames_per_class > 0
                            and class_saved_counts[class_name] >= args.max_frames_per_class
                        ):
                            break

                        if args.resize_width > 0 and args.resize_height > 0:
                            frame = cv2.resize(
                                frame,
                                (args.resize_width, args.resize_height),
                                interpolation=cv2.INTER_AREA,
                            )

                        image_name = (
                            f"{class_name}_{class_saved_counts[class_name]:06d}.{args.image_ext}"
                        )
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
                            relative_video_path = video_path.relative_to(args.video_dir)
                            writer.writerow(
                                [
                                    class_name,
                                    str(relative_video_path).replace("\\", "/"),
                                    frame_index,
                                    f"{timestamp_sec:.6f}",
                                    str(relative_image_path).replace("\\", "/"),
                                ]
                            )
                            saved_from_video += 1
                            class_saved_counts[class_name] += 1
                            total_saved += 1
                        else:
                            print(f"[WARN] Failed to write image: {image_path}")

                    frame_index += 1

                cap.release()

                relative_video_path = video_path.relative_to(args.video_dir)
                summary_rows.append(
                    (
                        class_name,
                        str(relative_video_path).replace("\\", "/"),
                        saved_from_video,
                    )
                )
                print(
                    f"[INFO] {relative_video_path} -> class '{class_name}', "
                    f"saved {saved_from_video} images "
                    f"(class total: {class_saved_counts[class_name]})"
                )

                if (
                    args.max_frames_per_class > 0
                    and class_saved_counts[class_name] >= args.max_frames_per_class
                ):
                    print(
                        f"[INFO] Reached max_frames_per_class={args.max_frames_per_class} "
                        f"for class '{class_name}', skipping remaining videos in this class."
                    )
                    break

    with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.writer(summary_file)
        writer.writerow(["class_name", "video_rel_path", "saved_images"])
        writer.writerows(summary_rows)

    print(f"[DONE] Dataset extraction finished. Total images: {total_saved}")
    print(f"[DONE] Manifest: {manifest_path}")
    print(f"[DONE] Summary : {summary_path}")


if __name__ == "__main__":
    main()
