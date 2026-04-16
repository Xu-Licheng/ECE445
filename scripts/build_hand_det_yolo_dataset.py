#!/usr/bin/env python3
"""Build YOLO hand-detection dataset from auto-labeled outputs."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_path: Path
    rel_path: Path
    group: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build train/val/test YOLO dataset for hand detector."
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        default=Path("dataset/hand_det/images_all"),
        help="Auto-labeled image root.",
    )
    parser.add_argument(
        "--labels_dir",
        type=Path,
        default=Path("dataset/hand_det/labels_all"),
        help="Auto-labeled YOLO label root.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/hand_det/yolo"),
        help="Output YOLO dataset root.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class_name",
        type=str,
        default="hand",
        help="Single class name for detection.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir before writing.",
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r <= 0.0 for r in ratios):
        raise ValueError("train/val/test ratios must all be > 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")


def collect_samples(images_dir: Path, labels_dir: Path) -> list[Sample]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

    samples: list[Sample] = []
    for image_path in sorted(images_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue

        rel_path = image_path.relative_to(images_dir)
        label_path = labels_dir / rel_path.with_suffix(".txt")
        if not label_path.exists():
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")

        parts = rel_path.parts
        group = parts[0] if len(parts) > 1 else "default"
        samples.append(
            Sample(
                image_path=image_path,
                label_path=label_path,
                rel_path=rel_path,
                group=group,
            )
        )

    if not samples:
        raise RuntimeError("No samples found in images_dir")
    return samples


def split_group(samples: list[Sample], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)

    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.group, []).append(sample)

    split_map: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}

    for group_name, group_samples in grouped.items():
        rng.shuffle(group_samples)
        n = len(group_samples)

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        n_test = n - n_train - n_val

        if n_test < 1:
            n_test = 1
            if n_train > n_val:
                n_train = max(1, n_train - 1)
            else:
                n_val = max(1, n_val - 1)

        if n_train + n_val + n_test != n:
            n_train = max(1, n - n_val - n_test)

        train_slice = group_samples[:n_train]
        val_slice = group_samples[n_train : n_train + n_val]
        test_slice = group_samples[n_train + n_val :]

        if not test_slice:
            test_slice = val_slice[-1:]
            val_slice = val_slice[:-1] if len(val_slice) > 1 else val_slice

        split_map["train"].extend(train_slice)
        split_map["val"].extend(val_slice)
        split_map["test"].extend(test_slice)

        print(
            f"[INFO] Group {group_name}: train={len(train_slice)}, val={len(val_slice)}, test={len(test_slice)}"
        )

    return split_map


def copy_split_files(split_map: dict[str, list[Sample]], output_dir: Path) -> None:
    for split_name, split_samples in split_map.items():
        image_out = output_dir / "images" / split_name
        label_out = output_dir / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for sample in split_samples:
            dst_image = image_out / sample.rel_path.name
            dst_label = label_out / sample.rel_path.with_suffix(".txt").name
            shutil.copy2(sample.image_path, dst_image)
            shutil.copy2(sample.label_path, dst_label)


def write_dataset_yaml(output_dir: Path, class_name: str) -> Path:
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = (
        f"path: {output_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def write_manifest(split_map: dict[str, list[Sample]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "group", "image_rel_path"])
        for split_name in ["train", "val", "test"]:
            for sample in split_map[split_name]:
                writer.writerow([split_name, sample.group, str(sample.rel_path).replace("\\", "/")])


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    samples = collect_samples(args.images_dir, args.labels_dir)
    split_map = split_group(samples, args.train_ratio, args.val_ratio, args.seed)

    copy_split_files(split_map, args.output_dir)
    yaml_path = write_dataset_yaml(args.output_dir, args.class_name)
    write_manifest(split_map, args.output_dir / "split_manifest.csv")

    print("[DONE] YOLO hand detection dataset ready")
    print(f"[DONE] output_dir : {args.output_dir}")
    print(f"[DONE] dataset_yaml: {yaml_path}")
    print(
        "[DONE] split counts: "
        f"train={len(split_map['train'])}, "
        f"val={len(split_map['val'])}, "
        f"test={len(split_map['test'])}"
    )


if __name__ == "__main__":
    main()
