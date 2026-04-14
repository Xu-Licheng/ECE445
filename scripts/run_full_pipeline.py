#!/usr/bin/env python3
"""Run end-to-end gesture pipeline: frame extraction -> training -> summary.

This script orchestrates existing scripts so the whole workflow can be started
with one command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command pipeline for gesture dataset extraction and training."
    )
    parser.add_argument("--video_dir", type=Path, default=Path("video"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/raw_frames"))
    parser.add_argument("--output_dir", type=Path, default=Path("training_runs"))
    parser.add_argument("--run_name", type=str, default="")

    parser.add_argument(
        "--resolution",
        type=int,
        default=160,
        help="Square resolution used for extracted frames and training input.",
    )
    parser.add_argument("--sample_every_n_frames", type=int, default=1)
    parser.add_argument("--max_frames_per_class", type=int, default=0)
    parser.add_argument("--image_ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg_quality", type=int, default=95)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--early_stop_patience", type=int, default=8)

    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pretrained MobileNetV3-Small for training.",
    )
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--onnx_opset", type=int, default=11)

    parser.add_argument(
        "--skip_extract",
        action="store_true",
        help="Skip frame extraction and train using existing dataset_dir.",
    )
    parser.add_argument(
        "--overwrite_dataset",
        action="store_true",
        help="Delete dataset_dir before extracting frames.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def validate_args(args: argparse.Namespace) -> None:
    if args.resolution < 32:
        raise ValueError("--resolution must be >= 32")
    if args.sample_every_n_frames < 1:
        raise ValueError("--sample_every_n_frames must be >= 1")
    if args.max_frames_per_class < 0:
        raise ValueError("--max_frames_per_class must be >= 0")
    if not (1 <= args.jpg_quality <= 100):
        raise ValueError("--jpg_quality must be between 1 and 100")

    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("train/val/test ratios must all be > 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")


def load_summary(summary_path: Path) -> dict | None:
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    validate_args(args)

    project_root = Path(__file__).resolve().parent.parent
    scripts_dir = project_root / "scripts"

    run_name = args.run_name.strip() or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not args.skip_extract:
        extract_cmd = [
            sys.executable,
            str(scripts_dir / "extract_frames_dataset.py"),
            "--video_dir",
            str(args.video_dir),
            "--output_dir",
            str(args.dataset_dir),
            "--sample_every_n_frames",
            str(args.sample_every_n_frames),
            "--resize_width",
            str(args.resolution),
            "--resize_height",
            str(args.resolution),
            "--image_ext",
            args.image_ext,
            "--jpg_quality",
            str(args.jpg_quality),
        ]
        if args.max_frames_per_class > 0:
            extract_cmd.extend(["--max_frames_per_class", str(args.max_frames_per_class)])
        if args.overwrite_dataset:
            extract_cmd.append("--overwrite")

        print("[STEP] 1/2 Extracting frames...")
        run_command(extract_cmd, cwd=project_root)
    else:
        print("[STEP] 1/2 Skipped frame extraction (--skip_extract)")

    train_cmd = [
        sys.executable,
        str(scripts_dir / "train_gesture_classifier.py"),
        "--dataset_dir",
        str(args.dataset_dir),
        "--output_dir",
        str(args.output_dir),
        "--run_name",
        run_name,
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--test_ratio",
        str(args.test_ratio),
        "--image_size",
        str(args.resolution),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--early_stop_patience",
        str(args.early_stop_patience),
        "--onnx_opset",
        str(args.onnx_opset),
    ]
    if args.pretrained:
        train_cmd.append("--pretrained")
    if args.export_onnx:
        train_cmd.append("--export_onnx")

    print("[STEP] 2/2 Training and evaluating...")
    run_command(train_cmd, cwd=project_root)

    run_dir = args.output_dir / run_name
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "classification_report.json"
    curves_path = run_dir / "training_curves.png"
    confusion_path = run_dir / "confusion_matrix.png"
    log_path = run_dir / "train.log"

    summary = load_summary(summary_path)
    print("\n[RESULT] Pipeline finished")
    print(f"Run directory: {run_dir}")
    print(f"Training log : {log_path}")
    print(f"Curves image : {curves_path}")
    print(f"Confusion mat: {confusion_path}")
    print(f"Report json  : {report_path}")

    if summary is not None:
        test_acc = summary.get("test_acc")
        best_val_acc = summary.get("best_val_acc")
        num_images = summary.get("num_images")
        num_classes = summary.get("num_classes")
        print("\n[SUMMARY]")
        print(f"Resolution   : {args.resolution} x {args.resolution}")
        print(f"Images       : {num_images}")
        print(f"Classes      : {num_classes}")
        if best_val_acc is not None:
            print(f"Best val acc : {best_val_acc:.4f}")
        if test_acc is not None:
            print(f"Test acc     : {test_acc:.4f}")
    else:
        print(f"[WARN] Summary file not found: {summary_path}")


if __name__ == "__main__":
    main()
