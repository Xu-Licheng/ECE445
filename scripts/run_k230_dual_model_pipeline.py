#!/usr/bin/env python3
"""Run full dual-model pipeline for K230 deployment preparation.

Stages:
1) Auto-label hand boxes with GroundingDINO.
2) Build YOLO hand detector dataset split.
3) Train hand detector and export ONNX.
4) Train gesture classifier.
5) Export gesture classifier static ONNX.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-command dual-model pipeline for K230 prep.")

    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--video_dir", type=Path, default=Path("video"))
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset/raw_frames"))

    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument("--epochs_gesture", type=int, default=25)
    parser.add_argument("--epochs_hand_det", type=int, default=80)
    parser.add_argument("--batch_gesture", type=int, default=32)
    parser.add_argument("--batch_hand_det", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--overwrite_dataset", action="store_true")

    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_auto_label", action="store_true")
    parser.add_argument("--skip_hand_det_train", action="store_true")
    parser.add_argument("--skip_gesture_train", action="store_true")

    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(command))
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def resolve_run_name(run_name: str) -> str:
    if run_name.strip():
        return run_name.strip()
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    scripts = root / "scripts"

    run_name = resolve_run_name(args.run_name)
    hand_det_project = root / "training_runs_hand_det"
    hand_det_run = f"hand_det_{run_name}"
    gesture_run = f"gesture_{run_name}"

    print("[INFO] K230 dual-model pipeline start")
    print(f"[INFO] project_root: {root}")
    print(f"[INFO] run_name    : {run_name}")

    if not args.skip_extract:
        extract_cmd = [
            sys.executable,
            str(scripts / "extract_frames_dataset.py"),
            "--video_dir",
            str(args.video_dir),
            "--output_dir",
            str(args.dataset_dir),
            "--sample_every_n_frames",
            "1",
            "--resize_width",
            str(args.resolution),
            "--resize_height",
            str(args.resolution),
            "--image_ext",
            "jpg",
            "--jpg_quality",
            "95",
        ]
        if args.overwrite_dataset:
            extract_cmd.append("--overwrite")

        print("[STEP] 1/5 Extract frames")
        run_command(extract_cmd, cwd=root)
    else:
        print("[STEP] 1/5 Skip extract")

    if not args.skip_auto_label:
        auto_label_cmd = [
            sys.executable,
            str(scripts / "auto_label_hand_boxes.py"),
            "--input_dir",
            str(args.dataset_dir),
            "--output_images_dir",
            "dataset/hand_det/images_all",
            "--output_labels_dir",
            "dataset/hand_det/labels_all",
            "--summary_csv",
            "dataset/hand_det/autolabel_summary.csv",
            "--device",
            args.device,
        ]

        build_yolo_cmd = [
            sys.executable,
            str(scripts / "build_hand_det_yolo_dataset.py"),
            "--images_dir",
            "dataset/hand_det/images_all",
            "--labels_dir",
            "dataset/hand_det/labels_all",
            "--output_dir",
            "dataset/hand_det/yolo",
            "--overwrite",
        ]

        print("[STEP] 2/5 Auto-label hand boxes")
        run_command(auto_label_cmd, cwd=root)
        print("[STEP] 2.5/5 Build YOLO hand-det dataset")
        run_command(build_yolo_cmd, cwd=root)
    else:
        print("[STEP] 2/5 Skip auto-label and dataset build")

    if not args.skip_hand_det_train:
        train_hand_cmd = [
            sys.executable,
            str(scripts / "train_hand_detector_yolo.py"),
            "--data_yaml",
            "dataset/hand_det/yolo/dataset.yaml",
            "--imgsz",
            "320",
            "--epochs",
            str(args.epochs_hand_det),
            "--batch",
            str(args.batch_hand_det),
            "--device",
            args.device,
            "--project",
            str(hand_det_project),
            "--run_name",
            hand_det_run,
            "--export_onnx",
        ]

        print("[STEP] 3/5 Train hand detector")
        run_command(train_hand_cmd, cwd=root)
    else:
        print("[STEP] 3/5 Skip hand-detector training")

    best_hand_pt = hand_det_project / hand_det_run / "weights" / "best.pt"

    if not args.skip_gesture_train:
        train_gesture_cmd = [
            sys.executable,
            str(scripts / "train_gesture_classifier.py"),
            "--dataset_dir",
            str(args.dataset_dir),
            "--output_dir",
            "training_runs",
            "--run_name",
            gesture_run,
            "--image_size",
            str(args.resolution),
            "--epochs",
            str(args.epochs_gesture),
            "--batch_size",
            str(args.batch_gesture),
            "--device",
            args.device,
            "--pretrained",
        ]

        print("[STEP] 4/5 Train gesture classifier")
        run_command(train_gesture_cmd, cwd=root)
    else:
        print("[STEP] 4/5 Skip gesture training")

    best_gesture_pth = root / "training_runs" / gesture_run / "best_model.pth"
    export_gesture_cmd = [
        sys.executable,
        str(scripts / "export_gesture_onnx_static.py"),
        "--model_path",
        str(best_gesture_pth),
        "--output_onnx",
        str(root / "training_runs" / gesture_run / "gesture_classifier_static.onnx"),
        "--image_size",
        str(args.resolution),
        "--opset",
        "12",
        "--device",
        "cpu",
    ]

    print("[STEP] 5/5 Export static ONNX for gesture model")
    run_command(export_gesture_cmd, cwd=root)

    print("\n[DONE] Dual-model pipeline finished")
    print(f"[DONE] hand detector best pt : {best_hand_pt}")
    print(f"[DONE] gesture best pth      : {best_gesture_pth}")
    print("[DONE] Next: convert both ONNX to K230 kmodel with nncase tooling")


if __name__ == "__main__":
    main()
