#!/usr/bin/env python3
"""Train a hand detector (single-class) with Ultralytics YOLO."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hand detector using YOLO.")
    parser.add_argument(
        "--data_yaml",
        type=Path,
        default=Path("dataset/hand_det/yolo/dataset.yaml"),
        help="Path to YOLO dataset.yaml.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Base YOLO model for fine-tuning.",
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("training_runs_hand_det"),
        help="Training project output root.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name. If empty, timestamp name is used.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export best model to ONNX after training.",
    )
    parser.add_argument("--onnx_opset", type=int, default=12)
    parser.add_argument(
        "--simplify_onnx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to simplify ONNX graph during export.",
    )
    return parser.parse_args()


def resolve_run_name(run_name: str) -> str:
    if run_name.strip():
        return run_name.strip()
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ultralytics is required for hand detector training. "
            "Install it manually with: pip install ultralytics"
        ) from exc

    if not args.data_yaml.exists():
        raise FileNotFoundError(f"data_yaml not found: {args.data_yaml}")

    run_name = resolve_run_name(args.run_name)
    args.project.mkdir(parents=True, exist_ok=True)

    print("[INFO] Start hand detector training")
    print(f"[INFO] data_yaml: {args.data_yaml}")
    print(f"[INFO] model    : {args.model}")
    print(f"[INFO] device   : {args.device}")

    detector = YOLO(args.model)
    detector.train(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=args.device,
        project=str(args.project),
        name=run_name,
        pretrained=True,
        verbose=True,
    )

    save_dir = Path(detector.trainer.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    if not best_pt.exists():
        raise RuntimeError(f"Training finished but best.pt not found: {best_pt}")

    exported_onnx: str | None = None
    if args.export_onnx:
        print("[INFO] Export best.pt to ONNX")
        export_model = YOLO(str(best_pt))
        exported_onnx = export_model.export(
            format="onnx",
            imgsz=args.imgsz,
            dynamic=False,
            simplify=args.simplify_onnx,
            opset=args.onnx_opset,
        )

    summary = {
        "run_name": run_name,
        "save_dir": str(save_dir),
        "best_pt": str(best_pt),
        "last_pt": str(last_pt),
        "exported_onnx": str(exported_onnx) if exported_onnx is not None else None,
        "data_yaml": str(args.data_yaml),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
    }

    summary_path = save_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[DONE] Hand detector training finished")
    print(f"[DONE] save_dir     : {save_dir}")
    print(f"[DONE] best_pt      : {best_pt}")
    if exported_onnx is not None:
        print(f"[DONE] exported_onnx: {exported_onnx}")
    print(f"[DONE] summary      : {summary_path}")


if __name__ == "__main__":
    main()
