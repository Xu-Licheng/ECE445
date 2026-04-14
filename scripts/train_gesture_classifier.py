#!/usr/bin/env python3
"""Train a gesture classifier from extracted frame images.

Expected dataset layout:
    dataset/raw_frames/
        class_a/*.jpg
        class_b/*.jpg
        class_c/*.jpg
        class_d/*.jpg

Features:
1) Stratified train/val/test split
2) Training logs written to file
3) Curves for loss/accuracy and confusion matrix figure
4) Save best and final checkpoints
5) Optional ONNX export for deployment pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    class_name: str
    label_idx: int


class FrameImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, record.label_idx


class FileLogger:
    def __init__(self, log_path: Path) -> None:
        self._file = log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {message}"
        print(line)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train gesture image classifier.")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("dataset/raw_frames"),
        help="Dataset directory with class subfolders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("training_runs"),
        help="Directory to store logs, figures and checkpoints.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="Optional run name. If empty, timestamp-based name is used.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--image_size", type=int, default=160)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Use 0 on Windows first for maximum compatibility.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda | cuda:0",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained MobileNetV3-Small backbone.",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=8,
        help="Stop if val acc does not improve for N epochs. 0 disables it.",
    )
    parser.add_argument(
        "--export_onnx",
        action="store_true",
        help="Export best checkpoint to ONNX at run end.",
    )
    parser.add_argument("--onnx_opset", type=int, default=11)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ratios = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(r <= 0 for r in ratios):
        raise ValueError("train/val/test ratios must all be > 0")

    ratio_sum = sum(ratios)
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum:.6f}"
        )

    if args.image_size < 32:
        raise ValueError("--image_size should be >= 32")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def collect_records(dataset_dir: Path) -> tuple[list[ImageRecord], list[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError(f"No class folders found in: {dataset_dir}")

    records: list[ImageRecord] = []
    class_names: list[str] = []

    for class_dir in class_dirs:
        image_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not image_paths:
            continue

        label_idx = len(class_names)
        class_names.append(class_dir.name)

        for image_path in image_paths:
            records.append(
                ImageRecord(
                    image_path=image_path,
                    class_name=class_dir.name,
                    label_idx=label_idx,
                )
            )

    if len(class_names) < 2:
        raise RuntimeError(
            "Need at least two non-empty class folders for classification training"
        )

    if len(records) < len(class_names) * 3:
        raise RuntimeError(
            "Too few images for train/val/test split. Extract more frames first."
        )

    return records, class_names


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    labels = [record.label_idx for record in records]
    indices = np.arange(len(records))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed,
        shuffle=True,
    )

    train_val_labels = [labels[idx] for idx in train_val_idx]
    val_share_in_train_val = val_ratio / (train_ratio + val_ratio)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_share_in_train_val,
        stratify=train_val_labels,
        random_state=seed,
        shuffle=True,
    )

    return list(train_idx), list(val_idx), list(test_idx)


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    collect_predictions: bool = False,
) -> tuple[float, float, list[int], list[int]]:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_count += batch_size

        if collect_predictions:
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(preds.detach().cpu().tolist())

    avg_loss = total_loss / max(total_count, 1)
    avg_acc = total_correct / max(total_count, 1)
    return avg_loss, avg_acc, all_targets, all_predictions


def save_split_manifest(
    records: list[ImageRecord],
    dataset_dir: Path,
    train_idx: list[int],
    val_idx: list[int],
    test_idx: list[int],
    output_path: Path,
) -> None:
    split_by_index: dict[int, str] = {}
    for idx in train_idx:
        split_by_index[idx] = "train"
    for idx in val_idx:
        split_by_index[idx] = "val"
    for idx in test_idx:
        split_by_index[idx] = "test"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "class_name", "label", "split"])
        for idx, record in enumerate(records):
            rel_path = record.image_path.relative_to(dataset_dir)
            writer.writerow(
                [
                    str(rel_path).replace("\\", "/"),
                    record.class_name,
                    record.label_idx,
                    split_by_index[idx],
                ]
            )


def save_history_csv(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return
    headers = [
        "epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
        "epoch_seconds",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_training_curves(history: list[dict[str, float]], output_path: Path) -> None:
    if not history:
        return

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    val_acc = [row["val_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_title("Loss Curve")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="train_acc")
    axes[1].plot(epochs, val_acc, label="val_acc")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy Curve")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_confusion(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Test Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    run_name = args.run_name.strip() or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = FileLogger(run_dir / "train.log")
    try:
        device = resolve_device(args.device)
        logger.log(f"Using device: {device}")
        logger.log(f"Arguments: {vars(args)}")

        records, class_names = collect_records(args.dataset_dir)
        logger.log(f"Found {len(records)} images in {len(class_names)} classes")
        logger.log(f"Class names: {class_names}")

        try:
            train_idx, val_idx, test_idx = stratified_split(
                records=records,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )
        except ValueError as exc:
            raise RuntimeError(
                "Stratified split failed. Usually this means a class has too few images. "
                "Extract more frames or reduce test/val ratios."
            ) from exc

        logger.log(
            "Split counts -> "
            f"train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}"
        )

        save_split_manifest(
            records=records,
            dataset_dir=args.dataset_dir,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            output_path=run_dir / "split_manifest.csv",
        )

        with (run_dir / "classes.txt").open("w", encoding="utf-8") as file:
            for name in class_names:
                file.write(name + "\n")

        train_transform, eval_transform = make_transforms(args.image_size)

        train_records = [records[idx] for idx in train_idx]
        val_records = [records[idx] for idx in val_idx]
        test_records = [records[idx] for idx in test_idx]

        train_dataset = FrameImageDataset(train_records, train_transform)
        val_dataset = FrameImageDataset(val_records, eval_transform)
        test_dataset = FrameImageDataset(test_records, eval_transform)

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

        model = build_model(num_classes=len(class_names), pretrained=args.pretrained)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

        history: list[dict[str, float]] = []
        best_val_acc = -1.0
        no_improve_epochs = 0

        logger.log("Start training loop")
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc, _, _ = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )
            val_loss, val_acc, _, _ = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )

            scheduler.step(val_acc)
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - epoch_start

            history_row = {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(lr_now),
                "epoch_seconds": float(elapsed),
            }
            history.append(history_row)

            logger.log(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"lr={lr_now:.6f}, sec={elapsed:.2f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "class_names": class_names,
                        "image_size": args.image_size,
                    },
                    run_dir / "best_model.pth",
                )
                logger.log("Saved best_model.pth")
            else:
                no_improve_epochs += 1

            if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
                logger.log(
                    "Early stopping triggered due to no val accuracy improvement "
                    f"for {no_improve_epochs} epochs"
                )
                break

        save_history_csv(history, run_dir / "metrics.csv")
        plot_training_curves(history, run_dir / "training_curves.png")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
            },
            run_dir / "final_model.pth",
        )
        logger.log("Saved final_model.pth")

        best_model_path = run_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.log("Loaded best checkpoint for test evaluation")

        test_loss, test_acc, y_true, y_pred = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            collect_predictions=True,
        )
        logger.log(f"Test metrics | loss={test_loss:.4f}, accuracy={test_acc:.4f}")

        plot_confusion(y_true, y_pred, class_names, run_dir / "confusion_matrix.png")

        report = classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            output_dict=True,
            digits=4,
            zero_division=0,
        )
        with (run_dir / "classification_report.json").open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_val_acc": best_val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "num_classes": len(class_names),
                    "num_images": len(records),
                    "classes": class_names,
                },
                file,
                indent=2,
            )

        if args.export_onnx:
            onnx_path = run_dir / "best_model.onnx"
            dummy = torch.randn(1, 3, args.image_size, args.image_size, device=device)
            model.eval()
            torch.onnx.export(
                model,
                dummy,
                str(onnx_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=args.onnx_opset,
            )
            logger.log(f"Exported ONNX model: {onnx_path}")

        logger.log(f"Run finished. Outputs saved to: {run_dir}")

    finally:
        logger.close()


if __name__ == "__main__":
    main()
