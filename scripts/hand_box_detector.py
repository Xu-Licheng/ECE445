#!/usr/bin/env python3
"""Utilities for hand bounding-box detection.

Detection strategy:
1) Prefer MediaPipe Hands when available.
2) Fallback to simple skin-color contour detection in OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    mp = None

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    YOLO = None


@dataclass(frozen=True)
class HandBoxResult:
    box: tuple[int, int, int, int] | None
    method: str
    score: float


class HandBoxDetector:
    def __init__(
        self,
        detector: str = "auto",
        yolo_model_path: str | Path | None = None,
        yolo_imgsz: int = 320,
        yolo_conf: float = 0.25,
        yolo_iou: float = 0.45,
        yolo_max_det: int = 2,
        box_expand_ratio: float = 1.25,
        min_area_ratio: float = 0.01,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 1,
    ) -> None:
        if detector not in {"auto", "yolo", "mediapipe", "skin"}:
            raise ValueError("detector must be one of: auto, yolo, mediapipe, skin")
        if box_expand_ratio < 1.0:
            raise ValueError("box_expand_ratio must be >= 1.0")
        if not (0.0 < min_area_ratio <= 1.0):
            raise ValueError("min_area_ratio must be in (0, 1]")
        if yolo_imgsz < 64:
            raise ValueError("yolo_imgsz must be >= 64")
        if not (0.0 <= yolo_conf <= 1.0):
            raise ValueError("yolo_conf must be in [0, 1]")
        if not (0.0 <= yolo_iou <= 1.0):
            raise ValueError("yolo_iou must be in [0, 1]")

        self.box_expand_ratio = box_expand_ratio
        self.min_area_ratio = min_area_ratio
        self._requested_detector = detector
        self._yolo_model_path = Path(yolo_model_path) if yolo_model_path else None
        self._yolo_imgsz = yolo_imgsz
        self._yolo_conf = yolo_conf
        self._yolo_iou = yolo_iou
        self._yolo_max_det = yolo_max_det

        self._active_detector = self._resolve_active_detector(detector)
        self._hands = None
        self._yolo_detector = None

        if self._active_detector == "yolo":
            if YOLO is None:
                raise RuntimeError("ultralytics is not installed for yolo detector")
            if self._yolo_model_path is None:
                raise ValueError("yolo_model_path must be provided when using yolo detector")
            if not self._yolo_model_path.exists():
                raise FileNotFoundError(f"Hand detector model not found: {self._yolo_model_path}")
            self._yolo_detector = YOLO(str(self._yolo_model_path))

        if self._active_detector == "mediapipe":
            hands_cls = mp.solutions.hands.Hands
            self._hands = hands_cls(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    @property
    def active_detector(self) -> str:
        return self._active_detector

    def detect(self, frame_bgr: "cv2.typing.MatLike") -> HandBoxResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return HandBoxResult(box=None, method=self._active_detector, score=0.0)

        if self._active_detector == "yolo":
            return self._detect_with_yolo(frame_bgr)

        if self._active_detector == "mediapipe":
            return self._detect_with_mediapipe(frame_bgr)

        return self._detect_with_skin(frame_bgr)

    def _resolve_active_detector(self, detector: str) -> str:
        if detector == "yolo":
            return "yolo"
        if detector == "skin":
            return "skin"
        if detector == "mediapipe":
            if mp is None:
                raise RuntimeError(
                    "--hand_detector mediapipe requested but mediapipe is not installed."
                )
            return "mediapipe"
        if self._yolo_model_path is not None:
            if YOLO is None:
                raise RuntimeError(
                    "--hand_detector auto with model path requires ultralytics installation."
                )
            return "yolo"
        if mp is not None:
            return "mediapipe"
        raise RuntimeError(
            "--hand_detector auto could not find available backend. "
            "Provide --hand_det_model_path for yolo or install mediapipe."
        )

    def _detect_with_yolo(self, frame_bgr: "cv2.typing.MatLike") -> HandBoxResult:
        if self._yolo_detector is None:
            return HandBoxResult(box=None, method="yolo", score=0.0)

        height, width = frame_bgr.shape[:2]
        outputs = self._yolo_detector.predict(
            source=frame_bgr,
            verbose=False,
            imgsz=self._yolo_imgsz,
            conf=self._yolo_conf,
            iou=self._yolo_iou,
            max_det=self._yolo_max_det,
        )
        if not outputs:
            return HandBoxResult(box=None, method="yolo", score=0.0)

        boxes = outputs[0].boxes
        if boxes is None or len(boxes) == 0:
            return HandBoxResult(box=None, method="yolo", score=0.0)

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()

        best_idx = -1
        best_area = -1.0
        for idx, box in enumerate(xyxy):
            x1, y1, x2, y2 = box.tolist()
            area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
            if area > best_area:
                best_area = area
                best_idx = idx

        if best_idx < 0:
            return HandBoxResult(box=None, method="yolo", score=0.0)

        x1, y1, x2, y2 = xyxy[best_idx].tolist()
        box = _expand_and_clip_box(
            int(round(x1)),
            int(round(y1)),
            int(round(x2)),
            int(round(y2)),
            width,
            height,
            self.box_expand_ratio,
        )
        return HandBoxResult(box=box, method="yolo", score=float(confs[best_idx]))

    def _detect_with_mediapipe(self, frame_bgr: "cv2.typing.MatLike") -> HandBoxResult:
        if self._hands is None:
            return HandBoxResult(box=None, method="mediapipe", score=0.0)

        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(frame_rgb)

        if not results.multi_hand_landmarks:
            return HandBoxResult(box=None, method="mediapipe", score=0.0)

        best_box: tuple[int, int, int, int] | None = None
        best_area = -1
        best_score = 0.0

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            xs = [lm.x for lm in hand_landmarks.landmark]
            ys = [lm.y for lm in hand_landmarks.landmark]

            x1 = int(min(xs) * width)
            y1 = int(min(ys) * height)
            x2 = int(max(xs) * width)
            y2 = int(max(ys) * height)

            box = _expand_and_clip_box(
                x1,
                y1,
                x2,
                y2,
                width,
                height,
                self.box_expand_ratio,
            )
            area = (box[2] - box[0]) * (box[3] - box[1])

            score = 1.0
            if results.multi_handedness and idx < len(results.multi_handedness):
                score = float(results.multi_handedness[idx].classification[0].score)

            if area > best_area:
                best_area = area
                best_box = box
                best_score = score

        return HandBoxResult(box=best_box, method="mediapipe", score=best_score)

    def _detect_with_skin(self, frame_bgr: "cv2.typing.MatLike") -> HandBoxResult:
        height, width = frame_bgr.shape[:2]

        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandBoxResult(box=None, method="skin", score=0.0)

        best = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(best))
        if area < self.min_area_ratio * float(width * height):
            return HandBoxResult(box=None, method="skin", score=0.0)

        x, y, w_box, h_box = cv2.boundingRect(best)
        box = _expand_and_clip_box(
            x,
            y,
            x + w_box,
            y + h_box,
            width,
            height,
            self.box_expand_ratio,
        )
        score = area / float(width * height)
        return HandBoxResult(box=box, method="skin", score=score)


def _expand_and_clip_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    expand_ratio: float,
) -> tuple[int, int, int, int]:
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    cx = x1 + box_w / 2.0
    cy = y1 + box_h / 2.0

    new_w = box_w * expand_ratio
    new_h = box_h * expand_ratio

    nx1 = int(max(0, round(cx - new_w / 2.0)))
    ny1 = int(max(0, round(cy - new_h / 2.0)))
    nx2 = int(min(width, round(cx + new_w / 2.0)))
    ny2 = int(min(height, round(cy + new_h / 2.0)))

    if nx2 <= nx1:
        nx2 = min(width, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(height, ny1 + 1)
    return nx1, ny1, nx2, ny2
