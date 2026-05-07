#!/usr/bin/env python3
"""Small runtime debounce state machine for live gesture outputs."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GestureDebounceConfig:
    min_confidence: float = 0.6
    hold_confidence: float = 0.5
    min_top_margin: float = 0.1
    stable_frames: int = 3
    min_response_seconds: float = 0.5
    hold_last_seconds: float = 0.3
    no_hand_timeout_seconds: float = 0.2
    low_confidence_label: str = "default"
    no_hand_label: str = "no_hand"


@dataclass(frozen=True)
class GestureDebounceResult:
    label: str
    confidence: float
    raw_label: str | None
    raw_confidence: float
    raw_margin: float
    changed: bool
    is_valid_gesture: bool
    hand_detected: bool
    hand_missing_timed_out: bool
    reason: str


class GestureDebouncer:
    """Debounce frame-by-frame gesture predictions into a stable output label."""

    def __init__(self, config: GestureDebounceConfig) -> None:
        self.config = config
        self._validate_config(config)

        self._output_label: str | None = None
        self._output_confidence = 0.0
        self._output_is_valid = False
        self._last_output_time: float | None = None

        self._last_valid_label: str | None = None
        self._last_valid_confidence = 0.0
        self._last_valid_time: float | None = None

        self._pending_label: str | None = None
        self._pending_count = 0
        self._last_hand_time: float | None = None

    def update(
        self,
        raw_label: str | None,
        raw_confidence: float,
        second_confidence: float = 0.0,
        hand_detected: bool = True,
        now: float | None = None,
    ) -> GestureDebounceResult:
        if now is None:
            now = time.perf_counter()

        raw_confidence = float(raw_confidence)
        second_confidence = float(second_confidence)
        raw_margin = raw_confidence - second_confidence if raw_label is not None else 0.0

        if not hand_detected:
            return self._handle_no_hand(
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
            )

        self._last_hand_time = now

        if raw_label is None:
            self._reset_pending()
            return self._handle_invalid_prediction(
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                reason="missing_prediction",
            )

        margin_ok = raw_margin >= self.config.min_top_margin
        same_as_valid_output = self._output_is_valid and raw_label == self._output_label

        if same_as_valid_output and raw_confidence >= self.config.hold_confidence and margin_ok:
            self._reset_pending()
            return self._commit_or_keep(
                desired_label=raw_label,
                desired_confidence=raw_confidence,
                valid_output=True,
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                hand_detected=True,
                hand_missing_timed_out=False,
                reason="held_by_hysteresis",
            )

        if raw_confidence < self.config.min_confidence:
            self._reset_pending()
            return self._handle_invalid_prediction(
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                reason="low_confidence",
            )

        if not margin_ok:
            self._reset_pending()
            return self._handle_invalid_prediction(
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                reason="low_top_margin",
            )

        if raw_label == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = raw_label
            self._pending_count = 1

        if self._pending_count < self.config.stable_frames:
            return self._handle_invalid_prediction(
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                reason=f"waiting_stable_frames:{self._pending_count}/{self.config.stable_frames}",
            )

        return self._commit_or_keep(
            desired_label=raw_label,
            desired_confidence=raw_confidence,
            valid_output=True,
            now=now,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            raw_margin=raw_margin,
            hand_detected=True,
            hand_missing_timed_out=False,
            reason="stable_prediction",
        )

    def _handle_no_hand(
        self,
        now: float,
        raw_label: str | None,
        raw_confidence: float,
        raw_margin: float,
    ) -> GestureDebounceResult:
        self._reset_pending()

        hand_missing_timed_out = (
            self._last_hand_time is None
            or now - self._last_hand_time >= self.config.no_hand_timeout_seconds
        )
        if not hand_missing_timed_out and self._output_label is not None:
            return GestureDebounceResult(
                label=self._output_label,
                confidence=self._output_confidence,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                changed=False,
                is_valid_gesture=self._output_is_valid,
                hand_detected=False,
                hand_missing_timed_out=False,
                reason="no_hand_timeout_hold",
            )

        return self._commit_or_keep(
            desired_label=self.config.no_hand_label,
            desired_confidence=0.0,
            valid_output=False,
            now=now,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            raw_margin=raw_margin,
            hand_detected=False,
            hand_missing_timed_out=True,
            reason="no_hand",
        )

    def _handle_invalid_prediction(
        self,
        now: float,
        raw_label: str | None,
        raw_confidence: float,
        raw_margin: float,
        reason: str,
    ) -> GestureDebounceResult:
        if (
            self._last_valid_label is not None
            and self._last_valid_time is not None
            and now - self._last_valid_time <= self.config.hold_last_seconds
        ):
            return self._commit_or_keep(
                desired_label=self._last_valid_label,
                desired_confidence=self._last_valid_confidence,
                valid_output=True,
                remember_valid=False,
                now=now,
                raw_label=raw_label,
                raw_confidence=raw_confidence,
                raw_margin=raw_margin,
                hand_detected=True,
                hand_missing_timed_out=False,
                reason=f"{reason}_hold_last",
            )

        return self._commit_or_keep(
            desired_label=self.config.low_confidence_label,
            desired_confidence=raw_confidence,
            valid_output=False,
            now=now,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            raw_margin=raw_margin,
            hand_detected=True,
            hand_missing_timed_out=False,
            reason=reason,
        )

    def _commit_or_keep(
        self,
        desired_label: str,
        desired_confidence: float,
        valid_output: bool,
        now: float,
        raw_label: str | None,
        raw_confidence: float,
        raw_margin: float,
        hand_detected: bool,
        hand_missing_timed_out: bool,
        reason: str,
        remember_valid: bool = True,
    ) -> GestureDebounceResult:
        changed = False

        if self._output_label is None:
            self._set_output(
                desired_label,
                desired_confidence,
                valid_output,
                now,
                remember_valid,
            )
            changed = True
        elif desired_label == self._output_label:
            self._output_confidence = float(desired_confidence)
            self._output_is_valid = valid_output
            if remember_valid:
                self._remember_valid_output(now)
        elif self._can_change_output(now):
            self._set_output(
                desired_label,
                desired_confidence,
                valid_output,
                now,
                remember_valid,
            )
            changed = True
        else:
            reason = f"rate_limited:{reason}->{desired_label}"

        return GestureDebounceResult(
            label=self._output_label or desired_label,
            confidence=self._output_confidence,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            raw_margin=raw_margin,
            changed=changed,
            is_valid_gesture=self._output_is_valid,
            hand_detected=hand_detected,
            hand_missing_timed_out=hand_missing_timed_out,
            reason=reason,
        )

    def _set_output(
        self,
        label: str,
        confidence: float,
        is_valid: bool,
        now: float,
        remember_valid: bool = True,
    ) -> None:
        self._output_label = label
        self._output_confidence = float(confidence)
        self._output_is_valid = is_valid
        self._last_output_time = now
        if remember_valid:
            self._remember_valid_output(now)

    def _remember_valid_output(self, now: float) -> None:
        if not self._output_is_valid or self._output_label is None:
            return
        self._last_valid_label = self._output_label
        self._last_valid_confidence = self._output_confidence
        self._last_valid_time = now

    def _can_change_output(self, now: float) -> bool:
        if self._last_output_time is None:
            return True
        return now - self._last_output_time >= self.config.min_response_seconds

    def _reset_pending(self) -> None:
        self._pending_label = None
        self._pending_count = 0

    @staticmethod
    def _validate_config(config: GestureDebounceConfig) -> None:
        if not (0.0 <= config.min_confidence <= 1.0):
            raise ValueError("min_confidence must be in [0, 1]")
        if not (0.0 <= config.hold_confidence <= 1.0):
            raise ValueError("hold_confidence must be in [0, 1]")
        if config.hold_confidence > config.min_confidence:
            raise ValueError("hold_confidence must be <= min_confidence")
        if not (0.0 <= config.min_top_margin <= 1.0):
            raise ValueError("min_top_margin must be in [0, 1]")
        if config.stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        if config.min_response_seconds < 0.0:
            raise ValueError("min_response_seconds must be >= 0")
        if config.hold_last_seconds < 0.0:
            raise ValueError("hold_last_seconds must be >= 0")
        if config.no_hand_timeout_seconds < 0.0:
            raise ValueError("no_hand_timeout_seconds must be >= 0")
        if not config.low_confidence_label:
            raise ValueError("low_confidence_label must be non-empty")
        if not config.no_hand_label:
            raise ValueError("no_hand_label must be non-empty")
