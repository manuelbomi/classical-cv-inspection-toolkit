"""Camera tamper / occlusion detection using classical background modelling.

Two operational failure modes are detected, both defined relative to a
rolling background model that is built up from "normal" frames:

* ``camera_blocked`` -- something is covering or blocking the lens (a hand,
  a cloth, a spray-painted lens, a cap placed in front of it). This shows up
  as a sudden, large, spatially *uniform* change together with a collapse
  in edge density -- a blocked view has almost no edges left, because
  whatever is covering the lens is usually close, out of focus, and
  featureless relative to the scene it replaces.

* ``camera_moved`` -- the camera (or its mount) has been bumped, re-aimed,
  or its housing rotated. This shows up as a large structural change from
  the background that *persists* across several consecutive frames but,
  unlike blocking, still contains a normal (non-collapsed) amount of edge
  detail -- it is simply a different, still-detailed, scene.

The implementation is intentionally simple and needs no training data:
a running-average background model (``cv2.accumulateWeighted``), absolute
frame differencing against that background, a plain standard-deviation
check (a uniform/blank frame has ~zero variance), and a Canny edge-density
heuristic. See the README's "Why this matters" section for the operational
motivation (monitoring the health of the camera/pipeline itself, not just
what it sees).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class TamperDetectionParams:
    """Tunable parameters for :class:`TamperDetector`."""

    # Learning rate for the running-average background model. Smaller means
    # the background adapts more slowly to legitimate slow scene changes
    # (e.g. gradual daylight change).
    alpha: float = 0.05

    # A pixel is considered "changed" if it differs from the background by
    # more than this many intensity levels (0-255).
    diff_threshold: int = 25

    # "camera_blocked" thresholds.
    blocked_diff_ratio: float = 0.5
    blocked_std_threshold: float = 12.0
    blocked_edge_ratio: float = 0.25

    # "camera_moved" thresholds: a structural change has to be large AND
    # persist for this many consecutive frames before being confirmed, to
    # avoid flagging brief transient motion (e.g. someone walking through).
    moved_diff_ratio: float = 0.35
    moved_sustained_frames: int = 5

    # Number of frames required before the background model is considered
    # trustworthy enough to make tamper decisions against.
    min_background_frames: int = 3


class TamperDetector:
    """Stateful, frame-by-frame camera tamper/occlusion detector.

    Usage::

        detector = TamperDetector()
        for frame in frames:
            result = detector.update(frame)
            if result["tampered"]:
                handle(result["reason"])
    """

    def __init__(self, params: Optional[TamperDetectionParams] = None):
        self.params = params or TamperDetectionParams()
        self._background: Optional[np.ndarray] = None
        self._baseline_edge_density: Optional[float] = None
        self._frame_count = 0
        self._moved_streak = 0

    @property
    def is_established(self) -> bool:
        """Whether enough frames have been seen to trust the background."""
        return self._frame_count >= self.params.min_background_frames

    def reset(self) -> None:
        """Discard all state and start learning a background from scratch."""
        params = self.params
        self.__init__(params)

    def update(self, frame: np.ndarray) -> dict:
        """Process one new frame and return a tamper-status dictionary.

        Returns a dict with keys: ``tampered`` (bool), ``reason`` (one of
        ``None``, ``"camera_blocked"``, ``"camera_moved"``), ``diff_ratio``,
        ``edge_density``, ``std``, and ``frame_count``.
        """
        params = self.params
        gray = _to_gray(frame).astype(np.float32)
        edge_density = _edge_density(frame)

        if self._background is None:
            # First frame ever: nothing to compare against yet.
            self._background = gray.copy()
            self._baseline_edge_density = edge_density
            self._frame_count = 1
            return _result(False, None, 0.0, edge_density, self._frame_count)

        background_u8 = self._background.astype(np.uint8)
        current_u8 = gray.astype(np.uint8)
        diff = cv2.absdiff(current_u8, background_u8)
        diff_ratio = float(np.mean(diff > params.diff_threshold))
        std = float(current_u8.std())

        baseline_edges = self._baseline_edge_density or 1e-6
        edge_collapsed = edge_density < baseline_edges * params.blocked_edge_ratio

        blocked = (
            self.is_established
            and diff_ratio > params.blocked_diff_ratio
            and (std < params.blocked_std_threshold or edge_collapsed)
        )

        moved = False
        if self.is_established and not blocked and diff_ratio > params.moved_diff_ratio:
            self._moved_streak += 1
            if self._moved_streak >= params.moved_sustained_frames:
                moved = True
        else:
            self._moved_streak = 0

        tampered = blocked or moved
        reason = "camera_blocked" if blocked else ("camera_moved" if moved else None)

        if not tampered:
            # Only fold "normal" frames into the background model, so a
            # sustained occlusion/move doesn't get absorbed as the new
            # normal while it is actively being flagged.
            cv2.accumulateWeighted(gray, self._background, params.alpha)
            self._baseline_edge_density = (
                (1 - params.alpha) * baseline_edges + params.alpha * edge_density
            )

        self._frame_count += 1

        return _result(tampered, reason, diff_ratio, edge_density, self._frame_count, std=std)


def _result(
    tampered: bool,
    reason: Optional[str],
    diff_ratio: float,
    edge_density: float,
    frame_count: int,
    std: Optional[float] = None,
) -> dict:
    return {
        "tampered": bool(tampered),
        "reason": reason,
        "diff_ratio": diff_ratio,
        "edge_density": edge_density,
        "std": std,
        "frame_count": frame_count,
    }


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _edge_density(image: np.ndarray) -> float:
    gray = _to_gray(image)
    edges = cv2.Canny(gray, 50, 150)
    return float(np.mean(edges > 0))
