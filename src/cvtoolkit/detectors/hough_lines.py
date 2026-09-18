"""Alignment / edge detection using the probabilistic Hough Line Transform.

Wraps ``cv2.HoughLinesP`` (the probabilistic variant of the Hough line
transform, which returns line *segments* with real endpoints instead of
infinite lines) to find straight edges: a conveyor rail, a part edge, a
fiber, a shelf line, and similar alignment checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class LineDetectionParams:
    """Tunable parameters for :func:`detect_lines`.

    ``canny_low``/``canny_high`` control the edge map fed into the Hough
    transform; ``threshold`` is the minimum number of votes a line needs in
    Hough (parameter) space to be reported; ``min_line_length`` and
    ``max_line_gap`` control how segments are merged/discarded.
    """

    canny_low: int = 50
    canny_high: int = 150
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 40
    min_line_length: float = 40.0
    max_line_gap: float = 10.0


def detect_lines(
    image: np.ndarray, params: Optional[LineDetectionParams] = None
) -> List[dict]:
    """Detect straight line segments in ``image``.

    Returns a list of dictionaries with ``point1``, ``point2`` (pixel
    endpoints), ``length`` (pixels), and ``angle_deg`` (in
    ``(-180, 180]``, measured from the positive x-axis, image coordinates),
    sorted longest-first.
    """
    if params is None:
        params = LineDetectionParams()

    gray = _to_gray(image)
    edges = cv2.Canny(gray, params.canny_low, params.canny_high)

    theta = np.deg2rad(params.theta_deg)
    lines = cv2.HoughLinesP(
        edges,
        params.rho,
        theta,
        params.threshold,
        minLineLength=params.min_line_length,
        maxLineGap=params.max_line_gap,
    )

    results: List[dict] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            results.append(
                {
                    "point1": (int(x1), int(y1)),
                    "point2": (int(x2), int(y2)),
                    "length": length,
                    "angle_deg": angle,
                }
            )

    results.sort(key=lambda line: -line["length"])
    return results


def draw_lines(image: np.ndarray, lines: List[dict]) -> np.ndarray:
    """Return a copy of ``image`` with detected line segments drawn on it."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    for line in lines:
        cv2.line(annotated, line["point1"], line["point2"], (0, 255, 255), 2)
    return annotated


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
