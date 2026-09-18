"""Round-object detection using the Hough Circle Transform.

Wraps ``cv2.HoughCircles`` to find round objects (bottles, vials, wheels,
pills, washers, ...) in an image and returns them as plain dictionaries so
callers do not need to know anything about OpenCV's raw output format.

See the README's "Design decisions" section for a plain-English explanation
of how the Hough transform actually works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CircleDetectionParams:
    """Tunable parameters for :func:`detect_circles`.

    These map directly onto ``cv2.HoughCircles`` arguments. Defaults are
    reasonable for medium-resolution frames (roughly 480p-1080p) with
    objects tens of pixels in radius; they will usually need re-tuning for a
    different camera, lens, or working distance (see the README's
    "Limitations & production hardening notes").
    """

    dp: float = 1.2
    min_dist: float = 20.0
    param1: float = 100.0
    param2: float = 30.0
    min_radius: int = 5
    max_radius: int = 0  # 0 means "no upper limit", per OpenCV's convention
    blur_kernel: int = 9  # 0 disables the pre-blur; must be odd otherwise


def detect_circles(
    image: np.ndarray, params: Optional[CircleDetectionParams] = None
) -> List[dict]:
    """Detect circular objects in ``image``.

    Parameters
    ----------
    image:
        A BGR or single-channel grayscale image (e.g. from ``cv2.imread`` or
        a video frame).
    params:
        Tunable Hough parameters. Sensible defaults are used if omitted.

    Returns
    -------
    A list of ``{"center": (x, y), "radius": r}`` dictionaries, one per
    detected circle, sorted left-to-right then top-to-bottom for
    deterministic output ordering.
    """
    if params is None:
        params = CircleDetectionParams()

    gray = _to_gray(image)
    if params.blur_kernel and params.blur_kernel > 0:
        kernel_size = params.blur_kernel | 1  # force odd
        gray = cv2.medianBlur(gray, kernel_size)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=params.dp,
        minDist=params.min_dist,
        param1=params.param1,
        param2=params.param2,
        minRadius=params.min_radius,
        maxRadius=params.max_radius,
    )

    results: List[dict] = []
    if circles is not None:
        for x, y, r in circles[0]:
            results.append({"center": (float(x), float(y)), "radius": float(r)})

    results.sort(key=lambda c: (c["center"][0], c["center"][1]))
    return results


def draw_circles(image: np.ndarray, circles: List[dict]) -> np.ndarray:
    """Return a copy of ``image`` with detected circles drawn on it."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    for circle in circles:
        center = (int(round(circle["center"][0])), int(round(circle["center"][1])))
        radius = int(round(circle["radius"]))
        cv2.circle(annotated, center, radius, (0, 255, 0), 2)
        cv2.circle(annotated, center, 2, (0, 0, 255), 3)
    return annotated


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
