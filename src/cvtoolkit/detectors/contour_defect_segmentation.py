"""Blob segmentation and defect flagging via thresholding + contours.

The approach is deliberately simple and fully deterministic:

1. Blur slightly to reduce sensor noise.
2. Threshold (Otsu's method by default) to get a binary foreground/background
   mask.
3. Clean the mask up with morphological opening/closing (removes speckle
   noise, fills small holes).
4. Find external contours ("blobs") and measure each one's area and
   bounding-box aspect ratio.
5. Flag any blob whose area or aspect ratio falls outside a configured
   "expected" range as a defect.

This is a good fit when the part being inspected has a known, roughly
fixed size and shape on-screen (fixed camera, fixed working distance) and
defects manifest as a wrong outline: cracks, chips, foreign material,
deformation, wrong part in the wrong slot, etc.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DefectSegmentationParams:
    """Tunable parameters for :func:`segment_and_flag_defects`."""

    blur_kernel: int = 5
    threshold_value: int = 0  # 0 -> use Otsu's automatic threshold
    invert: bool = False
    morph_kernel: int = 5
    morph_iterations: int = 2
    min_area: float = 50.0
    expected_area_range: Tuple[float, float] = (200.0, 5000.0)
    expected_aspect_ratio_range: Tuple[float, float] = (0.6, 1.7)


def segment_and_flag_defects(
    image: np.ndarray, params: Optional[DefectSegmentationParams] = None
) -> List[dict]:
    """Segment blobs in ``image`` and flag ones outside the expected shape.

    Returns a list of dictionaries, one per detected blob (bigger than
    ``min_area``), each with ``bbox`` (x, y, w, h), ``area``,
    ``aspect_ratio`` (bounding-box width / height), and ``is_defect``.
    """
    if params is None:
        params = DefectSegmentationParams()

    gray = _to_gray(image)
    if params.blur_kernel and params.blur_kernel > 0:
        kernel_size = params.blur_kernel | 1
        gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    base_flag = cv2.THRESH_BINARY_INV if params.invert else cv2.THRESH_BINARY
    if params.threshold_value <= 0:
        _, binary = cv2.threshold(gray, 0, 255, base_flag | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, params.threshold_value, 255, base_flag)

    kernel = np.ones((params.morph_kernel, params.morph_kernel), np.uint8)
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=params.morph_iterations
    )
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, kernel, iterations=params.morph_iterations
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area, max_area = params.expected_area_range
    min_aspect, max_aspect = params.expected_aspect_ratio_range

    results: List[dict] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < params.min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = (w / h) if h > 0 else 0.0

        within_area = min_area <= area <= max_area
        within_aspect = min_aspect <= aspect_ratio <= max_aspect
        is_defect = not (within_area and within_aspect)

        results.append(
            {
                "bbox": (int(x), int(y), int(w), int(h)),
                "area": float(area),
                "aspect_ratio": float(aspect_ratio),
                "is_defect": bool(is_defect),
            }
        )

    return results


def draw_defects(image: np.ndarray, blobs: List[dict]) -> np.ndarray:
    """Return a copy of ``image`` with each blob's bbox drawn: red = defect."""
    annotated = image.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    for blob in blobs:
        x, y, w, h = blob["bbox"]
        color = (0, 0, 255) if blob["is_defect"] else (0, 255, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    return annotated


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
