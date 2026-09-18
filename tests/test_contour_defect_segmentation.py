import cv2
import numpy as np

from cvtoolkit.detectors.contour_defect_segmentation import (
    DefectSegmentationParams,
    segment_and_flag_defects,
)


def _canvas() -> np.ndarray:
    return np.zeros((300, 300), dtype=np.uint8)


def test_normal_blob_not_flagged_as_defect():
    img = _canvas()
    cv2.circle(img, (80, 80), 30, 255, -1)  # area ~2827, aspect ratio ~1.0

    params = DefectSegmentationParams(
        expected_area_range=(1500, 4000), expected_aspect_ratio_range=(0.7, 1.3)
    )
    blobs = segment_and_flag_defects(img, params)

    assert len(blobs) == 1
    assert blobs[0]["is_defect"] is False


def test_malformed_blob_flagged_as_defect():
    img = _canvas()
    # Long thin rectangle: area is in range, but aspect ratio is way outside
    # the expected range -- simulating a smear, crack, or misshapen part.
    cv2.rectangle(img, (30, 140), (270, 150), 255, -1)

    params = DefectSegmentationParams(
        expected_area_range=(1500, 4000), expected_aspect_ratio_range=(0.7, 1.3)
    )
    blobs = segment_and_flag_defects(img, params)

    assert len(blobs) == 1
    assert blobs[0]["is_defect"] is True


def test_undersized_blob_flagged_as_defect():
    img = _canvas()
    cv2.circle(img, (80, 80), 8, 255, -1)  # tiny blob, area well below range

    params = DefectSegmentationParams(
        min_area=10, expected_area_range=(1500, 4000), expected_aspect_ratio_range=(0.7, 1.3)
    )
    blobs = segment_and_flag_defects(img, params)

    assert len(blobs) == 1
    assert blobs[0]["is_defect"] is True
