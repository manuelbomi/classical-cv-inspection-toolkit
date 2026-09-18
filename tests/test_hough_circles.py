import cv2
import numpy as np

from cvtoolkit.detectors.hough_circles import CircleDetectionParams, detect_circles


def test_detects_drawn_circle_center_and_radius():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    center = (150, 150)
    radius = 40
    cv2.circle(img, center, radius, (255, 255, 255), -1)

    params = CircleDetectionParams(
        dp=1.2, min_dist=50, param1=50, param2=25, min_radius=20, max_radius=60
    )
    circles = detect_circles(img, params)

    assert len(circles) >= 1
    best = min(circles, key=lambda c: abs(c["radius"] - radius))
    assert abs(best["center"][0] - center[0]) < 8
    assert abs(best["center"][1] - center[1]) < 8
    assert abs(best["radius"] - radius) < 10


def test_no_circles_in_blank_image():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    circles = detect_circles(img)
    assert circles == []
