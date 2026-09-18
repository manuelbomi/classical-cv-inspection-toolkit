import cv2
import numpy as np

from cvtoolkit.detectors.hough_lines import LineDetectionParams, detect_lines


def test_detects_drawn_horizontal_line():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    y = 100
    cv2.line(img, (20, y), (380, y), (255, 255, 255), 3)

    params = LineDetectionParams(threshold=30, min_line_length=100, max_line_gap=5)
    lines = detect_lines(img, params)

    assert len(lines) >= 1
    best = lines[0]  # longest detected segment
    is_near_horizontal = abs(best["angle_deg"]) < 5 or abs(abs(best["angle_deg"]) - 180) < 5
    assert is_near_horizontal
    assert abs(best["point1"][1] - y) < 5
    assert abs(best["point2"][1] - y) < 5


def test_detects_drawn_diagonal_line_angle():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.line(img, (20, 20), (280, 280), (255, 255, 255), 3)  # ~45 degrees

    params = LineDetectionParams(threshold=30, min_line_length=100, max_line_gap=5)
    lines = detect_lines(img, params)

    assert len(lines) >= 1
    best = lines[0]
    assert abs(abs(best["angle_deg"]) - 45) < 5


def test_no_lines_in_blank_image():
    img = np.zeros((200, 400, 3), dtype=np.uint8)
    lines = detect_lines(img)
    assert lines == []
