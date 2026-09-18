import numpy as np

from cvtoolkit.detectors.tamper_detection import TamperDetector


def _normal_frame(variant: int = 0) -> np.ndarray:
    """A textured "normal scene" frame: a checkerboard plus a little noise.

    The checkerboard gives the frame real edge content (so edge density is
    non-trivial) and real variance (so it isn't mistaken for a uniform
    blocked frame). ``variant`` adds a small amount of independent noise so
    consecutive frames are "near-identical" rather than bit-for-bit equal.
    """
    rng = np.random.default_rng(42 + variant)
    frame = np.full((240, 320, 3), 60, dtype=np.uint8)
    for i in range(0, 240, 20):
        for j in range(0, 320, 20):
            if (i // 20 + j // 20) % 2 == 0:
                frame[i : i + 20, j : j + 20] = 180
    noise = rng.integers(-3, 4, size=frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def _shifted_frame() -> np.ndarray:
    """A structurally very different, but still detailed, scene."""
    frame = np.full((240, 320, 3), 60, dtype=np.uint8)
    for i in range(0, 240, 20):
        for j in range(0, 320, 20):
            if (i // 20 + j // 20) % 2 == 1:  # inverted phase vs. _normal_frame
                frame[i : i + 20, j : j + 20] = 180
    return frame


def _black_frame() -> np.ndarray:
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _establish_background(detector: TamperDetector, num_frames: int = 6) -> None:
    for v in range(num_frames):
        detector.update(_normal_frame(v))


def test_no_false_positive_on_near_identical_normal_frames():
    detector = TamperDetector()
    detector.update(_normal_frame(0))

    for v in range(1, 4):
        result = detector.update(_normal_frame(v))
        assert result["tampered"] is False
        assert result["reason"] is None


def test_black_frame_flagged_as_blocked_after_background_established():
    detector = TamperDetector()
    _establish_background(detector)
    assert detector.is_established

    result = detector.update(_black_frame())

    assert result["tampered"] is True
    assert result["reason"] == "camera_blocked"


def test_shifted_scene_eventually_flagged_as_moved():
    detector = TamperDetector()
    _establish_background(detector)

    result = None
    for _ in range(10):
        result = detector.update(_shifted_frame())
        if result["tampered"]:
            break

    assert result is not None
    assert result["tampered"] is True
    assert result["reason"] == "camera_moved"


def test_first_frame_ever_is_never_flagged():
    detector = TamperDetector()
    result = detector.update(_black_frame())
    assert result["tampered"] is False
