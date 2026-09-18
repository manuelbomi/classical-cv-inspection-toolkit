"""Generate a synthetic demo video for the classical CV inspection toolkit.

Draws circles ("parts/vials on a conveyor") moving left-to-right along a
fixed line, at regular spacing, with every 5th item swapped for an
intentionally irregular/malformed blob ("defective" item). Two horizontal
guide lines bracket the conveyor for the alignment/edge detector to pick up.
Partway through the clip the frame is briefly blacked out to simulate the
camera being blocked, so the same clip can also demo the tamper detector.

None of this depends on any real footage or proprietary data -- everything
is drawn procedurally with OpenCV primitives.

Usage
-----
::

    python scripts/generate_demo_video.py --output demo.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
FPS = 20
DURATION_S = 10
NUM_FRAMES = FPS * DURATION_S

CONVEYOR_Y = HEIGHT - 150
ITEM_RADIUS = 28
ITEM_SPACING = 140
SPEED_PX_PER_FRAME = 4


def make_frame(frame_idx: int) -> np.ndarray:
    """Render one frame of the synthetic conveyor scene."""
    frame = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)  # dark gray background

    # Conveyor guide lines, for the alignment / edge detector demo.
    top_y = CONVEYOR_Y - ITEM_RADIUS - 20
    bottom_y = CONVEYOR_Y + ITEM_RADIUS + 20
    cv2.line(frame, (0, top_y), (WIDTH, top_y), (90, 90, 90), 3)
    cv2.line(frame, (0, bottom_y), (WIDTH, bottom_y), (90, 90, 90), 3)

    offset = (frame_idx * SPEED_PX_PER_FRAME) % ITEM_SPACING
    x = -ITEM_SPACING + offset
    item_index = 0
    while x < WIDTH + ITEM_SPACING:
        is_defect = (item_index % 5 == 4)  # every 5th item is malformed
        _draw_item(frame, x, CONVEYOR_Y, is_defect, item_index)
        x += ITEM_SPACING
        item_index += 1

    return frame


def _draw_item(frame: np.ndarray, x: float, y: float, is_defect: bool, item_index: int) -> None:
    center = (int(x), int(y))
    if not is_defect:
        cv2.circle(frame, center, ITEM_RADIUS, (200, 200, 200), -1)
        cv2.circle(frame, center, ITEM_RADIUS, (255, 255, 255), 2)
        return

    # Irregular/malformed blob: a jittered polygon instead of a clean circle,
    # standing in for a chipped, deformed, or foreign-material part.
    rng = np.random.default_rng(item_index)
    n_points = 9
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = ITEM_RADIUS * (0.5 + rng.random(n_points))
    points = np.array(
        [[x + r * np.cos(a), y + r * np.sin(a)] for a, r in zip(angles, radii)],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [points], (150, 150, 210))
    cv2.polylines(frame, [points], True, (255, 255, 255), 2)


def generate(output_path: Path, occlusion_frames: "set[int] | None" = None) -> None:
    """Render the full clip and write it to ``output_path`` as an .mp4."""
    occlusion_frames = occlusion_frames or set()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for: {output_path}")

    try:
        for i in range(NUM_FRAMES):
            frame = make_frame(i)
            if i in occlusion_frames:
                frame[:] = 0  # simulate the camera lens being fully blocked
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="demo.mp4", help="Output .mp4 file path.")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Briefly "block" the camera around the 6-second mark, so this one clip
    # can also demo the tamper detector.
    occlusion_start = FPS * 6
    occlusion_frames = set(range(occlusion_start, occlusion_start + FPS // 2))

    generate(output_path, occlusion_frames=occlusion_frames)
    print(f"Wrote {NUM_FRAMES} frames ({NUM_FRAMES / FPS:.1f}s) to {output_path}")


if __name__ == "__main__":
    main()
