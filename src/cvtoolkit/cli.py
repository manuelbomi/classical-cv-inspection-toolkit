"""Command-line interface for running a detector over a video or image folder.

Example
-------
::

    python -m cvtoolkit inspect \\
        --input demo.mp4 \\
        --detector hough_circles \\
        --output-dir output/hough_circles

Writes annotated frames (and, when the input is a video, an annotated
``.mp4``) plus a ``report.json`` describing the detections found in each
frame, into ``--output-dir``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

from .detectors import contour_defect_segmentation, hough_circles, hough_lines, tamper_detection

DETECTOR_CHOICES = (
    "hough_circles",
    "hough_lines",
    "contour_defect_segmentation",
    "tamper_detection",
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cvtoolkit", description="Classical computer-vision inspection toolkit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Run a detector over a video file or a directory of images."
    )
    inspect_parser.add_argument(
        "--input", required=True, help="Path to a video file, or a directory of images."
    )
    inspect_parser.add_argument(
        "--detector", required=True, choices=DETECTOR_CHOICES, help="Which detector to run."
    )
    inspect_parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory to write annotated frames/video and report.json into.",
    )
    inspect_parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of frames processed (useful for quick smoke tests).",
    )
    return parser


def _iter_frames(input_path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield (name, frame) pairs from either an image directory or a video."""
    if input_path.is_dir():
        files = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        for file_path in files:
            frame = cv2.imread(str(file_path))
            if frame is not None:
                yield file_path.name, frame
        return

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield f"frame_{index:05d}.png", frame
            index += 1
    finally:
        cap.release()


def _run_detector(
    name: str, frame: np.ndarray, tamper_state: Optional[tamper_detection.TamperDetector]
):
    if name == "hough_circles":
        detections = hough_circles.detect_circles(frame)
        annotated = hough_circles.draw_circles(frame, detections)
    elif name == "hough_lines":
        detections = hough_lines.detect_lines(frame)
        annotated = hough_lines.draw_lines(frame, detections)
    elif name == "contour_defect_segmentation":
        detections = contour_defect_segmentation.segment_and_flag_defects(frame)
        annotated = contour_defect_segmentation.draw_defects(frame, detections)
    elif name == "tamper_detection":
        assert tamper_state is not None
        result = tamper_state.update(frame)
        detections = result
        annotated = _annotate_tamper(frame, result)
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown detector: {name}")
    return detections, annotated


def _annotate_tamper(frame: np.ndarray, result: dict) -> np.ndarray:
    annotated = frame.copy()
    if annotated.ndim == 2:
        annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
    if result["tampered"]:
        text, color = f"TAMPER: {result['reason']}", (0, 0, 255)
    else:
        text, color = "OK", (0, 200, 0)
    cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return annotated


def run_inspect(args: argparse.Namespace) -> dict:
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    tamper_state = (
        tamper_detection.TamperDetector() if args.detector == "tamper_detection" else None
    )

    report = {"input": str(input_path), "detector": args.detector, "frames": []}
    writer = None

    try:
        for i, (name, frame) in enumerate(_iter_frames(input_path)):
            if args.max_frames is not None and i >= args.max_frames:
                break

            detections, annotated = _run_detector(args.detector, frame, tamper_state)
            cv2.imwrite(str(frames_dir / name), annotated)

            if input_path.is_file():
                if writer is None:
                    height, width = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_dir / "annotated.mp4"), fourcc, 20.0, (width, height)
                    )
                writer.write(annotated)

            report["frames"].append({"frame": name, "detections": detections})
    finally:
        if writer is not None:
            writer.release()

    report_path = output_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=_json_default)

    return report


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        report = run_inspect(args)
        report_path = Path(args.output_dir) / "report.json"
        print(f"Processed {len(report['frames'])} frame(s). Report written to {report_path}")
        return 0

    parser.print_help()  # pragma: no cover - unreachable while "inspect" is the only command
    return 1


if __name__ == "__main__":
    sys.exit(main())
