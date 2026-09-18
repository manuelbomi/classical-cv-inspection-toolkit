"""Generate `frontend/src/data/summary.json` from real detector output.

This reads the `report.json` files written by `python -m cvtoolkit inspect
--detector <name> --output-dir output/<name> ...` for all four detectors and
produces a small, strongly-typed-friendly summary (real counts/stats plus one
representative frame's real detections per detector) that
`frontend/src/App.tsx` renders. It does not invent any numbers -- everything
here is derived directly from the CLI's own JSON output.

Usage
-----
Run all four detectors first (adjust `--input` to your demo video path)::

    python -m cvtoolkit inspect --input demo.mp4 --detector hough_circles --output-dir output/hough_circles
    python -m cvtoolkit inspect --input demo.mp4 --detector hough_lines --output-dir output/hough_lines
    python -m cvtoolkit inspect --input demo.mp4 --detector contour_defect_segmentation --output-dir output/contours
    python -m cvtoolkit inspect --input demo.mp4 --detector tamper_detection --output-dir output/tamper

Then copy a representative annotated frame from each `output/<name>/frames/`
directory into `frontend/public/results/<slug>.png` (and, for the README,
into `docs/screenshots/detector-<slug>.png`), and run::

    python scripts/generate_frontend_summary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (report.json path, representative frame filename, public image path)
DETECTOR_CONFIG = {
    "hough_circles": {
        "report": "output/hough_circles/report.json",
        "frame": "frame_00097.png",
        "image": "/results/hough-circles.png",
        "label": "Round-Object Detection",
        "description": (
            "Finds circular parts (bottles, vials, pills, washers, wheels) using the "
            "Hough Circle Transform."
        ),
    },
    "hough_lines": {
        "report": "output/hough_lines/report.json",
        "frame": "frame_00022.png",
        "image": "/results/hough-lines.png",
        "label": "Alignment / Edge Detection",
        "description": (
            "Finds straight edges (conveyor rails, part edges) using the probabilistic "
            "Hough Line Transform."
        ),
    },
    "contour_defect_segmentation": {
        "report": "output/contours/report.json",
        "frame": "frame_00000.png",
        "image": "/results/contour-defects.png",
        "label": "Defect / Shape Segmentation",
        "description": (
            "Thresholds and outlines blobs, flagging any whose size or shape falls "
            "outside the expected range as a defect."
        ),
    },
    "tamper_detection": {
        "report": "output/tamper/report.json",
        "frame": "frame_00122.png",
        "image": "/results/tamper-detection.png",
        "label": "Camera Tamper / Occlusion Detection",
        "description": (
            "Watches the camera feed's own statistics (background diff, frame variance, "
            "edge density) to flag blocking or movement of the camera itself."
        ),
    },
}


def _load(report_path: Path) -> dict:
    with open(report_path) as f:
        return json.load(f)


def _frame_by_name(report: dict, name: str) -> dict:
    for fr in report["frames"]:
        if fr["frame"] == name:
            return fr
    raise KeyError(f"Frame {name!r} not found in report with {len(report['frames'])} frames")


def build_summary(root: Path) -> dict:
    circles = _load(root / DETECTOR_CONFIG["hough_circles"]["report"])
    lines = _load(root / DETECTOR_CONFIG["hough_lines"]["report"])
    contours = _load(root / DETECTOR_CONFIG["contour_defect_segmentation"]["report"])
    tamper = _load(root / DETECTOR_CONFIG["tamper_detection"]["report"])

    out: dict = {"sourceVideo": "demo.mp4", "detectors": {}}

    c_counts = [len(fr["detections"]) for fr in circles["frames"]]
    c_cfg = DETECTOR_CONFIG["hough_circles"]
    c_frame = _frame_by_name(circles, c_cfg["frame"])
    out["detectors"]["hough_circles"] = {
        "detector": "hough_circles",
        "label": c_cfg["label"],
        "description": c_cfg["description"],
        "totalFrames": len(circles["frames"]),
        "framesWithDetections": sum(1 for n in c_counts if n > 0),
        "totalDetections": sum(c_counts),
        "avgDetectionsPerFrame": round(sum(c_counts) / len(c_counts), 2),
        "maxDetectionsInFrame": max(c_counts),
        "representativeFrame": {
            "frame": c_frame["frame"],
            "image": c_cfg["image"],
            "detections": c_frame["detections"],
        },
    }

    l_counts = [len(fr["detections"]) for fr in lines["frames"]]
    l_cfg = DETECTOR_CONFIG["hough_lines"]
    l_frame = _frame_by_name(lines, l_cfg["frame"])
    out["detectors"]["hough_lines"] = {
        "detector": "hough_lines",
        "label": l_cfg["label"],
        "description": l_cfg["description"],
        "totalFrames": len(lines["frames"]),
        "framesWithDetections": sum(1 for n in l_counts if n > 0),
        "totalDetections": sum(l_counts),
        "avgDetectionsPerFrame": round(sum(l_counts) / len(l_counts), 2),
        "maxDetectionsInFrame": max(l_counts),
        "representativeFrame": {
            "frame": l_frame["frame"],
            "image": l_cfg["image"],
            "detections": l_frame["detections"],
        },
    }

    blob_counts = [len(fr["detections"]) for fr in contours["frames"]]
    defect_counts = [
        sum(1 for d in fr["detections"] if d["is_defect"]) for fr in contours["frames"]
    ]
    d_cfg = DETECTOR_CONFIG["contour_defect_segmentation"]
    d_frame = _frame_by_name(contours, d_cfg["frame"])
    out["detectors"]["contour_defect_segmentation"] = {
        "detector": "contour_defect_segmentation",
        "label": d_cfg["label"],
        "description": d_cfg["description"],
        "totalFrames": len(contours["frames"]),
        "framesWithDetections": sum(1 for n in blob_counts if n > 0),
        "totalDetections": sum(blob_counts),
        "totalDefectsFlagged": sum(defect_counts),
        "framesWithDefect": sum(1 for n in defect_counts if n > 0),
        "representativeFrame": {
            "frame": d_frame["frame"],
            "image": d_cfg["image"],
            "detections": d_frame["detections"],
        },
    }

    t_cfg = DETECTOR_CONFIG["tamper_detection"]
    t_frame = _frame_by_name(tamper, t_cfg["frame"])
    tampered_frames = [fr for fr in tamper["frames"] if fr["detections"]["tampered"]]
    reason_counts: dict = {}
    for fr in tampered_frames:
        reason = fr["detections"]["reason"]
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    out["detectors"]["tamper_detection"] = {
        "detector": "tamper_detection",
        "label": t_cfg["label"],
        "description": t_cfg["description"],
        "totalFrames": len(tamper["frames"]),
        "tamperedFrameCount": len(tampered_frames),
        "reasonBreakdown": reason_counts,
        "firstTamperedFrame": tampered_frames[0]["frame"] if tampered_frames else None,
        "representativeFrame": {
            "frame": t_frame["frame"],
            "image": t_cfg["image"],
            "detections": t_frame["detections"],
        },
    }

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT),
        help="Repo root containing output/ (default: this script's repo root).",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "frontend" / "src" / "data" / "summary.json"),
        help="Where to write the generated summary JSON.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    summary = build_summary(root)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary for {len(summary['detectors'])} detectors to {out_path}")


if __name__ == "__main__":
    main()
