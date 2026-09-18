/**
 * Types mirroring the real `report.json` shape produced by
 * `python -m cvtoolkit inspect --detector <name> ...` (see
 * `src/cvtoolkit/cli.py::run_inspect` and each detector module under
 * `src/cvtoolkit/detectors/`).
 *
 * `report.json`'s top-level shape is:
 *   { input: string, detector: string, frames: [{ frame: string, detections: ... }] }
 *
 * `detections` differs by detector -- each detector's return type is
 * reproduced below verbatim (field names copied from the Python
 * dataclasses / dict literals, not guessed).
 */

/** From `hough_circles.detect_circles()`: one detected circle. */
export interface CircleDetection {
  /** [x, y] center in pixel coordinates. */
  center: [number, number];
  radius: number;
}

/** From `hough_lines.detect_lines()`: one detected line segment. */
export interface LineDetection {
  point1: [number, number];
  point2: [number, number];
  length: number;
  /** Degrees, (-180, 180], measured from the positive x-axis, image coordinates. */
  angle_deg: number;
}

/** From `contour_defect_segmentation.segment_and_flag_defects()`: one blob. */
export interface BlobDetection {
  /** [x, y, w, h] bounding box in pixel coordinates. */
  bbox: [number, number, number, number];
  area: number;
  aspect_ratio: number;
  is_defect: boolean;
}

/** From `tamper_detection.TamperDetector.update()`: per-frame tamper status. */
export interface TamperResult {
  tampered: boolean;
  reason: "camera_blocked" | "camera_moved" | null;
  diff_ratio: number;
  edge_density: number;
  std: number | null;
  frame_count: number;
}

export type DetectorName =
  | "hough_circles"
  | "hough_lines"
  | "contour_defect_segmentation"
  | "tamper_detection";

/** One representative frame's real detections, plus the annotated PNG that was written by the CLI. */
export interface RepresentativeFrame<T> {
  /** Filename as written under output/<detector>/frames/. */
  frame: string;
  /** Path (relative to the app's public root) to the annotated PNG. */
  image: string;
  detections: T;
}

interface BaseDetectorSummary {
  detector: DetectorName;
  label: string;
  description: string;
  totalFrames: number;
}

export interface CircleDetectorSummary extends BaseDetectorSummary {
  detector: "hough_circles";
  framesWithDetections: number;
  totalDetections: number;
  avgDetectionsPerFrame: number;
  maxDetectionsInFrame: number;
  representativeFrame: RepresentativeFrame<CircleDetection[]>;
}

export interface LineDetectorSummary extends BaseDetectorSummary {
  detector: "hough_lines";
  framesWithDetections: number;
  totalDetections: number;
  avgDetectionsPerFrame: number;
  maxDetectionsInFrame: number;
  representativeFrame: RepresentativeFrame<LineDetection[]>;
}

export interface ContourDetectorSummary extends BaseDetectorSummary {
  detector: "contour_defect_segmentation";
  framesWithDetections: number;
  totalDetections: number;
  totalDefectsFlagged: number;
  framesWithDefect: number;
  representativeFrame: RepresentativeFrame<BlobDetection[]>;
}

export interface TamperDetectorSummary extends BaseDetectorSummary {
  detector: "tamper_detection";
  tamperedFrameCount: number;
  reasonBreakdown: Record<string, number>;
  firstTamperedFrame: string | null;
  representativeFrame: RepresentativeFrame<TamperResult>;
}

export interface ResultsSummary {
  sourceVideo: string;
  detectors: {
    hough_circles: CircleDetectorSummary;
    hough_lines: LineDetectorSummary;
    contour_defect_segmentation: ContourDetectorSummary;
    tamper_detection: TamperDetectorSummary;
  };
}
