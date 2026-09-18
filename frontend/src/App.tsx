import { useState } from "react";
import rawSummary from "./data/summary.json";
import type {
  ResultsSummary,
  CircleDetectorSummary,
  LineDetectorSummary,
  ContourDetectorSummary,
  TamperDetectorSummary,
} from "./types";

// `summary.json` is generated directly from real `report.json` output written
// by `python -m cvtoolkit inspect ...` (see scripts used to produce
// frontend/src/data/summary.json) -- the shape is guaranteed by that
// generation step, so we assert it onto the strong types in types.ts rather
// than re-declare a looser JSON-inferred type.
const summary = rawSummary as unknown as ResultsSummary;

type TabKey = keyof ResultsSummary["detectors"];

const TABS: { key: TabKey; short: string }[] = [
  { key: "hough_circles", short: "Round Objects" },
  { key: "hough_lines", short: "Alignment" },
  { key: "contour_defect_segmentation", short: "Defects" },
  { key: "tamper_detection", short: "Camera Health" },
];

function StatTile({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="stat-tile">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}

function CircleCard({ data }: { data: CircleDetectorSummary }) {
  return (
    <>
      <div className="stat-row">
        <StatTile label="Frames Processed" value={data.totalFrames} />
        <StatTile label="Frames With Detections" value={data.framesWithDetections} />
        <StatTile label="Total Circles Found" value={data.totalDetections} />
        <StatTile label="Avg / Frame" value={data.avgDetectionsPerFrame} />
        <StatTile label="Max in One Frame" value={data.maxDetectionsInFrame} />
      </div>
      <p className="caption">
        Green outlines mark every circle the Hough Circle Transform voted for on frame{" "}
        <code>{data.representativeFrame.frame}</code>; the red dot marks each detected center.
        Real parts on the conveyor are picked up correctly, alongside the extra spurious circles
        that default (untuned) Hough parameters produce on this scene &mdash; exactly the
        parameter-sensitivity tradeoff documented in the README.
      </p>
    </>
  );
}

function LineCard({ data }: { data: LineDetectorSummary }) {
  const longest = data.representativeFrame.detections[0];
  return (
    <>
      <div className="stat-row">
        <StatTile label="Frames Processed" value={data.totalFrames} />
        <StatTile label="Frames With Detections" value={data.framesWithDetections} />
        <StatTile label="Total Line Segments" value={data.totalDetections} />
        <StatTile label="Avg / Frame" value={data.avgDetectionsPerFrame} />
        <StatTile label="Max in One Frame" value={data.maxDetectionsInFrame} />
      </div>
      <p className="caption">
        Yellow segments are straight edges found by the probabilistic Hough Line Transform on
        frame <code>{data.representativeFrame.frame}</code> &mdash; the two long horizontal lines
        are the conveyor guide rails
        {longest ? ` (longest detected segment: ${longest.length.toFixed(0)}px at ${longest.angle_deg.toFixed(1)}°)` : ""}
        , used to confirm the rail hasn&rsquo;t drifted out of alignment.
      </p>
    </>
  );
}

function ContourCard({ data }: { data: ContourDetectorSummary }) {
  return (
    <>
      <div className="stat-row">
        <StatTile label="Frames Processed" value={data.totalFrames} />
        <StatTile label="Blobs Found" value={data.totalDetections} />
        <StatTile label="Defects Flagged" value={data.totalDefectsFlagged} />
        <StatTile label="Frames With a Defect" value={data.framesWithDefect} />
      </div>
      <p className="caption">
        Each bounding box is a segmented blob on frame <code>{data.representativeFrame.frame}</code>:
        green means the blob&rsquo;s area and aspect ratio fell inside the expected range, red
        means it was flagged as a defect &mdash; here the irregular polygon standing in for a
        chipped/malformed part is correctly boxed in red while the four round good parts are
        boxed in green.
      </p>
    </>
  );
}

function TamperCard({ data }: { data: TamperDetectorSummary }) {
  const reasons = Object.entries(data.reasonBreakdown);
  return (
    <>
      <div className="stat-row">
        <StatTile label="Frames Processed" value={data.totalFrames} />
        <StatTile label="Tampered Frames" value={data.tamperedFrameCount} />
        <StatTile
          label="First Tampered Frame"
          value={data.firstTamperedFrame ?? "none"}
        />
        {reasons.map(([reason, count]) => (
          <StatTile key={reason} label={reason} value={count} />
        ))}
      </div>
      <p className="caption">
        The tamper detector watches the feed&rsquo;s own statistics (background diff, frame
        variance, edge density) rather than the scene contents. Frame{" "}
        <code>{data.representativeFrame.frame}</code> shows the simulated lens-blackout in the
        demo clip correctly caught and labeled <code>TAMPER: camera_blocked</code> in red, once
        the background model had been established.
      </p>
    </>
  );
}

function DetectorPanel({ tab }: { tab: TabKey }) {
  const d = summary.detectors[tab];
  return (
    <section className="panel">
      <div className="panel-image">
        <img src={d.representativeFrame.image} alt={`${d.label} — annotated output`} />
      </div>
      <div className="panel-body">
        <h2>{d.label}</h2>
        <p className="detector-description">{d.description}</p>
        {tab === "hough_circles" && (
          <CircleCard data={summary.detectors.hough_circles} />
        )}
        {tab === "hough_lines" && <LineCard data={summary.detectors.hough_lines} />}
        {tab === "contour_defect_segmentation" && (
          <ContourCard data={summary.detectors.contour_defect_segmentation} />
        )}
        {tab === "tamper_detection" && (
          <TamperCard data={summary.detectors.tamper_detection} />
        )}
      </div>
    </section>
  );
}

export default function App() {
  const [activeTab, setActiveTab] = useState<TabKey>("hough_circles");

  return (
    <div className="app">
      <header className="app-header">
        <h1>Classical CV Inspection Toolkit — Results Viewer</h1>
        <p>
          Real annotated output from all four detectors, run by the actual CLI (
          <code>python -m cvtoolkit inspect</code>) against the repo&rsquo;s own synthetic demo
          clip (<code>{summary.sourceVideo}</code>). Every image, count, and statistic below comes
          straight from the <code>report.json</code> each run produced &mdash; nothing here is a
          mockup.
        </p>
      </header>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.key}
            className={t.key === activeTab ? "tab active" : "tab"}
            onClick={() => setActiveTab(t.key)}
          >
            {t.short}
          </button>
        ))}
      </nav>

      <DetectorPanel tab={activeTab} />

      <section className="grid-preview">
        <h3>All four detectors at a glance</h3>
        <div className="grid">
          {TABS.map((t) => {
            const d = summary.detectors[t.key];
            return (
              <button
                key={t.key}
                className={
                  t.key === activeTab ? "grid-item active" : "grid-item"
                }
                onClick={() => setActiveTab(t.key)}
              >
                <img src={d.representativeFrame.image} alt={d.label} />
                <span>{d.label}</span>
              </button>
            );
          })}
        </div>
      </section>

      <footer className="app-footer">
        <p>
          Classical, deterministic, CPU-only computer vision &mdash; no training data, no GPU. See
          the project README for the full architecture and design rationale.
        </p>
      </footer>
    </div>
  );
}
