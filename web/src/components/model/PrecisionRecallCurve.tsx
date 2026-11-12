import React, { useState } from "react";
import { PR_SWEEP_DATA, PER_CLASS_EVALUATION, type PRCurvePoint } from "../../data/evaluationData";
import { EmptyState } from "../ui/EmptyState";
import { Button } from "../ui/Button";

export const PrecisionRecallCurve: React.FC = () => {
  const [selectedClassId, setSelectedClassId] = useState<string>("all");
  const [hoveredPoint, setHoveredPoint] = useState<PRCurvePoint | null>(null);

  const selectedClass = PER_CLASS_EVALUATION.find(
    (c) => c.id.toString() === selectedClassId
  );

  // SVG dimensions
  const width = 640;
  const height = 280;
  const padLeft = 46;
  const padRight = 30;
  const padTop = 24;
  const padBottom = 40;

  const chartW = width - padLeft - padRight;
  const chartH = height - padTop - padBottom;

  const getX = (recall: number) => padLeft + recall * chartW;
  const getY = (precision: number) => padTop + chartH - precision * chartH;

  // Path data for the PR curve (sorted by recall)
  const sortedPoints = [...PR_SWEEP_DATA].sort((a, b) => a.recall - b.recall);
  const pathData = sortedPoints
    .map((pt, i) => `${i === 0 ? "M" : "L"} ${getX(pt.recall)} ${getY(pt.precision)}`)
    .join(" ");

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">
            Precision-Recall Tradeoff Curve
          </h4>
          <p className="text-xs text-text-muted mt-0.5">
            Validation benchmark sweep across confidence thresholds 0.10 to 0.70
          </p>
        </div>

        {/* Class Selector */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <label htmlFor="pr-class-select" className="text-xs text-text-muted">
            Scope:
          </label>
          <select
            id="pr-class-select"
            value={selectedClassId}
            onChange={(e) => setSelectedClassId(e.target.value)}
            className="h-8 rounded-control border border-border-subtle bg-card px-2.5 text-xs text-text-primary focus:border-brand focus:outline-none focus:ring-1 focus:ring-brand font-medium"
          >
            <option value="all">All Classes (Aggregate Model)</option>
            <optgroup label="COCO Shared (10)">
              {PER_CLASS_EVALUATION.filter((c) => c.is_coco).map((c) => (
                <option key={c.id} value={c.id.toString()}>
                  {c.name} (AP: {c.finetuned_ap.toFixed(3)})
                </option>
              ))}
            </optgroup>
            <optgroup label="Novel Concepts (10)">
              {PER_CLASS_EVALUATION.filter((c) => !c.is_coco).map((c) => (
                <option key={c.id} value={c.id.toString()}>
                  {c.name} (AP: {c.finetuned_ap.toFixed(3)})
                </option>
              ))}
            </optgroup>
          </select>
        </div>
      </div>

      <div className="relative rounded-panel border border-border-subtle bg-card p-3 sm:p-5">
        {selectedClassId !== "all" ? (
          /* Clear Empty State when per-class data is missing */
          <div className="py-8">
            <EmptyState
              title={`Per-Class PR Curve Unavailable for ${selectedClass?.name || "Selected Class"}`}
              description={`Confidence threshold sweep data was recorded at the aggregate 20-class model level in the benchmark report (docs/fine_tuned_metrics.json). Per-class PR point curves were not generated during this evaluation run.`}
              action={
                <Button size="sm" onClick={() => setSelectedClassId("all")}>
                  View Aggregate Model Curve
                </Button>
              }
            />
          </div>
        ) : (
          /* Aggregate Real PR Curve */
          <div className="flex flex-col gap-3">
            <div className="overflow-x-auto">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="w-full h-auto min-w-[500px] select-none"
                role="img"
                aria-label="Precision-Recall curve across thresholds"
              >
                {/* Horizontal gridlines & Y ticks (Precision: 0.0 to 1.0) */}
                {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => {
                  const y = getY(tick);
                  return (
                    <g key={`y-${tick}`}>
                      <line
                        x1={padLeft}
                        y1={y}
                        x2={width - padRight}
                        y2={y}
                        stroke="currentColor"
                        className="text-border-subtle"
                        strokeWidth="1"
                        strokeDasharray={tick === 0 ? undefined : "3 3"}
                      />
                      <text
                        x={padLeft - 8}
                        y={y + 3.5}
                        textAnchor="end"
                        className="fill-text-muted text-[10px] font-mono"
                      >
                        {tick.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                {/* Vertical gridlines & X ticks (Recall: 0.0 to 1.0) */}
                {[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map((tick) => {
                  const x = getX(tick);
                  return (
                    <g key={`x-${tick}`}>
                      <line
                        x1={x}
                        y1={padTop}
                        x2={x}
                        y2={padTop + chartH}
                        stroke="currentColor"
                        className="text-border-subtle"
                        strokeWidth="1"
                        strokeDasharray={tick === 0 ? undefined : "3 3"}
                      />
                      <text
                        x={x}
                        y={padTop + chartH + 16}
                        textAnchor="middle"
                        className="fill-text-muted text-[10px] font-mono"
                      >
                        {tick.toFixed(1)}
                      </text>
                    </g>
                  );
                })}

                {/* Axis Titles */}
                <text
                  x={padLeft + chartW / 2}
                  y={padTop + chartH + 32}
                  textAnchor="middle"
                  className="fill-text-secondary text-[11px] font-medium"
                >
                  Recall (0.0 → 1.0)
                </text>
                <text
                  x={14}
                  y={padTop + chartH / 2}
                  textAnchor="middle"
                  transform={`rotate(-90 14 ${padTop + chartH / 2})`}
                  className="fill-text-secondary text-[11px] font-medium"
                >
                  Precision (0.0 → 1.0)
                </text>

                {/* Shaded Area under PR curve */}
                <path
                  d={`${pathData} L ${getX(sortedPoints[sortedPoints.length - 1].recall)} ${getY(0)} L ${getX(sortedPoints[0].recall)} ${getY(0)} Z`}
                  className="fill-brand/10"
                />

                {/* PR Curve Line */}
                <path
                  d={pathData}
                  fill="none"
                  stroke="var(--color-brand)"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Plotted Threshold Points */}
                {PR_SWEEP_DATA.map((pt) => {
                  const cx = getX(pt.recall);
                  const cy = getY(pt.precision);
                  const isOpt = pt.isOptimal;

                  return (
                    <g key={pt.threshold} className="cursor-pointer">
                      {isOpt && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r="9"
                          fill="none"
                          stroke="var(--color-brand)"
                          strokeWidth="1.5"
                          strokeDasharray="2 2"
                          className="animate-pulse"
                        />
                      )}
                      <circle
                        cx={cx}
                        cy={cy}
                        r={isOpt ? "5" : "3.5"}
                        className={
                          isOpt
                            ? "fill-brand stroke-card stroke-2"
                            : "fill-card stroke-brand stroke-2 hover:fill-brand transition-colors"
                        }
                        onMouseEnter={() => setHoveredPoint(pt)}
                        onMouseLeave={() => setHoveredPoint(null)}
                      />
                      {/* Direct label for optimal threshold */}
                      {isOpt && (
                        <g>
                          <rect
                            x={cx + 8}
                            y={cy - 22}
                            width="96"
                            height="20"
                            rx="3"
                            className="fill-elevated stroke-border-default stroke-[1]"
                          />
                          <text
                            x={cx + 56}
                            y={cy - 9}
                            textAnchor="middle"
                            className="fill-brand text-[10px] font-mono font-medium"
                          >
                            Optimal: 0.25 (F1 {pt.f1})
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>

            {/* Sweep Data Strip below curve */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-2 border-t border-border-subtle">
              <div className="rounded-control bg-subtle p-2">
                <div className="text-[10px] uppercase text-text-muted font-medium">Optimal Operating Point</div>
                <div className="text-sm font-bold font-mono text-brand mt-0.5">Threshold 0.25</div>
                <div className="text-[11px] text-text-muted mt-0.5">Peak F1: 0.6398</div>
              </div>
              <div className="rounded-control bg-subtle p-2">
                <div className="text-[10px] uppercase text-text-muted font-medium">Precision @ 0.25</div>
                <div className="text-sm font-bold font-mono text-text-primary mt-0.5">65.0%</div>
                <div className="text-[11px] text-text-muted mt-0.5">TP / (TP + FP)</div>
              </div>
              <div className="rounded-control bg-subtle p-2">
                <div className="text-[10px] uppercase text-text-muted font-medium">Recall @ 0.25</div>
                <div className="text-sm font-bold font-mono text-text-primary mt-0.5">63.0%</div>
                <div className="text-[11px] text-text-muted mt-0.5">TP / (TP + FN)</div>
              </div>
              <div className="rounded-control bg-subtle p-2">
                <div className="text-[10px] uppercase text-text-muted font-medium">Max Recall Point</div>
                <div className="text-sm font-bold font-mono text-text-primary mt-0.5">78.0%</div>
                <div className="text-[11px] text-text-muted mt-0.5">@ 0.10 threshold (P: 42%)</div>
              </div>
            </div>

            {/* Hover details badge */}
            {hoveredPoint && (
              <div className="text-xs text-text-secondary bg-elevated rounded-control p-2 border border-border-subtle flex items-center justify-between">
                <span className="font-medium text-text-primary">
                  Threshold: <span className="font-mono text-brand">{hoveredPoint.threshold.toFixed(2)}</span>
                </span>
                <span className="font-mono">Precision: {(hoveredPoint.precision * 100).toFixed(1)}%</span>
                <span className="font-mono">Recall: {(hoveredPoint.recall * 100).toFixed(1)}%</span>
                <span className="font-mono">F1 Score: {hoveredPoint.f1.toFixed(4)}</span>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
