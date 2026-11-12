import React, { useState } from "react";
import { BENCHMARK_METRICS, type BenchmarkMetric } from "../../data/evaluationData";

interface HoveredData {
  metric: BenchmarkMetric;
  type: "baseline" | "fineTuned";
  value: number;
  /** SVG-space coordinates (viewBox units) */
  svgX: number;
  svgY: number;
}

export const GroupedMetricBars: React.FC = () => {
  const [hovered, setHovered] = useState<HoveredData | null>(null);

  // SVG dimensions
  const width = 640;
  const height = 240;
  const padLeft = 40;
  const padRight = 20;
  const padTop = 32;
  const padBottom = 48;

  const chartWidth = width - padLeft - padRight;
  const chartHeight = height - padTop - padBottom;

  const maxVal = 0.45; // metrics range from 0 to ~0.35
  const gridTicks = [0.0, 0.1, 0.2, 0.3, 0.4];

  const groupWidth = chartWidth / BENCHMARK_METRICS.length;
  const barWidth = 26;
  const gap = 4;

  const getY = (val: number) => padTop + chartHeight - (val / maxVal) * chartHeight;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">
            Baseline vs Fine-Tuned Benchmark Comparison
          </h4>
          <p className="text-xs text-text-muted mt-0.5">
            Grouped metrics across mAP@0.5 and COCO mAP@[.5:.95]
          </p>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs text-text-secondary select-none">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-text-muted/60" />
            <span>Baseline (Pre-trained)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-brand" />
            <span className="font-medium text-text-primary">Fine-Tuned</span>
          </div>
        </div>
      </div>

      <div className="relative w-full overflow-x-auto rounded-panel border border-border-subtle bg-card p-3 sm:p-4">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto min-w-[460px] select-none"
          role="img"
          aria-label="Grouped bar chart comparing baseline and fine-tuned model metrics"
        >
          {/* Horizontal gridlines and Y-axis tick labels */}
          {gridTicks.map((tick) => {
            const y = getY(tick);
            return (
              <g key={tick}>
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
                  {tick.toFixed(2)}
                </text>
              </g>
            );
          })}

          {/* Grouped bars */}
          {BENCHMARK_METRICS.map((m, idx) => {
            const groupX = padLeft + idx * groupWidth + (groupWidth - (barWidth * 2 + gap)) / 2;
            const baselineX = groupX;
            const fineTunedX = groupX + barWidth + gap;

            const baseY = getY(m.baseline);
            const baseH = Math.max(0, padTop + chartHeight - baseY);

            const fineY = getY(m.fineTuned);
            const fineH = Math.max(0, padTop + chartHeight - fineY);

            const shortLabels = [
              "Novel mAP@0.5",
              "COCO Shared",
              "Overall 20-Class",
              "COCO [.5:.95]",
            ];

            return (
              <g key={m.id}>
                {/* Baseline bar */}
                <rect
                  x={baselineX}
                  y={baseY}
                  width={barWidth}
                  height={baseH}
                  rx="2"
                  className="fill-text-muted/40 hover:fill-text-muted/60 transition-colors cursor-pointer"
                  onMouseEnter={() => {
                    setHovered({
                      metric: m,
                      type: "baseline",
                      value: m.baseline,
                      svgX: baselineX + barWidth / 2,
                      svgY: baseY,
                    });
                  }}
                  onMouseLeave={() => setHovered(null)}
                />
                {/* Baseline direct label */}
                <text
                  x={baselineX + barWidth / 2}
                  y={baseY - 4}
                  textAnchor="middle"
                  className="fill-text-muted text-[10px] font-mono tabular-nums"
                >
                  {m.baseline.toFixed(3)}
                </text>

                {/* Fine-Tuned bar */}
                <rect
                  x={fineTunedX}
                  y={fineY}
                  width={barWidth}
                  height={fineH}
                  rx="2"
                  className="fill-brand hover:opacity-90 transition-opacity cursor-pointer"
                  onMouseEnter={() => {
                    setHovered({
                      metric: m,
                      type: "fineTuned",
                      value: m.fineTuned,
                      svgX: fineTunedX + barWidth / 2,
                      svgY: fineY,
                    });
                  }}
                  onMouseLeave={() => setHovered(null)}
                />
                {/* Fine-Tuned direct label */}
                <text
                  x={fineTunedX + barWidth / 2}
                  y={fineY - 4}
                  textAnchor="middle"
                  className="fill-brand text-[10px] font-mono font-medium tabular-nums"
                >
                  {m.fineTuned.toFixed(3)}
                </text>

                {/* X-axis group label */}
                <text
                  x={groupX + barWidth + gap / 2}
                  y={padTop + chartHeight + 18}
                  textAnchor="middle"
                  className="fill-text-secondary text-[11px] font-medium"
                >
                  {shortLabels[idx]}
                </text>
                {/* Delta badge text */}
                <text
                  x={groupX + barWidth + gap / 2}
                  y={padTop + chartHeight + 32}
                  textAnchor="middle"
                  className="fill-brand text-[10px] font-mono"
                >
                  ↑ {m.deltaPts}
                </text>
              </g>
            );
          })}
          {/* In-SVG vector tooltip: responsive, respects viewport, cannot clip */}
          {hovered && (
            <g
              className="pointer-events-none transition-all duration-150"
              transform={`translate(${hovered.svgX}, ${Math.max(48, hovered.svgY - 10)})`}
            >
              <rect
                x="-65"
                y="-44"
                width="130"
                height="44"
                rx="4"
                className="fill-card stroke-border-subtle"
                strokeWidth="1"
              />
              <text
                x="0"
                y="-30"
                textAnchor="middle"
                className="fill-text-primary text-[10px] font-semibold"
              >
                {hovered.metric.name}
              </text>
              <text
                x="0"
                y="-18"
                textAnchor="middle"
                className="fill-text-secondary text-[9px]"
              >
                {hovered.type === "baseline" ? "Baseline" : "Fine-Tuned"}:{" "}
                <tspan className="font-mono font-medium fill-text-primary">
                  {hovered.value.toFixed(4)}
                </tspan>
              </text>
              <text
                x="0"
                y="-7"
                textAnchor="middle"
                className="fill-brand text-[8px] font-mono"
              >
                Target: {hovered.metric.target}
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
};
