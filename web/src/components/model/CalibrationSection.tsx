/**
 * CalibrationSection — lazy-loaded reliability diagram + before/after metrics.
 *
 * Data is sourced from the real calibration.json and thresholds.json
 * files via the /v1/models/{id}/thresholds API endpoint.
 * When a model has no calibration file, an empty state is shown.
 */
import React, { useEffect, useState } from "react";
import { Info, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { fetchModelThresholds } from "../../api";
import type { ModelThresholdsResponse } from "../../types";
import { Skeleton } from "../ui/Skeleton";
import { Badge } from "../ui/Badge";

// ---------------------------------------------------------------------------
// Types for the calibration JSON file (fetched from the API route)
// ---------------------------------------------------------------------------
interface CalibrationBin {
  bin_lower: number;
  bin_upper: number;
  mean_confidence: number;
  accuracy: number;
  count: number;
}

interface CalibrationData {
  model_id: string;
  temperature: number;
  ece_before: number;
  ece_after: number;
  nll_before?: number;
  nll_after?: number;
  date: string;
  data_hash: string;
  bins_before: CalibrationBin[];
  bins_after: CalibrationBin[];
}

// ---------------------------------------------------------------------------
// Reliability Diagram — pure SVG, no external chart library
// ---------------------------------------------------------------------------
function ReliabilityDiagram({
  binsBefore,
  binsAfter,
}: {
  binsBefore: CalibrationBin[];
  binsAfter: CalibrationBin[];
}) {
  const W = 320;
  const H = 220;
  const PAD = { top: 16, right: 16, bottom: 40, left: 44 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  const toX = (v: number) => PAD.left + v * IW;
  const toY = (v: number) => PAD.top + (1 - v) * IH;

  // Bar width proportional to 1/numBins
  const n = binsBefore.length;
  const barW = IW / n;

  const gridLines = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full max-w-sm"
      role="img"
      aria-label="Reliability diagram showing confidence calibration before and after temperature scaling"
    >
      {/* Grid */}
      {gridLines.map((v) => (
        <line
          key={v}
          x1={PAD.left}
          y1={toY(v)}
          x2={W - PAD.right}
          y2={toY(v)}
          stroke="currentColor"
          strokeOpacity={0.08}
          strokeWidth={1}
        />
      ))}
      {gridLines.map((v) => (
        <text
          key={v}
          x={PAD.left - 6}
          y={toY(v) + 4}
          textAnchor="end"
          fontSize={9}
          fill="currentColor"
          fillOpacity={0.4}
          fontFamily="monospace"
        >
          {(v * 100).toFixed(0)}%
        </text>
      ))}

      {/* Perfect calibration diagonal */}
      <line
        x1={toX(0)}
        y1={toY(0)}
        x2={toX(1)}
        y2={toY(1)}
        stroke="currentColor"
        strokeOpacity={0.2}
        strokeWidth={1}
        strokeDasharray="4 3"
      />

      {/* Bins before calibration */}
      {binsBefore.map((b, i) => {
        const x = PAD.left + i * barW + barW * 0.08;
        const bw = barW * 0.4;
        const acc = Math.max(0, Math.min(1, b.accuracy));
        const bh = acc * IH;
        return b.count > 0 ? (
          <rect
            key={`before-${i}`}
            x={x}
            y={toY(acc)}
            width={bw}
            height={bh}
            fill="#94a3b8"
            fillOpacity={0.5}
            rx={1}
          />
        ) : null;
      })}

      {/* Bins after calibration */}
      {binsAfter.map((b, i) => {
        const x = PAD.left + i * barW + barW * 0.52;
        const bw = barW * 0.4;
        const acc = Math.max(0, Math.min(1, b.accuracy));
        const bh = acc * IH;
        return b.count > 0 ? (
          <rect
            key={`after-${i}`}
            x={x}
            y={toY(acc)}
            width={bw}
            height={bh}
            fill="#6366f1"
            fillOpacity={0.75}
            rx={1}
          />
        ) : null;
      })}

      {/* X-axis label ticks */}
      {binsBefore.map((b, i) => (
        <text
          key={`xt-${i}`}
          x={toX(b.bin_lower + 0.05)}
          y={H - PAD.bottom + 14}
          textAnchor="middle"
          fontSize={8}
          fill="currentColor"
          fillOpacity={0.4}
          fontFamily="monospace"
        >
          {(b.bin_lower * 100).toFixed(0)}
        </text>
      ))}

      {/* Axis labels */}
      <text
        x={W / 2}
        y={H - 2}
        textAnchor="middle"
        fontSize={9}
        fill="currentColor"
        fillOpacity={0.5}
      >
        Confidence score
      </text>
      <text
        x={8}
        y={H / 2}
        textAnchor="middle"
        fontSize={9}
        fill="currentColor"
        fillOpacity={0.5}
        transform={`rotate(-90, 8, ${H / 2})`}
      >
        Accuracy
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Before/After row for P / R / F1
// ---------------------------------------------------------------------------
interface PRFRow {
  class_name: string;
  class_id: number;
  threshold: number;
  precision_raw: number;
  recall_raw: number;
  f1_raw: number;
  precision: number;
  recall: number;
  f1: number;
}

function DeltaBadge({ delta, unit = "pp" }: { delta: number; unit?: string }) {
  const abs = Math.abs(delta);
  if (abs < 0.5) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[10px] font-mono text-text-tertiary">
        <Minus className="w-2.5 h-2.5" />
        0
      </span>
    );
  }
  if (delta > 0) {
    return (
      <span className="inline-flex items-center gap-0.5 text-[10px] font-mono text-status-success">
        <TrendingUp className="w-2.5 h-2.5" />+{abs.toFixed(0)}
        {unit}
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-0.5 text-[10px] font-mono text-status-warning">
      <TrendingDown className="w-2.5 h-2.5" />
      {abs.toFixed(0)}
      {unit}
    </span>
  );
}

function PerClassTable({ rows }: { rows: PRFRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px] border-collapse">
        <thead>
          <tr className="border-b border-border-subtle text-left">
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] pr-3 w-28">
              Class
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-16">
              Threshold
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-20">
              Precision
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-20">
              Recall
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-16">
              F1
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-24">
              ΔPrecision
            </th>
            <th className="pb-2 font-semibold text-text-tertiary uppercase tracking-wider text-[10px] text-center w-20">
              ΔRecall
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle">
          {rows.map((row) => {
            const dP = (row.precision - row.precision_raw) * 100;
            const dR = (row.recall - row.recall_raw) * 100;
            return (
              <tr key={row.class_name} className="hover:bg-subtle/40 transition-colors">
                <td className="py-1.5 pr-3 font-medium text-text-primary truncate max-w-[112px]">
                  {row.class_name}
                </td>
                <td className="py-1.5 text-center font-mono text-text-secondary">
                  {Math.round(row.threshold * 100)}%
                </td>
                <td className="py-1.5 text-center font-mono text-text-primary">
                  {(row.precision * 100).toFixed(0)}%
                </td>
                <td className="py-1.5 text-center font-mono text-text-secondary">
                  {(row.recall * 100).toFixed(0)}%
                </td>
                <td className="py-1.5 text-center font-mono text-text-primary font-medium">
                  {(row.f1 * 100).toFixed(0)}%
                </td>
                <td className="py-1.5 text-center">
                  <DeltaBadge delta={dP} />
                </td>
                <td className="py-1.5 text-center">
                  <DeltaBadge delta={dR} />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main exported component
// ---------------------------------------------------------------------------
interface CalibrationSectionProps {
  modelId: string;
}

export const CalibrationSection: React.FC<CalibrationSectionProps> = ({ modelId }) => {
  const [thresholds, setThresholds] = useState<ModelThresholdsResponse | null>(null);
  const [calibration, setCalibration] = useState<CalibrationData | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const t = await fetchModelThresholds(modelId);
        if (active) setThresholds(t);
      } catch (e) {
        const msg = (e as Error).message;
        if (active) {
          if (!msg.includes("No threshold file")) setError(msg);
        }
      }

      try {
        const r = await fetch(`/calibration/${modelId}_calibration.json`);
        if (r.ok) {
          const data = (await r.json()) as CalibrationData;
          if (active) setCalibration(data);
        }
      } catch {
        // no calibration file for this model — empty state handled below
      } finally {
        if (active) setIsLoading(false);
      }
    };
    void load();
    return () => { active = false; };
  }, [modelId]);

  const rows: PRFRow[] = thresholds?.per_class.map((c) => ({
    class_name: c.class_name,
    class_id: c.class_id,
    threshold: c.threshold,
    precision_raw: c.precision_raw,
    recall_raw: c.recall_raw,
    f1_raw: c.f1_raw,
    precision: c.precision,
    recall: c.recall,
    f1: c.f1,
  })) ?? [];

  if (isLoading) {
    return (
      <div className="rounded-panel border border-border-subtle bg-card p-4 sm:p-5 flex flex-col gap-4">
        <Skeleton className="h-5 w-56" />
        <div className="flex gap-4 flex-wrap">
          <Skeleton className="h-48 w-72" />
          <div className="flex flex-col gap-2 flex-1">
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-4/5" />
          </div>
        </div>
      </div>
    );
  }

  // Empty state — model has no calibration files
  if (!thresholds && !calibration && !error) {
    return (
      <div className="rounded-panel border border-border-subtle bg-card p-4 sm:p-5 flex flex-col items-center justify-center gap-3 text-center py-8">
        <div className="flex h-10 w-10 items-center justify-center rounded-control bg-elevated border border-border-subtle text-text-tertiary">
          <Info className="w-5 h-5" />
        </div>
        <div>
          <p className="text-sm font-semibold text-text-primary">No calibration data</p>
          <p className="text-xs text-text-muted mt-1 max-w-xs leading-relaxed">
            Calibration files are not available for this model. Only the fine-tuned model ships with
            per-class thresholds and calibration.
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-panel border border-status-warning/30 bg-status-warning-subtle p-4 text-xs text-status-warning">
        {error}
      </div>
    );
  }

  const eceBefore = calibration?.ece_before ?? thresholds?.per_class[0]?.f1_raw ?? null;
  const eceAfter = calibration?.ece_after ?? null;
  const temperature = calibration?.temperature ?? null;

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h3 className="text-sm font-semibold text-text-primary">Score Calibration</h3>
        <p className="text-xs text-text-muted leading-relaxed max-w-2xl">
          Scores are calibrated on validation data using temperature scaling (T ={" "}
          <span className="font-mono">{temperature?.toFixed(3) ?? "—"}</span>). A well-calibrated
          model means a 70% confidence score reflects roughly 70% real-world accuracy. Bars near
          the diagonal are ideal.
        </p>
      </div>

      {/* ECE badges */}
      {eceBefore !== null && eceAfter !== null && (
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-control bg-subtle border border-border-subtle text-xs">
            <span className="text-text-muted">ECE before:</span>
            <span className="font-mono font-semibold text-text-primary">
              {(eceBefore * 100).toFixed(2)}%
            </span>
          </div>
          <span className="text-text-tertiary text-xs">→</span>
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-control bg-brand/5 border border-brand/20 text-xs">
            <span className="text-text-muted">ECE after:</span>
            <span className="font-mono font-semibold text-brand">
              {(eceAfter * 100).toFixed(2)}%
            </span>
          </div>
          <Badge variant="brand" className="text-[10px]">
            {(((eceBefore - eceAfter) / eceBefore) * 100).toFixed(0)}% lower error
          </Badge>
        </div>
      )}

      {/* Chart + legend */}
      {calibration && (
        <div className="flex flex-col sm:flex-row gap-6 items-start">
          <div className="shrink-0 w-full sm:w-72 text-text-primary">
            <ReliabilityDiagram
              binsBefore={calibration.bins_before}
              binsAfter={calibration.bins_after}
            />
          </div>
          <div className="flex flex-col gap-3 text-xs text-text-secondary leading-relaxed min-w-0">
            {/* Legend */}
            <div className="flex items-center gap-4 text-[11px]">
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-slate-400/50" />
                <span>Before (raw)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-3 h-3 rounded-sm bg-indigo-500/75" />
                <span>After (calibrated)</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-6 border-t border-dashed border-current opacity-30" />
                <span>Ideal</span>
              </div>
            </div>
            <p>
              <strong className="text-text-primary">How to read this chart:</strong> Each bar
              shows the fraction of predictions in that confidence bucket that were actually
              correct (Intersection-over-Union ≥ 0.5). The dashed diagonal is perfect
              calibration. Bars to the right of the diagonal mean the model was over-confident;
              bars to the left mean under-confident.
            </p>
            <p>
              Scores shown in the Detect inspector are calibrated on validation data. The raw
              score order is preserved — calibration only adjusts magnitude, not ranking.
            </p>
          </div>
        </div>
      )}

      {/* Per-class before/after table */}
      {rows.length > 0 && (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-0.5">
            <h4 className="text-xs font-semibold text-text-primary">
              Per-class: Precision / Recall / F1 — global vs recommended threshold
            </h4>
            <p className="text-[11px] text-text-muted leading-snug">
              ΔPrecision is the gain from using the recommended threshold instead of 0.25 global.
              A negative ΔRecall shows how much recall is traded for that precision gain.
            </p>
          </div>
          <PerClassTable rows={rows} />
        </div>
      )}
    </div>
  );
};

export default CalibrationSection;
