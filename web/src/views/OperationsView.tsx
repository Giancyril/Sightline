import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock,
  Cpu,
  Pause,
  Play,
  RefreshCw,
  Zap,
} from "lucide-react";
import { fetchOperationsTelemetry } from "../api";
import type { OperationsTelemetry } from "../types";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { EmptyState } from "../components/ui/EmptyState";
import { Skeleton } from "../components/ui/Skeleton";
import { cn } from "../lib/utils";

interface OperationsViewProps {
  onNavigateToDetect?: () => void;
}

export const OperationsView: React.FC<OperationsViewProps> = ({
  onNavigateToDetect,
}) => {
  const [data, setData] = useState<OperationsTelemetry | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const timerRef = useRef<number | null>(null);

  const loadData = useCallback(async () => {
    try {
      const telemetry = await fetchOperationsTelemetry();
      setData(telemetry);
      setError(null);
      setLastUpdated(new Date());
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to load telemetry");
    } finally {
      setLoading(false);
    }
  }, []);

  // Polling setup: 3000ms interval, pauses when isPaused is true
  useEffect(() => {
    let mounted = true;
    void (async () => {
      try {
        const telemetry = await fetchOperationsTelemetry();
        if (mounted) {
          setData(telemetry);
          setError(null);
          setLastUpdated(new Date());
        }
      } catch (err: unknown) {
        if (mounted) {
          setError(err instanceof Error ? err.message : "Failed to load telemetry");
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    if (isPaused) {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
      return;
    }

    timerRef.current = window.setInterval(() => {
      void loadData();
    }, 3000);

    return () => {
      if (timerRef.current !== null) {
        window.clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [isPaused, loadData]);

  if (loading && !data) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Skeleton className="h-28 rounded-panel" />
          <Skeleton className="h-28 rounded-panel" />
          <Skeleton className="h-28 rounded-panel" />
          <Skeleton className="h-28 rounded-panel" />
        </div>
        <Skeleton className="h-72 rounded-panel" />
        <Skeleton className="h-72 rounded-panel" />
      </div>
    );
  }

  const hasRecordedData =
    data && (data.total_requests > 0 || data.drift.sample_count > 0);

  return (
    <div className="space-y-6">
      {/* Control bar: Refresh, Auto-refresh Pause/Play, Last Updated */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-panel bg-card border border-border-subtle">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Button
              variant={isPaused ? "primary" : "secondary"}
              size="sm"
              onClick={() => setIsPaused((prev) => !prev)}
              className="gap-1.5 h-8 text-xs font-mono"
            >
              {isPaused ? (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" /> Resume Stream
                </>
              ) : (
                <>
                  <Pause className="w-3.5 h-3.5 fill-current" /> Pause Stream
                </>
              )}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => void loadData()}
              className="gap-1.5 h-8 text-xs font-mono"
            >
              <RefreshCw className="w-3.5 h-3.5" /> Refresh
            </Button>
          </div>

          <Badge
            variant={isPaused ? "neutral" : "brand"}
            dot
            dotColor={isPaused ? "bg-text-tertiary" : "bg-brand animate-pulse"}
          >
            {isPaused ? "Polling Paused" : "Live Polling (3s)"}
          </Badge>
        </div>

        {lastUpdated && (
          <div className="flex items-center gap-1.5 text-xs text-text-tertiary font-mono">
            <Clock className="w-3.5 h-3.5" />
            <span>Updated {lastUpdated.toLocaleTimeString()}</span>
          </div>
        )}
      </div>

      {error && (
        <div className="p-4 rounded-panel bg-status-danger-subtle border border-status-danger/30 text-status-danger text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Top Telemetry KPI Cards */}
      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* 1. Model & Version */}
          <div className="p-4 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div className="flex items-center justify-between text-text-secondary text-xs mb-2">
              <span className="font-medium">Active Model</span>
              <Cpu className="w-4 h-4 text-text-tertiary" />
            </div>
            <div>
              <div className="text-base font-bold text-text-primary tracking-tight truncate">
                {data.model_id}
              </div>
              <div className="flex items-center gap-2 mt-1">
                <Badge variant="brand" mono>
                  {data.model_version}
                </Badge>
                <span className="text-[11px] text-text-tertiary truncate">
                  SSD MobileNet V2
                </span>
              </div>
            </div>
          </div>

          {/* 2. Latency p50 / p95 */}
          <div className="p-4 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div className="flex items-center justify-between text-text-secondary text-xs mb-2">
              <span className="font-medium">Latency (p50 / p95)</span>
              <Zap className="w-4 h-4 text-text-tertiary" />
            </div>
            <div>
              <div className="text-xl font-bold font-mono text-text-primary">
                {data.p50_latency_ms}{" "}
                <span className="text-xs font-normal text-text-tertiary">
                  / {data.p95_latency_ms} ms
                </span>
              </div>
              <div className="text-[11px] text-text-tertiary mt-1 font-mono">
                {data.total_requests} total requests processed
              </div>
            </div>
          </div>

          {/* 3. Error Rate */}
          <div className="p-4 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div className="flex items-center justify-between text-text-secondary text-xs mb-2">
              <span className="font-medium">Error Rate</span>
              <Activity className="w-4 h-4 text-text-tertiary" />
            </div>
            <div>
              <div className="flex items-baseline gap-2">
                <span
                  className={cn(
                    "text-xl font-bold font-mono",
                    data.error_rate_pct > 5
                      ? "text-status-danger"
                      : data.error_rate_pct > 0
                        ? "text-status-warning"
                        : "text-status-success"
                  )}
                >
                  {data.error_rate_pct}%
                </span>
                <Badge
                  variant={
                    data.error_rate_pct === 0
                      ? "success"
                      : data.error_rate_pct > 5
                        ? "danger"
                        : "warning"
                  }
                >
                  {data.error_rate_pct === 0 ? "Healthy" : "Elevated Errors"}
                </Badge>
              </div>
              <div className="text-[11px] text-text-tertiary mt-1 font-mono">
                {data.total_errors} errors / {data.total_requests} calls
              </div>
            </div>
          </div>

          {/* 4. Drift Status */}
          <div className="p-4 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div className="flex items-center justify-between text-text-secondary text-xs mb-2">
              <span className="font-medium">Confidence Drift</span>
              {data.drift.status === "warning" ? (
                <AlertTriangle className="w-4 h-4 text-status-warning" />
              ) : (
                <CheckCircle2 className="w-4 h-4 text-status-success" />
              )}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <Badge
                  variant={data.drift.status === "warning" ? "warning" : "success"}
                  dot
                >
                  {data.drift.status === "warning" ? "Drift Warning" : "Nominal"}
                </Badge>
                <span className="font-mono text-xs font-semibold text-text-primary">
                  PSI: {data.drift.psi.toFixed(4)}
                </span>
              </div>
              <div className="text-[11px] text-text-tertiary mt-1 font-mono">
                Threshold {data.drift.threshold} ({data.drift.sample_count} live samples)
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main View Area: Empty State vs Live Charts */}
      {!hasRecordedData ? (
        <EmptyState
          icon={<Activity className="w-8 h-8 text-brand" />}
          title="No Live Telemetry Yet"
          description="Inference metrics, latency profiles, and confidence drift telemetry are collected dynamically as detections execute. Run detections or execute the load test to stream live data."
          action={
            <Button
              onClick={onNavigateToDetect}
              className="gap-2 text-xs font-mono"
            >
              Go to Detect Tab →
            </Button>
          }
          className="my-8"
        />
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Latency p50 / p95 Over Time Timeline */}
          <div className="p-5 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-sm font-semibold text-text-primary">
                  Latency Timeline (Recent Requests)
                </h3>
                <span className="text-[11px] font-mono text-text-tertiary">
                  Target: &lt; 300 ms
                </span>
              </div>
              <p className="text-xs text-text-secondary mb-4 leading-relaxed">
                Wall-clock execution duration for inbound HTTP requests in chronological sequence.
              </p>
            </div>

            <div className="h-56 w-full flex items-center justify-center bg-subtle/50 rounded-control border border-border-subtle p-3">
              <LatencyChart timeline={data.timeline} />
            </div>

            <div className="flex items-center justify-between text-[11px] font-mono text-text-tertiary mt-3 pt-2 border-t border-border-subtle">
              <div className="flex items-center gap-3">
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm bg-brand" /> Request Latency
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5 bg-status-warning" /> p50 ({data.p50_latency_ms}ms)
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="w-2.5 h-0.5 bg-status-danger" /> p95 ({data.p95_latency_ms}ms)
                </span>
              </div>
              <span>{data.timeline.length} sampled points</span>
            </div>
          </div>

          {/* Confidence Histogram vs Validation Baseline */}
          <div className="p-5 rounded-panel bg-card border border-border-subtle flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-1">
                <h3 className="text-sm font-semibold text-text-primary">
                  Confidence Distribution vs Validation
                </h3>
                <Badge
                  variant={data.drift.status === "warning" ? "warning" : "success"}
                  mono
                >
                  PSI: {data.drift.psi.toFixed(4)}
                </Badge>
              </div>
              <p className="text-xs text-text-secondary mb-4 leading-relaxed">
                Comparison of live detection confidences (bars) vs the static validation baseline (lines) across 10 uniform intervals.
              </p>
            </div>

            <div className="h-56 w-full flex items-center justify-center bg-subtle/50 rounded-control border border-border-subtle p-3">
              <ConfidenceDistributionChart
                baseline={data.drift.baseline_distribution}
                live={data.drift.live_distribution}
                binEdges={data.drift.bin_edges}
              />
            </div>

            <div className="flex items-center justify-between text-[11px] font-mono text-text-tertiary mt-3 pt-2 border-t border-border-subtle">
              <div className="flex items-center gap-4">
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-brand" /> Live Detections
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="w-2.5 h-0.5 bg-text-secondary" /> Validation Baseline
                </span>
              </div>
              <span className="truncate max-w-[200px]" title={data.drift.message}>
                {data.drift.message}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ---------------------------------------------------------------------------
// SVG Latency Timeline Chart Component
// ---------------------------------------------------------------------------

function LatencyChart({
  timeline,
}: {
  timeline: Array<{
    timestamp: number;
    latency_ms: number;
    is_error: boolean;
  }>;
}) {
  if (!timeline || timeline.length === 0) {
    return (
      <div className="text-xs text-text-tertiary font-mono">
        No request samples in rolling window
      </div>
    );
  }

  const W = 460;
  const H = 180;
  const PAD = { top: 16, right: 16, bottom: 24, left: 36 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  const maxLatency = Math.max(
    50,
    Math.ceil((Math.max(...timeline.map((t) => t.latency_ms)) * 1.25) / 50) * 50
  );

  const toX = (idx: number) => {
    if (timeline.length === 1) return PAD.left + IW / 2;
    return PAD.left + (idx / (timeline.length - 1)) * IW;
  };

  const toY = (val: number) => {
    const clamped = Math.max(0, Math.min(maxLatency, val));
    return PAD.top + (1 - clamped / maxLatency) * IH;
  };

  const points = timeline.map((t, i) => `${toX(i)},${toY(t.latency_ms)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full select-none">
      {/* Horizontal grid lines */}
      {[0, 0.5, 1.0].map((frac) => {
        const yVal = frac * maxLatency;
        const y = toY(yVal);
        return (
          <g key={frac}>
            <line
              x1={PAD.left}
              x2={PAD.left + IW}
              y1={y}
              y2={y}
              stroke="var(--border-subtle, #333)"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={y + 3}
              textAnchor="end"
              className="text-[9px] fill-text-tertiary font-mono"
            >
              {Math.round(yVal)}
            </text>
          </g>
        );
      })}

      {/* SLA 300ms reference line if within bounds */}
      {300 <= maxLatency && (
        <g>
          <line
            x1={PAD.left}
            x2={PAD.left + IW}
            y1={toY(300)}
            y2={toY(300)}
            stroke="var(--status-danger, #ef4444)"
            strokeDasharray="2 2"
            strokeWidth={1}
            opacity={0.6}
          />
          <text
            x={PAD.left + IW - 4}
            y={toY(300) - 4}
            textAnchor="end"
            className="text-[8px] fill-status-danger font-mono"
          >
            SLA 300ms
          </text>
        </g>
      )}

      {/* Timeline Polyline */}
      <polyline
        fill="none"
        stroke="var(--brand, #3b82f6)"
        strokeWidth={2}
        points={points}
      />

      {/* Point dots */}
      {timeline.map((t, i) => (
        <circle
          key={i}
          cx={toX(i)}
          cy={toY(t.latency_ms)}
          r={t.is_error ? 4 : 2.5}
          fill={t.is_error ? "var(--status-danger, #ef4444)" : "var(--brand, #3b82f6)"}
          stroke="var(--bg-card, #111)"
          strokeWidth={1}
        >
          <title>{`${t.latency_ms} ms (${t.is_error ? "Error" : "200 OK"})`}</title>
        </circle>
      ))}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// SVG Confidence Distribution Comparison Chart Component
// ---------------------------------------------------------------------------

function ConfidenceDistributionChart({
  baseline,
  live,
  binEdges,
}: {
  baseline: number[];
  live: number[];
  binEdges: number[];
}) {
  const W = 460;
  const H = 180;
  const PAD = { top: 16, right: 16, bottom: 28, left: 36 };
  const IW = W - PAD.left - PAD.right;
  const IH = H - PAD.top - PAD.bottom;

  const numBins = Math.min(baseline.length, live.length, 10);
  const maxVal = Math.max(0.3, ...baseline, ...live);

  const toY = (val: number) => PAD.top + (1 - val / maxVal) * IH;
  const barSlotW = IW / numBins;
  const barW = Math.max(4, barSlotW * 0.65);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full select-none">
      {/* Background horizontal grid lines */}
      {[0, 0.5, 1.0].map((frac) => {
        const val = frac * maxVal;
        const y = toY(val);
        return (
          <g key={frac}>
            <line
              x1={PAD.left}
              x2={PAD.left + IW}
              y1={y}
              y2={y}
              stroke="var(--border-subtle, #333)"
              strokeDasharray="3 3"
              strokeWidth={1}
            />
            <text
              x={PAD.left - 6}
              y={y + 3}
              textAnchor="end"
              className="text-[9px] fill-text-tertiary font-mono"
            >
              {(val * 100).toFixed(0)}%
            </text>
          </g>
        );
      })}

      {/* Live Distribution Bars */}
      {Array.from({ length: numBins }).map((_, i) => {
        const xCenter = PAD.left + i * barSlotW + barSlotW / 2;
        const barH = ((live[i] || 0) / maxVal) * IH;
        const y = PAD.top + IH - barH;

        return (
          <g key={i}>
            <rect
              x={xCenter - barW / 2}
              y={y}
              width={barW}
              height={Math.max(1, barH)}
              fill="var(--brand, #3b82f6)"
              rx={2}
              opacity={0.85}
            >
              <title>{`Bin [${binEdges[i]}-${binEdges[i + 1]}]: Live ${(live[i] * 100).toFixed(1)}%`}</title>
            </rect>

            {/* X-axis bin interval labels */}
            <text
              x={xCenter}
              y={PAD.top + IH + 14}
              textAnchor="middle"
              className="text-[9px] fill-text-tertiary font-mono"
            >
              {binEdges[i]}
            </text>
          </g>
        );
      })}

      {/* Baseline Distribution Stepped Reference Line */}
      {Array.from({ length: numBins }).map((_, i) => {
        const x1 = PAD.left + i * barSlotW + 2;
        const x2 = PAD.left + (i + 1) * barSlotW - 2;
        const y = toY(baseline[i] || 0);

        return (
          <g key={`base-${i}`}>
            <line
              x1={x1}
              x2={x2}
              y1={y}
              y2={y}
              stroke="var(--text-secondary, #999)"
              strokeWidth={2}
              strokeDasharray="4 2"
            >
              <title>{`Baseline: ${(baseline[i] * 100).toFixed(1)}%`}</title>
            </line>
          </g>
        );
      })}
    </svg>
  );
}
