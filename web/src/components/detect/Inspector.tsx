import React, { useMemo } from "react";
import {
  Sliders,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Inbox,
  Clock,
  Layers,
  Box,
} from "lucide-react";
import type { DetectionItem } from "../../types";
import { colorForClass } from "../../lib/colors";
import { Slider } from "../ui/Slider";
import { Button } from "../ui/Button";
import { Skeleton } from "../ui/Skeleton";
import { cn } from "../../lib/utils";

interface InspectorProps {
  detections: DetectionItem[];
  allDetections: DetectionItem[];
  inferenceMs: number | null;
  confThreshold: number;
  nmsThreshold: number;
  onConfChange: (val: number) => void;
  onNmsChange: (val: number) => void;
  selectedIndex: number | null;
  onSelectIndex: (idx: number | null) => void;
  isLoading: boolean;
  errorMsg: string | null;
  onRetry: () => void;
  isApiOffline?: boolean;
  className?: string;
}

export const Inspector: React.FC<InspectorProps> = ({
  detections,
  inferenceMs,
  confThreshold,
  nmsThreshold,
  onConfChange,
  onNmsChange,
  selectedIndex,
  onSelectIndex,
  isLoading,
  errorMsg,
  onRetry,
  isApiOffline = false,
  className,
}) => {
  // Unique classes count
  const uniqueClassesCount = useMemo(() => {
    return new Set(detections.map((d) => d.class_id)).size;
  }, [detections]);

  // Loading skeleton state
  if (isLoading) {
    return (
      <aside
        className={cn(
          "flex flex-col h-full bg-card border border-border-subtle rounded-panel p-4 gap-4 select-none",
          className
        )}
      >
        <div className="flex items-center justify-between border-b border-border-subtle pb-3">
          <Skeleton className="h-4 w-28" />
          <Skeleton className="h-4 w-14" />
        </div>

        {/* Summary skeleton */}
        <div className="grid grid-cols-3 gap-2 py-2 border-b border-border-subtle">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>

        {/* Sliders skeleton */}
        <div className="flex flex-col gap-3 py-1">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>

        {/* Table skeleton */}
        <div className="flex flex-col gap-2 pt-2 flex-1">
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
          <Skeleton className="h-9 w-full" />
        </div>
      </aside>
    );
  }

  // Error state
  if (errorMsg) {
    return (
      <aside
        className={cn(
          "flex flex-col items-center justify-center h-full bg-card border border-border-subtle rounded-panel p-6 text-center select-none",
          className
        )}
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-control bg-status-danger-subtle text-status-danger mb-3">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <h4 className="text-sm font-semibold text-text-primary mb-1">
          Inference error
        </h4>
        <p className="text-xs text-text-secondary max-w-xs mb-4 leading-relaxed font-mono text-[11px]">
          {errorMsg}
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
        >
          Retry detection
        </Button>
      </aside>
    );
  }

  // API Offline state
  if (isApiOffline) {
    return (
      <aside
        className={cn(
          "flex flex-col items-center justify-center h-full bg-card border border-border-subtle rounded-panel p-6 text-center select-none",
          className
        )}
      >
        <div className="flex h-10 w-10 items-center justify-center rounded-control bg-status-warning-subtle text-status-warning mb-3">
          <AlertTriangle className="w-5 h-5" />
        </div>
        <h4 className="text-sm font-semibold text-text-primary mb-1">
          Backend API offline
        </h4>
        <p className="text-xs text-text-secondary max-w-xs mb-4 leading-relaxed">
          Cannot connect to the FastAPI inference server. Verify the server is running.
        </p>
        <Button
          variant="secondary"
          size="sm"
          onClick={onRetry}
          leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
        >
          Check again
        </Button>
      </aside>
    );
  }

  return (
    <aside
      className={cn(
        "flex flex-col h-full bg-card border border-border-subtle rounded-panel overflow-hidden",
        className
      )}
    >
      {/* 1. Header & Summary Strip */}
      <div className="p-3.5 border-b border-border-subtle bg-subtle/50">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">
            Inspection
          </span>
          <span className="text-[11px] font-mono text-text-secondary">
            SSD MobileNet V2
          </span>
        </div>

        {/* 3 Summary metrics */}
        <div className="grid grid-cols-3 gap-2">
          <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
            <span className="text-[10px] text-text-tertiary flex items-center gap-1">
              <Box className="w-3 h-3 opacity-60" /> Objects
            </span>
            <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
              {detections.length}
            </span>
          </div>

          <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
            <span className="text-[10px] text-text-tertiary flex items-center gap-1">
              <Layers className="w-3 h-3 opacity-60" /> Classes
            </span>
            <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
              {uniqueClassesCount}
            </span>
          </div>

          <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
            <span className="text-[10px] text-text-tertiary flex items-center gap-1">
              <Clock className="w-3 h-3 opacity-60" /> Latency
            </span>
            <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
              {inferenceMs !== null ? `${inferenceMs.toFixed(1)} ms` : "—"}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Interactive Parameter Sliders */}
      <div className="p-3.5 border-b border-border-subtle flex flex-col gap-3">
        <div className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
          <Sliders className="w-3.5 h-3.5" />
          <span>Thresholds</span>
        </div>

        {/* Live Client-Filtering Confidence Slider */}
        <Slider
          label="Confidence threshold"
          value={confThreshold}
          min={0.05}
          max={0.95}
          step={0.05}
          onValueChange={onConfChange}
          formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        />

        {/* NMS IoU Slider */}
        <Slider
          label="NMS IoU threshold"
          value={nmsThreshold}
          min={0.10}
          max={0.90}
          step={0.05}
          onValueChange={onNmsChange}
          formatValue={(v) => `${(v * 100).toFixed(0)}%`}
        />
      </div>

      {/* 3. Detections Table or Empty State */}
      <div className="flex-1 overflow-y-auto p-2">
        {detections.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
            <div className="flex h-9 w-9 items-center justify-center rounded-control bg-elevated border border-border-subtle text-text-tertiary mb-2">
              <Inbox className="w-4 h-4" />
            </div>
            <p className="text-xs font-semibold text-text-primary mb-1">
              No objects detected
            </p>
            <p className="text-[11px] text-text-tertiary max-w-[220px] mb-3 leading-relaxed">
              No boxes met the {(confThreshold * 100).toFixed(0)}% confidence threshold.
            </p>
            {confThreshold > 0.15 && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onConfChange(0.15)}
                leftIcon={<Sparkles className="w-3.5 h-3.5 text-brand" />}
              >
                Lower to 15%
              </Button>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-1" role="table" aria-label="Detected objects">
            {/* Column Headers */}
            <div className="flex items-center justify-between px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider text-text-tertiary select-none">
              <span>Class</span>
              <div className="flex items-center gap-6">
                <span>Score</span>
                <span className="w-20 text-right">Box (px)</span>
              </div>
            </div>

            {/* Rows */}
            {detections.map((det, idx) => {
              const isSelected = selectedIndex === idx;
              const classCol = colorForClass(det.class_id);
              const scorePct = Math.round(det.confidence * 100);

              return (
                <button
                  key={idx}
                  type="button"
                  onClick={() => onSelectIndex(isSelected ? null : idx)}
                  className={cn(
                    "flex items-center justify-between w-full px-2.5 py-1.5 rounded-control text-left transition-colors border select-none",
                    isSelected
                      ? "bg-elevated border-brand shadow-sm"
                      : "bg-card border-transparent hover:bg-subtle hover:border-border-subtle"
                  )}
                >
                  {/* Left: Class dot + Class Name */}
                  <div className="flex items-center gap-2 min-w-0 pr-2">
                    <span
                      className="w-2.5 h-2.5 rounded-full shrink-0"
                      style={{ backgroundColor: classCol.hex }}
                    />
                    <span className="text-xs font-medium text-text-primary truncate">
                      {det.class_name}
                    </span>
                  </div>

                  {/* Right: Score Bar + Coordinates */}
                  <div className="flex items-center gap-4 shrink-0">
                    {/* Score Bar + Text */}
                    <div className="flex items-center gap-1.5">
                      <div className="w-12 h-1.5 rounded-full bg-elevated overflow-hidden border border-border-subtle">
                        <div
                          className="h-full rounded-full transition-all"
                          style={{
                            width: `${scorePct}%`,
                            backgroundColor: classCol.hex,
                          }}
                        />
                      </div>
                      <span className="font-mono text-[11px] font-medium text-text-primary tabular-nums min-w-[32px] text-right">
                        {scorePct}%
                      </span>
                    </div>

                    {/* Box Dimensions */}
                    <span className="font-mono text-[10px] text-text-tertiary tabular-nums w-20 text-right truncate">
                      {det.box_pixel.width}×{det.box_pixel.height}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
};
