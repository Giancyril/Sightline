import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Sliders,
  AlertTriangle,
  RotateCcw,
  Sparkles,
  Inbox,
  Clock,
  Layers,
  Box,
  X,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Pin,
  Cpu,
} from "lucide-react";
import type { DetectionItem, ModelRegistryItem } from "../../types";
import { fetchModels } from "../../api";
import { colorForClass } from "../../lib/colors";
import { Slider } from "../ui/Slider";
import { Button } from "../ui/Button";
import { Skeleton } from "../ui/Skeleton";
import { Badge } from "../ui/Badge";
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
  hoveredIndex: number | null;
  onHoverIndex: (idx: number | null) => void;
  isLoading: boolean;
  errorMsg: string | null;
  onRetry: () => void;
  isApiOffline?: boolean;
  selectedModelId?: string;
  onModelChange?: (modelId: string) => void;
  className?: string;
}

type SortField = "score" | "class" | "area";

export const Inspector: React.FC<InspectorProps> = ({
  detections,
  inferenceMs,
  confThreshold,
  nmsThreshold,
  onConfChange,
  onNmsChange,
  selectedIndex,
  onSelectIndex,
  hoveredIndex,
  onHoverIndex,
  isLoading,
  errorMsg,
  onRetry,
  isApiOffline = false,
  selectedModelId = "fine-tuned",
  onModelChange,
  className,
}) => {
  // Model registry state
  const [models, setModels] = useState<ModelRegistryItem[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState<boolean>(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const handleRetryModels = useCallback(() => {
    setIsLoadingModels(true);
    setModelsError(null);
    fetchModels()
      .then((res) => {
        setModels(res.models);
      })
      .catch((err) => {
        setModelsError((err as Error).message || "Failed to load models");
      })
      .finally(() => {
        setIsLoadingModels(false);
      });
  }, []);

  useEffect(() => {
    let active = true;
    fetchModels()
      .then((res) => {
        if (active) {
          setModels(res.models);
          setIsLoadingModels(false);
        }
      })
      .catch((err) => {
        if (active) {
          setModelsError((err as Error).message || "Failed to load models");
          setIsLoadingModels(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const activeModel = useMemo(() => {
    return models.find((m) => m.id === selectedModelId) || models[0] || null;
  }, [models, selectedModelId]);

  // Class filter state
  const [selectedClassFilter, setSelectedClassFilter] = useState<string | null>(null);

  // Sorting state
  const [sortField, setSortField] = useState<SortField>("score");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  // Unique classes count & summary counts
  const classCounts = useMemo(() => {
    const map = new Map<string, { count: number; id: number; hex: string }>();
    detections.forEach((d) => {
      const existing = map.get(d.class_name);
      if (existing) {
        existing.count += 1;
      } else {
        const c = colorForClass(d.class_id);
        map.set(d.class_name, { count: 1, id: d.class_id, hex: c.hex });
      }
    });
    return Array.from(map.entries()).map(([name, data]) => ({
      name,
      count: data.count,
      id: data.id,
      hex: data.hex,
    }));
  }, [detections]);

  // Filter & Sort pipeline
  const processedDetections = useMemo(() => {
    let list = detections.map((det, origIdx) => ({ det, origIdx }));

    // 1. Class chip filter
    if (selectedClassFilter) {
      list = list.filter((item) => item.det.class_name === selectedClassFilter);
    }

    // 2. Sorting
    list.sort((a, b) => {
      let cmp = 0;
      if (sortField === "score") {
        cmp = a.det.confidence - b.det.confidence;
      } else if (sortField === "class") {
        cmp = a.det.class_name.localeCompare(b.det.class_name);
      } else if (sortField === "area") {
        const areaA = a.det.box_pixel.width * a.det.box_pixel.height;
        const areaB = b.det.box_pixel.width * b.det.box_pixel.height;
        cmp = areaA - areaB;
      }
      return sortDir === "asc" ? cmp : -cmp;
    });

    return list;
  }, [detections, selectedClassFilter, sortField, sortDir]);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const pinnedDetection = selectedIndex !== null ? detections[selectedIndex] : null;

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
        <div className="grid grid-cols-3 gap-2 py-2 border-b border-border-subtle">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
        <div className="flex flex-col gap-3 py-1">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
        <div className="flex flex-col gap-2 pt-2 flex-1">
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
      {/* aria-live region: announces detection count to screen readers */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {detections.length} detection{detections.length !== 1 ? "s" : ""} at current threshold
      </div>
      {/* 1. Header & Summary Strip */}
      <div className="p-3.5 border-b border-border-subtle bg-subtle/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">
            Inspection
          </span>
          {activeModel && (
            <Badge variant="neutral" mono>
              {activeModel.version}
            </Badge>
          )}
        </div>

        {/* Model Selector */}
        <div className="mb-3">
          {isLoadingModels ? (
            <Skeleton className="h-8 w-full rounded-control" />
          ) : modelsError ? (
            <div className="flex items-center justify-between p-2 rounded-control bg-status-danger-subtle border border-status-danger/20 text-status-danger text-xs">
              <span className="truncate">{modelsError}</span>
              <button
                type="button"
                onClick={handleRetryModels}
                className="underline text-[11px] font-medium shrink-0 ml-2 hover:opacity-80"
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="relative">
              <select
                id="model-selector"
                aria-label="Select detection model"
                value={selectedModelId}
                disabled={isLoading}
                onChange={(e) => onModelChange?.(e.target.value)}
                className="w-full h-8 pl-2.5 pr-8 py-1 text-xs font-medium bg-card text-text-primary border border-border-subtle rounded-control hover:border-border-muted focus:outline-none focus:ring-1 focus:ring-brand focus:border-brand cursor-pointer appearance-none transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name} ({m.num_classes} classes)
                  </option>
                ))}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-text-tertiary">
                <Cpu className="w-3.5 h-3.5" />
              </div>
            </div>
          )}
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
              {classCounts.length}
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

      {/* 3. Class Filter Chips */}
      {classCounts.length > 0 && (
        <div className="px-3.5 py-2.5 border-b border-border-subtle flex items-center gap-1.5 overflow-x-auto text-xs">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-text-tertiary mr-1 shrink-0">
            Filter:
          </span>
          {classCounts.map((c) => {
            const isSelected = selectedClassFilter === c.name;
            return (
              <button
                key={c.name}
                type="button"
                onClick={() =>
                  setSelectedClassFilter(isSelected ? null : c.name)
                }
                className={cn(
                  "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-chip text-[11px] font-medium border transition-colors shrink-0",
                  isSelected
                    ? "bg-brand text-white border-brand shadow-sm"
                    : "bg-elevated border-border-subtle text-text-secondary hover:text-text-primary hover:border-border-muted"
                )}
              >
                <span
                  className="w-1.5 h-1.5 rounded-full"
                  style={{ backgroundColor: c.hex }}
                />
                <span>{c.name}</span>
                <span className="font-mono text-[10px] opacity-70">
                  {c.count}
                </span>
              </button>
            );
          })}

          {selectedClassFilter && (
            <button
              type="button"
              onClick={() => setSelectedClassFilter(null)}
              className="text-[10px] text-text-tertiary hover:text-status-danger underline ml-1 shrink-0"
            >
              Clear
            </button>
          )}
        </div>
      )}

      {/* 4. Pinned Detection Detail Card */}
      {pinnedDetection && (
        <div className="p-3 border-b border-border-subtle bg-elevated/40 text-xs flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Pin className="w-3.5 h-3.5 text-brand" />
              <span className="font-semibold text-text-primary">
                {pinnedDetection.class_name}
              </span>
              <Badge variant="neutral" mono>
                #{(pinnedDetection.confidence * 100).toFixed(1)}%
              </Badge>
            </div>
            <button
              type="button"
              onClick={() => onSelectIndex(null)}
              className="p-1 rounded-control text-text-tertiary hover:text-text-primary hover:bg-elevated"
              aria-label="Unpin detection"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[11px] font-mono bg-card border border-border-subtle p-2 rounded-control">
            <div>
              <span className="text-text-tertiary block text-[10px]">Box Origin</span>
              <span>
                [{pinnedDetection.box_pixel.xmin}, {pinnedDetection.box_pixel.ymin}]
              </span>
            </div>
            <div>
              <span className="text-text-tertiary block text-[10px]">Size</span>
              <span>
                {pinnedDetection.box_pixel.width}×{pinnedDetection.box_pixel.height} px
              </span>
            </div>
            <div>
              <span className="text-text-tertiary block text-[10px]">Total Area</span>
              <span>
                {(
                  pinnedDetection.box_pixel.width * pinnedDetection.box_pixel.height
                ).toLocaleString()}{" "}
                px²
              </span>
            </div>
            <div>
              <span className="text-text-tertiary block text-[10px]">Class ID</span>
              <span>ID #{pinnedDetection.class_id}</span>
            </div>
          </div>
        </div>
      )}

      {/* 5. Detections Table Header (Sortable) */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border-subtle text-[10px] font-semibold uppercase tracking-wider text-text-tertiary select-none">
        <button
          type="button"
          onClick={() => handleSort("class")}
          className="flex items-center gap-1 hover:text-text-primary focus-visible:outline-none"
        >
          <span>Class</span>
          {sortField === "class" ? (
            sortDir === "asc" ? (
              <ArrowUp className="w-3 h-3" />
            ) : (
              <ArrowDown className="w-3 h-3" />
            )
          ) : (
            <ArrowUpDown className="w-3 h-3 opacity-40" />
          )}
        </button>

        <div className="flex items-center gap-6">
          <button
            type="button"
            onClick={() => handleSort("score")}
            className="flex items-center gap-1 hover:text-text-primary focus-visible:outline-none"
          >
            <span>Score</span>
            {sortField === "score" ? (
              sortDir === "asc" ? (
                <ArrowUp className="w-3 h-3" />
              ) : (
                <ArrowDown className="w-3 h-3" />
              )
            ) : (
              <ArrowUpDown className="w-3 h-3 opacity-40" />
            )}
          </button>

          <button
            type="button"
            onClick={() => handleSort("area")}
            className="flex items-center gap-1 hover:text-text-primary focus-visible:outline-none w-20 justify-end"
          >
            <span>Area</span>
            {sortField === "area" ? (
              sortDir === "asc" ? (
                <ArrowUp className="w-3 h-3" />
              ) : (
                <ArrowDown className="w-3 h-3" />
              )
            ) : (
              <ArrowUpDown className="w-3 h-3 opacity-40" />
            )}
          </button>
        </div>
      </div>

      {/* 6. Detections Table or Empty State (Exposed as accessible list) */}
      <div
        className="flex-1 overflow-y-auto p-2"
        role="region"
        aria-label="Detected objects list"
      >
        {processedDetections.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-10 px-4 text-center">
            <div className="flex h-9 w-9 items-center justify-center rounded-control bg-elevated border border-border-subtle text-text-tertiary mb-2">
              <Inbox className="w-4 h-4" />
            </div>
            <p className="text-xs font-semibold text-text-primary mb-1">
              No objects match filters
            </p>
            <p className="text-[11px] text-text-tertiary max-w-[220px] mb-3 leading-relaxed">
              {selectedClassFilter
                ? `No ${selectedClassFilter} detected above ${(confThreshold * 100).toFixed(0)}%.`
                : `No boxes met the ${(confThreshold * 100).toFixed(0)}% threshold.`}
            </p>
            {selectedClassFilter ? (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setSelectedClassFilter(null)}
              >
                Reset class filter
              </Button>
            ) : (
              confThreshold > 0.15 && (
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => onConfChange(0.15)}
                  leftIcon={<Sparkles className="w-3.5 h-3.5 text-brand" />}
                >
                  Lower to 15%
                </Button>
              )
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-1" role="list">
            {processedDetections.map(({ det, origIdx }) => {
              const isSelected = selectedIndex === origIdx;
              const isHovered = hoveredIndex === origIdx;
              const classCol = colorForClass(det.class_id);
              const scorePct = Math.round(det.confidence * 100);
              const areaPx = det.box_pixel.width * det.box_pixel.height;

              return (
                <button
                  key={origIdx}
                  type="button"
                  role="listitem"
                  aria-label={`${det.class_name}, confidence ${scorePct}%, box size ${det.box_pixel.width} by ${det.box_pixel.height} pixels, area ${areaPx} pixels squared`}
                  onClick={() => onSelectIndex(isSelected ? null : origIdx)}
                  onMouseEnter={() => onHoverIndex(origIdx)}
                  onMouseLeave={() => onHoverIndex(null)}
                  className={cn(
                    "flex items-center justify-between w-full px-2.5 py-1.5 rounded-control text-left transition-colors border select-none",
                    isSelected
                      ? "bg-elevated border-brand shadow-sm ring-1 ring-brand"
                      : isHovered
                      ? "bg-subtle border-border-muted"
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

                  {/* Right: Score Bar + Area Coordinates */}
                  <div className="flex items-center gap-4 shrink-0">
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
