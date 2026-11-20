import React, { useCallback, useEffect, useRef, useState } from "react";
import { RotateCcw, AlertCircle, ChevronDown, ChevronUp } from "lucide-react";
import type { ClassThresholdResponseItem } from "../../types";
import { fetchModelThresholds } from "../../api";
import { Slider } from "../ui/Slider";
import { Skeleton } from "../ui/Skeleton";
import { cn } from "../../lib/utils";

interface PerClassThresholdEditorProps {
  modelId: string;
  /** Caller-controlled custom overrides: class_name -> threshold */
  customThresholds: Record<string, number>;
  onChange: (overrides: Record<string, number>) => void;
  className?: string;
}

export const PerClassThresholdEditor: React.FC<PerClassThresholdEditorProps> = ({
  modelId,
  customThresholds,
  onChange,
  className,
}) => {
  const [rows, setRows] = useState<ClassThresholdResponseItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);
  const prevModelId = useRef<string | null>(null);

  const load = useCallback((id: string) => {
    setIsLoading(true);
    setError(null);
    fetchModelThresholds(id)
      .then((data) => {
        setRows(data.per_class);
        setIsLoading(false);
        // Seed custom thresholds from recommended if not yet overridden
        const seed: Record<string, number> = {};
        for (const row of data.per_class) {
          if (!(row.class_name in customThresholds)) {
            seed[row.class_name] = row.threshold;
          }
        }
        if (Object.keys(seed).length > 0) {
          onChange({ ...customThresholds, ...seed });
        }
      })
      .catch((err) => {
        setError((err as Error).message || "Failed to load thresholds");
        setIsLoading(false);
      });
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (prevModelId.current !== modelId) {
      prevModelId.current = modelId;
      load(modelId);
    }
  }, [modelId, load]);

  const handleReset = useCallback(
    (className: string, recommended: number) => {
      onChange({ ...customThresholds, [className]: recommended });
    },
    [customThresholds, onChange]
  );

  const handleResetAll = useCallback(() => {
    const reset: Record<string, number> = {};
    for (const row of rows) {
      reset[row.class_name] = row.threshold;
    }
    onChange(reset);
  }, [rows, onChange]);

  const handleSliderChange = useCallback(
    (className: string, value: number) => {
      onChange({ ...customThresholds, [className]: value });
    },
    [customThresholds, onChange]
  );

  const handleInputChange = useCallback(
    (className: string, raw: string) => {
      const v = parseFloat(raw) / 100;
      if (!isNaN(v) && v >= 0 && v <= 1) {
        onChange({ ...customThresholds, [className]: Math.round(v * 100) / 100 });
      }
    },
    [customThresholds, onChange]
  );

  if (isLoading) {
    return (
      <div className={cn("flex flex-col gap-2 py-2", className)}>
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-full" />
        <Skeleton className="h-6 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div
        className={cn(
          "flex items-start gap-2 p-2.5 rounded-control bg-status-warning-subtle border border-status-warning/30 text-xs text-status-warning",
          className
        )}
      >
        <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
        <span className="leading-relaxed">{error}</span>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className={cn("text-xs text-text-tertiary py-2 text-center", className)}>
        No per-class thresholds available.
      </div>
    );
  }

  return (
    <div className={cn("flex flex-col gap-0", className)}>
      {/* Editor header */}
      <div className="flex items-center justify-between mb-2">
        <button
          type="button"
          onClick={() => setCollapsed((c) => !c)}
          className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-text-tertiary hover:text-text-primary transition-colors"
          aria-expanded={!collapsed}
        >
          {collapsed ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronUp className="w-3 h-3" />
          )}
          Per-class thresholds
        </button>
        <button
          type="button"
          onClick={handleResetAll}
          className="flex items-center gap-1 text-[10px] text-text-tertiary hover:text-brand transition-colors"
          aria-label="Reset all to recommended"
          title="Reset all to recommended"
        >
          <RotateCcw className="w-3 h-3" />
          Reset all
        </button>
      </div>

      {!collapsed && (
        <div
          className="flex flex-col gap-0 overflow-y-auto max-h-[280px] pr-0.5"
          role="list"
          aria-label="Per-class confidence threshold editor"
        >
          {rows.map((row) => {
            const current = customThresholds[row.class_name] ?? row.threshold;
            const isModified =
              Math.abs(current - row.threshold) > 0.001;

            return (
              <div
                key={row.class_name}
                role="listitem"
                className="flex flex-col gap-1.5 py-2 border-b border-border-subtle last:border-b-0"
              >
                {/* Class row header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5 min-w-0">
                    <span
                      className="w-2 h-2 rounded-full shrink-0"
                      style={{ backgroundColor: `#${Math.abs(row.class_id * 2654435761) % 0xFFFFFF | 0x444444}` }}
                      aria-hidden="true"
                    />
                    <span className="text-[11px] font-medium text-text-primary truncate">
                      {row.class_name}
                    </span>
                    {row.fell_back && (
                      <span className="text-[10px] text-text-tertiary font-mono shrink-0">(global)</span>
                    )}
                  </div>
                  <div className="flex items-center gap-1 shrink-0 ml-2">
                    {/* Numeric input */}
                    <input
                      type="number"
                      min={5}
                      max={95}
                      step={1}
                      value={Math.round(current * 100)}
                      onChange={(e) => handleInputChange(row.class_name, e.target.value)}
                      aria-label={`${row.class_name} threshold percentage`}
                      className="w-11 h-6 text-center font-mono text-[11px] bg-card border border-border-subtle rounded-control text-text-primary focus:outline-none focus:ring-1 focus:ring-brand focus:border-brand [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
                    />
                    <span className="text-[10px] text-text-tertiary font-mono">%</span>
                    {/* Reset to recommended */}
                    <button
                      type="button"
                      disabled={!isModified}
                      onClick={() => handleReset(row.class_name, row.threshold)}
                      className={cn(
                        "p-0.5 rounded transition-colors",
                        isModified
                          ? "text-brand hover:text-brand-hover hover:bg-brand/10"
                          : "text-transparent pointer-events-none"
                      )}
                      aria-label={`Reset ${row.class_name} to recommended (${Math.round(row.threshold * 100)}%)`}
                      title={`Reset to recommended (${Math.round(row.threshold * 100)}%)`}
                    >
                      <RotateCcw className="w-3 h-3" />
                    </button>
                  </div>
                </div>

                {/* Slider */}
                <Slider
                  value={current}
                  min={0.05}
                  max={0.95}
                  step={0.01}
                  onValueChange={(v) => handleSliderChange(row.class_name, v)}
                  formatValue={() => ""}
                />

                {/* Metrics mini row: P / R / F1 at recommended vs at current */}
                <div className="flex items-center gap-3 text-[10px] font-mono text-text-tertiary">
                  <span>
                    P{" "}
                    <span className="text-text-secondary">
                      {(row.precision * 100).toFixed(0)}%
                    </span>
                  </span>
                  <span>
                    R{" "}
                    <span className="text-text-secondary">
                      {(row.recall * 100).toFixed(0)}%
                    </span>
                  </span>
                  <span>
                    F1{" "}
                    <span className="text-text-secondary">
                      {(row.f1 * 100).toFixed(0)}%
                    </span>
                  </span>
                  {!row.fell_back && (
                    <span className="ml-auto text-brand font-sans text-[9px] font-medium shrink-0">
                      rec. {Math.round(row.threshold * 100)}%
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
