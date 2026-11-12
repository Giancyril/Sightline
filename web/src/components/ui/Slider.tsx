import React from "react";
import * as RadixSlider from "@radix-ui/react-slider";
import { cn } from "../../lib/utils";

export interface SliderProps {
  label?: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onValueChange: (value: number) => void;
  formatValue?: (value: number) => string;
  className?: string;
  disabled?: boolean;
}

export const Slider: React.FC<SliderProps> = ({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onValueChange,
  formatValue = (v) => `${(v * 100).toFixed(0)}%`,
  className,
  disabled = false,
}) => {
  return (
    <div className={cn("flex flex-col gap-1.5 w-full select-none", className)}>
      {(label || formatValue !== undefined) && (
        <div className="flex items-center justify-between text-xs">
          {label && (
            <span className="font-medium text-text-secondary">{label}</span>
          )}
          <span className="font-mono text-text-primary tabular-nums font-medium text-[11px] px-1.5 py-0.5 rounded-chip bg-elevated border border-border-subtle">
            {formatValue(value)}
          </span>
        </div>
      )}

      <RadixSlider.Root
        disabled={disabled}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(vals) => onValueChange(vals[0])}
        className={cn(
          "relative flex items-center select-none touch-none w-full h-5 cursor-pointer",
          disabled && "opacity-50 cursor-not-allowed"
        )}
      >
        <RadixSlider.Track className="bg-elevated border border-border-subtle relative grow rounded-full h-1.5 overflow-hidden">
          <RadixSlider.Range className="absolute bg-brand h-full rounded-full" />
        </RadixSlider.Track>
        <RadixSlider.Thumb
          aria-label={label || "Slider"}
          className="block w-4 h-4 bg-card border border-border-strong rounded-full shadow-sm hover:scale-105 transition-transform focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-canvas"
        />
      </RadixSlider.Root>
    </div>
  );
};
