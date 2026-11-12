import React, { useState, useMemo } from "react";
import { ArrowUpDown } from "lucide-react";
import { PER_CLASS_EVALUATION, type ClassEvaluation } from "../../data/evaluationData";
import { colorForClass } from "../../lib/colors";
import { cn } from "../../lib/utils";

type SortOption = "ap-desc" | "ap-asc" | "alpha" | "category";

export const PerClassApBars: React.FC = () => {
  const [sortOption, setSortOption] = useState<SortOption>("ap-desc");

  const sortedClasses = useMemo(() => {
    const list = [...PER_CLASS_EVALUATION];
    switch (sortOption) {
      case "ap-desc":
        return list.sort((a, b) => b.finetuned_ap - a.finetuned_ap);
      case "ap-asc":
        return list.sort((a, b) => a.finetuned_ap - b.finetuned_ap);
      case "alpha":
        return list.sort((a, b) => a.name.localeCompare(b.name));
      case "category":
        return list.sort((a, b) => {
          if (a.is_coco === b.is_coco) {
            return b.finetuned_ap - a.finetuned_ap;
          }
          return a.is_coco ? -1 : 1;
        });
      default:
        return list;
    }
  }, [sortOption]);

  const maxAp = 0.5; // Book is 0.4309, max scale at 0.50

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">
            Per-Class Average Precision (AP@0.5)
          </h4>
          <p className="text-xs text-text-muted mt-0.5">
            Ranked detection accuracy across all 20 target classes on validation set
          </p>
        </div>

        {/* Sort Controls */}
        <div className="flex items-center gap-1.5 self-start sm:self-auto">
          <ArrowUpDown className="w-3.5 h-3.5 text-text-tertiary" />
          <span className="text-xs text-text-muted">Sort:</span>
          <div className="flex items-center rounded-control bg-subtle p-0.5 border border-border-subtle text-xs">
            <button
              type="button"
              onClick={() => setSortOption("ap-desc")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] font-medium transition-colors",
                sortOption === "ap-desc"
                  ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                  : "text-text-secondary hover:text-text-primary"
              )}
            >
              Highest AP
            </button>
            <button
              type="button"
              onClick={() => setSortOption("ap-asc")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] font-medium transition-colors",
                sortOption === "ap-asc"
                  ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                  : "text-text-secondary hover:text-text-primary"
              )}
            >
              Lowest
            </button>
            <button
              type="button"
              onClick={() => setSortOption("category")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] font-medium transition-colors",
                sortOption === "category"
                  ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                  : "text-text-secondary hover:text-text-primary"
              )}
            >
              Category
            </button>
            <button
              type="button"
              onClick={() => setSortOption("alpha")}
              className={cn(
                "px-2 py-0.5 rounded-[4px] font-medium transition-colors",
                sortOption === "alpha"
                  ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                  : "text-text-secondary hover:text-text-primary"
              )}
            >
              A–Z
            </button>
          </div>
        </div>
      </div>

      {/* Horizontal Bar Chart Container */}
      <div className="relative rounded-panel border border-border-subtle bg-card p-3 sm:p-5">
        {/* Scale header / vertical reference gridlines */}
        <div className="relative mb-3 flex items-center justify-between text-[10px] font-mono text-text-muted border-b border-border-subtle pb-1.5 pl-24 sm:pl-32 pr-16 sm:pr-24">
          <span>0.00</span>
          <span>0.10</span>
          <span>0.20</span>
          <span>0.30</span>
          <span>0.40</span>
          <span>0.50</span>
        </div>

        {/* List of class bars */}
        <div className="flex flex-col gap-1.5">
          {sortedClasses.map((item: ClassEvaluation) => {
            const classColor = colorForClass(item.id).hex;
            const percentage = Math.min(100, Math.max(0, (item.finetuned_ap / maxAp) * 100));
            const baselinePct = Math.min(100, Math.max(0, (item.baseline_ap / maxAp) * 100));

            return (
              <div
                key={item.id}
                className="group flex items-center gap-2 sm:gap-3 py-1 hover:bg-elevated/40 rounded-control px-1 sm:px-1.5 transition-colors"
              >
                {/* Class Label Column */}
                <div className="w-24 sm:w-32 flex-shrink-0 flex items-center gap-1.5 overflow-hidden">
                  <span
                    className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: classColor }}
                    aria-hidden="true"
                  />
                  <span className="text-xs font-medium text-text-primary truncate">
                    {item.name}
                  </span>
                  <span className="text-[10px] text-text-muted font-mono hidden md:inline">
                    #{item.id.toString().padStart(2, "0")}
                  </span>
                </div>

                {/* Horizontal Bar Track */}
                <div className="relative flex-grow h-4 bg-subtle rounded-[3px] overflow-hidden flex items-center">
                  {/* Vertical hairline gridlines in background */}
                  <div className="absolute inset-0 flex justify-between pointer-events-none opacity-30">
                    <div className="w-px h-full bg-border-subtle" style={{ left: "20%" }} />
                    <div className="w-px h-full bg-border-subtle" style={{ left: "40%" }} />
                    <div className="w-px h-full bg-border-subtle" style={{ left: "60%" }} />
                    <div className="w-px h-full bg-border-subtle" style={{ left: "80%" }} />
                  </div>

                  {/* Baseline tick marker if applicable */}
                  {item.baseline_ap > 0 && (
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-text-muted/60 z-10"
                      style={{ left: `${baselinePct}%` }}
                      title={`Baseline AP: ${item.baseline_ap.toFixed(4)}`}
                    />
                  )}

                  {/* Filled bar in class color */}
                  <div
                    className="h-full rounded-[2px] transition-all duration-300"
                    style={{
                      width: `${percentage}%`,
                      backgroundColor: classColor,
                    }}
                  />
                </div>

                {/* Values & Metadata Column */}
                <div className="w-16 sm:w-24 flex-shrink-0 flex items-center justify-between font-mono text-xs">
                  <span className="text-text-primary font-medium tabular-nums">
                    {item.finetuned_ap.toFixed(4)}
                  </span>
                  <span
                    className={cn(
                      "text-[10px] hidden sm:inline tabular-nums",
                      item.delta >= 0 ? "text-brand" : "text-text-muted"
                    )}
                  >
                    {item.deltaPts}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* IBM Carbon Footer note */}
        <div className="mt-4 pt-3 border-t border-border-subtle flex flex-wrap items-center justify-between gap-2 text-[11px] text-text-muted">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-brand" />
              <span>Fine-Tuned AP (bar width)</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-1 h-3 bg-text-muted" />
              <span>COCO Baseline marker</span>
            </span>
          </div>
          <span>Direct labels &middot; Zero decoration &middot; Okabe-Ito Palette</span>
        </div>
      </div>
    </div>
  );
};
