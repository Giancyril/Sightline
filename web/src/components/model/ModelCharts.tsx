import React from "react";
import { GroupedMetricBars } from "./GroupedMetricBars";
import { PerClassApBars } from "./PerClassApBars";
import { PrecisionRecallCurve } from "./PrecisionRecallCurve";

export const ModelCharts: React.FC = () => {
  return (
    <div className="flex flex-col gap-8">
      {/* 1. Grouped Bars: Baseline vs Fine-Tuned */}
      <GroupedMetricBars />

      {/* 2. Precision-Recall Tradeoff Curve with Class Selector */}
      <PrecisionRecallCurve />

      {/* 3. Horizontal Per-Class AP Bars (Sorted) */}
      <PerClassApBars />
    </div>
  );
};

export default ModelCharts;
