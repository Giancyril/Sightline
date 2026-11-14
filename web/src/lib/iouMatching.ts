import type { DetectionItem, NormalizedBBox } from "../types";

export function computeIoU(a: NormalizedBBox, b: NormalizedBBox): number {
  const interY1 = Math.max(a.ymin, b.ymin);
  const interX1 = Math.max(a.xmin, b.xmin);
  const interY2 = Math.min(a.ymax, b.ymax);
  const interX2 = Math.min(a.xmax, b.xmax);

  const interH = Math.max(0, interY2 - interY1);
  const interW = Math.max(0, interX2 - interX1);
  const intersection = interH * interW;

  const areaA = Math.max(0, (a.ymax - a.ymin) * (a.xmax - a.xmin));
  const areaB = Math.max(0, (b.ymax - b.ymin) * (b.xmax - b.xmin));
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

export interface MatchedComparison {
  matched: Array<{ detA: DetectionItem; detB: DetectionItem; iou: number }>;
  onlyA: DetectionItem[];
  onlyB: DetectionItem[];
}

export function matchDetections(
  detsA: DetectionItem[],
  detsB: DetectionItem[],
  iouThreshold = 0.45,
  classAware = true
): MatchedComparison {
  const candidates: Array<{ iou: number; i: number; j: number }> = [];

  for (let i = 0; i < detsA.length; i++) {
    for (let j = 0; j < detsB.length; j++) {
      if (
        classAware &&
        detsA[i].class_name.toLowerCase() !== detsB[j].class_name.toLowerCase()
      ) {
        continue;
      }
      const overlap = computeIoU(
        detsA[i].box_normalized,
        detsB[j].box_normalized
      );
      if (overlap >= iouThreshold) {
        candidates.push({ iou: overlap, i, j });
      }
    }
  }

  candidates.sort((a, b) => b.iou - a.iou);

  const matchedA = new Set<number>();
  const matchedB = new Set<number>();
  const matched: Array<{
    detA: DetectionItem;
    detB: DetectionItem;
    iou: number;
  }> = [];

  for (const { iou, i, j } of candidates) {
    if (matchedA.has(i) || matchedB.has(j)) continue;
    matchedA.add(i);
    matchedB.add(j);
    matched.push({
      detA: detsA[i],
      detB: detsB[j],
      iou: Math.round(iou * 100) / 100,
    });
  }

  const onlyA = detsA.filter((_, i) => !matchedA.has(i));
  const onlyB = detsB.filter((_, j) => !matchedB.has(j));

  return { matched, onlyA, onlyB };
}

