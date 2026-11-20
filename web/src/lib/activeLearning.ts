/**
 * web/src/lib/activeLearning.ts
 * -----------------------------
 * Active learning uncertainty ranking and review queue utilities (Day 4 Commit 9).
 * Reuses Day 1 compare bipartite IoU matching and Day 3 decision thresholds.
 */

import { matchDetections } from "./iouMatching.ts";
import type { DetectionItem, ReviewQueueItem, ReviewStatus } from "../types";

export interface UncertaintyScoreResult {
  score: number;
  reasons: string[];
  thresholdScore: number;
  disagreementScore: number;
  numNearThreshold: number;
  numDisagreements: number;
}

/**
 * Score the uncertainty of an image's detections.
 * Evaluates:
 *   1. Bounding boxes near decision thresholds (default margin 0.15 around threshold).
 *   2. Model disagreements (unmatched boxes between primary and secondary detections via IoU bipartite matching).
 */
export function scoreItemUncertainty(
  primaryDetections: DetectionItem[],
  secondaryDetections?: DetectionItem[],
  thresholds: Record<string, number> = {},
  globalThreshold: number = 0.25,
  thresholdMargin: number = 0.15
): UncertaintyScoreResult {
  let thresholdScore = 0;
  let numNearThreshold = 0;
  const nearThresholdDetails: string[] = [];

  // 1. Evaluate proximity to decision thresholds
  for (const det of primaryDetections) {
    const th = thresholds[det.class_name] ?? globalThreshold;
    const delta = Math.abs(det.confidence - th);

    if (delta <= thresholdMargin) {
      const proximity = Math.max(0, 1 - delta / thresholdMargin);
      thresholdScore += proximity;
      numNearThreshold += 1;
      nearThresholdDetails.push(
        `${det.class_name} (${det.confidence.toFixed(2)} vs ${th.toFixed(2)}, Δ=${delta.toFixed(2)})`
      );
    }
  }

  // 2. Evaluate model disagreement via Day 1 bipartite IoU matching
  let disagreementScore = 0;
  let numDisagreements = 0;
  const disagreementDetails: string[] = [];

  if (secondaryDetections && secondaryDetections.length > 0) {
    const matchResult = matchDetections(primaryDetections, secondaryDetections, 0.45, true);

    for (const onlyA of matchResult.onlyA) {
      disagreementScore += onlyA.confidence;
      numDisagreements += 1;
      disagreementDetails.push(`Model A only: ${onlyA.class_name} (${onlyA.confidence.toFixed(2)})`);
    }

    for (const onlyB of matchResult.onlyB) {
      disagreementScore += onlyB.confidence;
      numDisagreements += 1;
      disagreementDetails.push(`Model B only: ${onlyB.class_name} (${onlyB.confidence.toFixed(2)})`);
    }
  }

  const score = Number((thresholdScore + disagreementScore).toFixed(4));
  const reasons: string[] = [];

  if (numNearThreshold > 0) {
    const detailStr = nearThresholdDetails.slice(0, 3).join("; ");
    const moreSuffix = nearThresholdDetails.length > 3 ? `; +${nearThresholdDetails.length - 3} more` : "";
    reasons.push(`${numNearThreshold} detection(s) near decision threshold: ${detailStr}${moreSuffix}`);
  }

  if (numDisagreements > 0) {
    const detailStr = disagreementDetails.slice(0, 3).join("; ");
    const moreSuffix = disagreementDetails.length > 3 ? `; +${disagreementDetails.length - 3} more` : "";
    reasons.push(`${numDisagreements} model disagreement(s): ${detailStr}${moreSuffix}`);
  }

  if (reasons.length === 0) {
    reasons.push("High confidence agreement: clean detections without borderline scores or disagreements");
  }

  return {
    score,
    reasons,
    thresholdScore: Number(thresholdScore.toFixed(4)),
    disagreementScore: Number(disagreementScore.toFixed(4)),
    numNearThreshold,
    numDisagreements,
  };
}

/**
 * Score and sort queue items descending by uncertainty score.
 */
export function rankQueueItems(
  items: Omit<ReviewQueueItem, "score" | "rank" | "reasons" | "thresholdScore" | "disagreementScore" | "numNearThreshold" | "numDisagreements">[],
  thresholds: Record<string, number> = {},
  globalThreshold: number = 0.25
): ReviewQueueItem[] {
  const scored = items.map((item) => {
    const result = scoreItemUncertainty(
      item.detections,
      item.detectionsSecondary,
      thresholds,
      globalThreshold
    );
    return {
      ...item,
      ...result,
      rank: 0,
    };
  });

  // Sort descending by: score, disagreements, near-threshold, then id
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    if (b.numDisagreements !== a.numDisagreements) return b.numDisagreements - a.numDisagreements;
    if (b.numNearThreshold !== a.numNearThreshold) return b.numNearThreshold - a.numNearThreshold;
    return a.id.localeCompare(b.id);
  });

  return scored.map((item, index) => ({
    ...item,
    rank: index + 1,
  }));
}

/**
 * Generate a default active learning review pool using the application's real sample images.
 */
export function createDefaultReviewQueue(): ReviewQueueItem[] {
  const rawItems: Array<{
    id: string;
    imageSrc: string;
    imageName: string;
    imageHash: string;
    imageDimensions: { width: number; height: number };
    detections: DetectionItem[];
    detectionsSecondary?: DetectionItem[];
    status: ReviewStatus;
  }> = [
    {
      id: "queue_sample_1",
      imageSrc: "/samples/sample-workspace.jpg",
      imageName: "sample-workspace.jpg",
      imageHash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      imageDimensions: { width: 1200, height: 800 },
      status: "pending",
      // Disagreement and borderline detection
      detections: [
        {
          class_id: 1,
          class_name: "Person",
          confidence: 0.27, // near 0.25 (delta 0.02)
          box_normalized: { ymin: 0.12, xmin: 0.24, ymax: 0.78, xmax: 0.58 },
          box_pixel: { ymin: 96, xmin: 288, ymax: 624, xmax: 696, width: 408, height: 528 },
          color_hex: "#3B82F6",
          color_rgb: [59, 130, 246],
        },
        {
          class_id: 2,
          class_name: "Laptop",
          confidence: 0.88,
          box_normalized: { ymin: 0.52, xmin: 0.45, ymax: 0.75, xmax: 0.72 },
          box_pixel: { ymin: 416, xmin: 540, ymax: 600, xmax: 864, width: 324, height: 184 },
          color_hex: "#10B981",
          color_rgb: [16, 185, 129],
        },
        {
          class_id: 3,
          class_name: "Chair",
          confidence: 0.24, // near 0.25 (delta 0.01)
          box_normalized: { ymin: 0.40, xmin: 0.15, ymax: 0.85, xmax: 0.42 },
          box_pixel: { ymin: 320, xmin: 180, ymax: 680, xmax: 504, width: 324, height: 360 },
          color_hex: "#F59E0B",
          color_rgb: [245, 158, 11],
        },
      ],
      detectionsSecondary: [
        {
          class_id: 2,
          class_name: "Laptop",
          confidence: 0.90,
          box_normalized: { ymin: 0.52, xmin: 0.45, ymax: 0.75, xmax: 0.72 },
          box_pixel: { ymin: 416, xmin: 540, ymax: 600, xmax: 864, width: 324, height: 184 },
          color_hex: "#10B981",
          color_rgb: [16, 185, 129],
        },
        {
          class_id: 4,
          class_name: "Book", // Disagreement: secondary predicted Book
          confidence: 0.76,
          box_normalized: { ymin: 0.65, xmin: 0.70, ymax: 0.82, xmax: 0.90 },
          box_pixel: { ymin: 520, xmin: 840, ymax: 656, xmax: 1080, width: 240, height: 136 },
          color_hex: "#EC4899",
          color_rgb: [236, 72, 153],
        },
      ],
    },
    {
      id: "queue_sample_2",
      imageSrc: "/samples/sample-desk.jpg",
      imageName: "sample-desk.jpg",
      imageHash: "b2d2f7a934b1d62c11ef842996fb92427ae41e4649b934ca495991b7852b8221",
      imageDimensions: { width: 1024, height: 768 },
      status: "pending",
      detections: [
        {
          class_id: 2,
          class_name: "Laptop",
          confidence: 0.94,
          box_normalized: { ymin: 0.35, xmin: 0.28, ymax: 0.72, xmax: 0.68 },
          box_pixel: { ymin: 268, xmin: 286, ymax: 552, xmax: 696, width: 410, height: 284 },
          color_hex: "#10B981",
          color_rgb: [16, 185, 129],
        },
        {
          class_id: 6,
          class_name: "Cell phone",
          confidence: 0.26, // near 0.25 (delta 0.01)
          box_normalized: { ymin: 0.60, xmin: 0.72, ymax: 0.80, xmax: 0.85 },
          box_pixel: { ymin: 460, xmin: 737, ymax: 614, xmax: 870, width: 133, height: 154 },
          color_hex: "#8B5CF6",
          color_rgb: [139, 92, 246],
        },
      ],
      detectionsSecondary: [
        {
          class_id: 2,
          class_name: "Laptop",
          confidence: 0.92,
          box_normalized: { ymin: 0.35, xmin: 0.28, ymax: 0.72, xmax: 0.68 },
          box_pixel: { ymin: 268, xmin: 286, ymax: 552, xmax: 696, width: 410, height: 284 },
          color_hex: "#10B981",
          color_rgb: [16, 185, 129],
        },
      ],
    },
    {
      id: "queue_sample_3",
      imageSrc: "/samples/sample-street.jpg",
      imageName: "sample-street.jpg",
      imageHash: "fa19c28a8d11c4e723bf842996fb92427ae41e4649b934ca495991b7852b8990",
      imageDimensions: { width: 1280, height: 720 },
      status: "pending",
      detections: [
        {
          class_id: 9,
          class_name: "Car",
          confidence: 0.96,
          box_normalized: { ymin: 0.45, xmin: 0.18, ymax: 0.78, xmax: 0.62 },
          box_pixel: { ymin: 324, xmin: 230, ymax: 561, xmax: 793, width: 563, height: 237 },
          color_hex: "#EF4444",
          color_rgb: [239, 68, 68],
        },
      ],
      detectionsSecondary: [
        {
          class_id: 9,
          class_name: "Car",
          confidence: 0.95,
          box_normalized: { ymin: 0.45, xmin: 0.18, ymax: 0.78, xmax: 0.62 },
          box_pixel: { ymin: 324, xmin: 230, ymax: 561, xmax: 793, width: 563, height: 237 },
          color_hex: "#EF4444",
          color_rgb: [239, 68, 68],
        },
      ],
    },
  ];

  return rankQueueItems(rawItems);
}
