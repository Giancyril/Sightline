import test from "node:test";
import assert from "node:assert/strict";

import {
  scoreItemUncertainty,
  rankQueueItems,
  createDefaultReviewQueue,
} from "../src/lib/activeLearning.ts";
import type { DetectionItem, ReviewStatus } from "../src/types.ts";

function makeDet(
  class_id: number,
  class_name: string,
  confidence: number,
  ymin = 0.1,
  xmin = 0.1,
  ymax = 0.5,
  xmax = 0.5
): DetectionItem {
  return {
    class_id,
    class_name,
    confidence,
    box_normalized: { ymin, xmin, ymax, xmax },
    box_pixel: { ymin: 10, xmin: 10, ymax: 50, xmax: 50, width: 40, height: 40 },
    color_hex: "#3B82F6",
  };
}

test("scoreItemUncertainty detects borderline scores near threshold", () => {
  // Global threshold = 0.25, det at 0.26 (delta = 0.01)
  const detNear = [makeDet(1, "Person", 0.26)];
  const res = scoreItemUncertainty(detNear, undefined, {}, 0.25, 0.15);

  assert.equal(res.numNearThreshold, 1);
  assert.equal(res.numDisagreements, 0);
  assert.ok(res.thresholdScore > 0.9);
  assert.ok(res.reasons.some((r) => r.includes("near decision threshold")));
  assert.ok(res.reasons.some((r) => r.includes("Person")));
});

test("scoreItemUncertainty detects model disagreements from bipartite IoU matching", () => {
  const primary = [makeDet(1, "Person", 0.9, 0.1, 0.1, 0.5, 0.5)];
  const secondary = [makeDet(2, "Laptop", 0.85, 0.6, 0.6, 0.9, 0.9)];

  const res = scoreItemUncertainty(primary, secondary, {}, 0.25, 0.15);

  assert.equal(res.numDisagreements, 2);
  assert.ok(res.disagreementScore > 1.7);
  assert.ok(res.reasons.some((r) => r.includes("model disagreement")));
});

test("rankQueueItems sorts candidates descending by compound uncertainty score", () => {
  const items = [
    {
      id: "clean",
      imageSrc: "/samples/sample-street.jpg",
      imageName: "clean.jpg",
      imageHash: "hash1",
      imageDimensions: { width: 800, height: 600 },
      detections: [makeDet(9, "Car", 0.95)],
      detectionsSecondary: [makeDet(9, "Car", 0.94)],
      status: "pending" as ReviewStatus,
    },
    {
      id: "near_th",
      imageSrc: "/samples/sample-desk.jpg",
      imageName: "near_th.jpg",
      imageHash: "hash2",
      imageDimensions: { width: 800, height: 600 },
      detections: [makeDet(3, "Chair", 0.27)], // delta 0.02 vs 0.25
      status: "pending" as ReviewStatus,
    },
    {
      id: "disagreement",
      imageSrc: "/samples/sample-workspace.jpg",
      imageName: "disagree.jpg",
      imageHash: "hash3",
      imageDimensions: { width: 800, height: 600 },
      detections: [makeDet(1, "Person", 0.88, 0.1, 0.1, 0.5, 0.5)],
      detectionsSecondary: [makeDet(4, "Book", 0.82, 0.6, 0.6, 0.9, 0.9)],
      status: "pending" as ReviewStatus,
    },
  ];

  const ranked = rankQueueItems(items, {}, 0.25);

  assert.equal(ranked.length, 3);
  assert.equal(ranked[0].id, "disagreement");
  assert.equal(ranked[0].rank, 1);
  assert.equal(ranked[1].id, "near_th");
  assert.equal(ranked[1].rank, 2);
  assert.equal(ranked[2].id, "clean");
  assert.equal(ranked[2].rank, 3);
});

test("createDefaultReviewQueue initializes valid queue items with ranks and reasons", () => {
  const queue = createDefaultReviewQueue();
  assert.ok(queue.length >= 3);
  assert.equal(queue[0].rank, 1);
  assert.ok(queue[0].score >= queue[1].score);
  assert.ok(queue[0].reasons.length > 0);
  assert.equal(queue[0].status, "pending");
});
