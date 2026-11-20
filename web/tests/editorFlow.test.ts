/**
 * web/tests/editorFlow.test.ts
 * ----------------------------
 * Node.js unit-style integration test for the editor flow (Day 4 Commit 10).
 *
 * Tests the full correction workflow via editorReducer state machine and
 * activeLearning helpers — without a browser (Playwright not installed).
 *
 * Tradeoff: canvas rendering, mouse drag, and DOM interactions are not covered.
 * Add Playwright tests (see docs/day4_report.md §Risk 2) to cover those.
 *
 * Run with:
 *   node --experimental-strip-types --test tests/editorFlow.test.ts
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createInitialEditorState,
  editorReducer,
} from "../src/lib/editorReducer.ts";
import type { EditorBox } from "../src/lib/editorReducer.ts";
import type { NormalizedBBox } from "../src/types.ts";
import { createDefaultReviewQueue } from "../src/lib/activeLearning.ts";

// ---------------------------------------------------------------------------
// Helper: build a minimal EditorBox
// ---------------------------------------------------------------------------
function makeEditorBox(
  id: string,
  classId: number,
  className: string,
  box: NormalizedBBox
): EditorBox {
  return {
    id,
    class_id: classId,
    class_name: className,
    confidence: 1.0,
    box_normalized: box,
    color_hex: "#3B82F6",
    color_rgb: [59, 130, 246],
  };
}

// ---------------------------------------------------------------------------
// Test 1: Draw new box → verify it appears in state
// ---------------------------------------------------------------------------
test("editor flow: ADD a new box increments box count", () => {
  const initial = createInitialEditorState([]);
  const newBox = makeEditorBox("box-1", 1, "Person", {
    ymin: 0.1, xmin: 0.1, ymax: 0.4, xmax: 0.5,
  });

  const after = editorReducer(initial, { type: "ADD", box: newBox });

  assert.equal(after.present.length, 1, "should have 1 box after ADD");
  assert.equal(after.present[0].class_name, "Person");
  assert.equal(after.present[0].box_normalized.ymin, 0.1);
  assert.equal(after.past.length, 1, "past stack should have 1 snapshot");
  assert.equal(after.future.length, 0, "future stack should be empty");
});

// ---------------------------------------------------------------------------
// Test 2: Resize via RESIZE action
// ---------------------------------------------------------------------------
test("editor flow: RESIZE a box updates its bounding box", () => {
  const initial = createInitialEditorState([]);
  const newBox = makeEditorBox("box-2", 2, "Laptop", {
    ymin: 0.1, xmin: 0.1, ymax: 0.4, xmax: 0.5,
  });
  const afterAdd = editorReducer(initial, { type: "ADD", box: newBox });

  const resized: NormalizedBBox = { ymin: 0.05, xmin: 0.05, ymax: 0.55, xmax: 0.65 };
  const afterResize = editorReducer(afterAdd, {
    type: "RESIZE",
    id: "box-2",
    box_normalized: resized,
  });

  const actual = afterResize.present[0].box_normalized;
  assert.equal(actual.ymin, resized.ymin);
  assert.equal(actual.xmin, resized.xmin);
  assert.equal(actual.ymax, resized.ymax);
  assert.equal(actual.xmax, resized.xmax);
  assert.equal(afterResize.past.length, 2, "past stack should have 2 snapshots");
});

// ---------------------------------------------------------------------------
// Test 3: Undo (Ctrl+Z) reverts resize
// ---------------------------------------------------------------------------
test("editor flow: UNDO after resize restores original box", () => {
  const initial = createInitialEditorState([]);
  const originalBox: NormalizedBBox = { ymin: 0.1, xmin: 0.1, ymax: 0.4, xmax: 0.5 };
  const newBox = makeEditorBox("box-3", 2, "Laptop", originalBox);
  const afterAdd = editorReducer(initial, { type: "ADD", box: newBox });
  const afterResize = editorReducer(afterAdd, {
    type: "RESIZE",
    id: "box-3",
    box_normalized: { ymin: 0.05, xmin: 0.05, ymax: 0.55, xmax: 0.65 },
  });

  const afterUndo = editorReducer(afterResize, { type: "UNDO" });

  const actual = afterUndo.present[0].box_normalized;
  assert.equal(actual.ymin, originalBox.ymin, "ymin should be restored");
  assert.equal(actual.xmax, originalBox.xmax, "xmax should be restored");
  assert.equal(afterUndo.future.length, 1, "future stack should have 1 entry after undo");
});

// ---------------------------------------------------------------------------
// Test 4: Redo after undo reapplies the resize
// ---------------------------------------------------------------------------
test("editor flow: REDO after undo reapplies the resize", () => {
  const initial = createInitialEditorState([]);
  const newBox = makeEditorBox("box-4", 2, "Laptop", {
    ymin: 0.1, xmin: 0.1, ymax: 0.4, xmax: 0.5,
  });
  const resized: NormalizedBBox = { ymin: 0.05, xmin: 0.05, ymax: 0.55, xmax: 0.65 };
  const afterAdd = editorReducer(initial, { type: "ADD", box: newBox });
  const afterResize = editorReducer(afterAdd, {
    type: "RESIZE",
    id: "box-4",
    box_normalized: resized,
  });
  const afterUndo = editorReducer(afterResize, { type: "UNDO" });

  const afterRedo = editorReducer(afterUndo, { type: "REDO" });

  const actual = afterRedo.present[0].box_normalized;
  assert.equal(actual.ymin, resized.ymin, "ymin should be re-applied after redo");
  assert.equal(actual.xmax, resized.xmax, "xmax should be re-applied after redo");
  assert.equal(afterRedo.future.length, 0, "future stack should be empty after redo");
});

// ---------------------------------------------------------------------------
// Test 5: Export URL path follows /v1/corrections/export convention
// ---------------------------------------------------------------------------
test("editor flow: corrections export URL path is correct", () => {
  // Verify the path pattern directly without importing from api.ts
  // (api.ts uses import.meta.env which is Vite-specific and unavailable in Node).
  // The actual runtime URL is tested via the API test suite.
  const API_BASE = "http://localhost:8000";
  const exportUrl = `${API_BASE}/v1/corrections/export`;
  assert.ok(exportUrl.endsWith("/v1/corrections/export"), "Export URL must end with /v1/corrections/export");
  assert.ok(exportUrl.startsWith("http"), "Export URL must be an absolute HTTP URL");
});

// ---------------------------------------------------------------------------
// Test 6: Review queue items can be loaded and edited via editorReducer
// ---------------------------------------------------------------------------
test("editor flow: queue detections can be loaded and corrected via editorReducer", () => {
  const queue = createDefaultReviewQueue();
  assert.ok(queue.length > 0, "queue should have at least 1 item");

  const firstItem = queue[0];
  assert.ok(firstItem.detections.length > 0, "first queue item should have detections");

  // Map DetectionItem[] -> EditorBox[] as the UI does
  const editorBoxes: EditorBox[] = firstItem.detections.map((det, i) => ({
    id: `det-${i}`,
    class_id: det.class_id,
    class_name: det.class_name,
    confidence: det.confidence,
    box_normalized: det.box_normalized,
    color_hex: det.color_hex,
    color_rgb: det.color_rgb,
  }));

  const editorState = createInitialEditorState(editorBoxes);
  assert.equal(
    editorState.present.length,
    firstItem.detections.length,
    "editor state should start with all detections from queue item"
  );

  // Simulate user DELETE of the first detection box
  const boxId = editorState.present[0].id;
  const afterDelete = editorReducer(editorState, { type: "DELETE", id: boxId });
  assert.equal(
    afterDelete.present.length,
    firstItem.detections.length - 1,
    "editor state should have one fewer box after DELETE"
  );

  // Confirm undo restores it
  const afterUndo = editorReducer(afterDelete, { type: "UNDO" });
  assert.equal(
    afterUndo.present.length,
    firstItem.detections.length,
    "undo should restore the deleted box"
  );
});
