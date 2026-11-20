/**
 * web/src/lib/editorReducer.test.ts
 * ---------------------------------
 * Unit tests for pure editor state machine: add, move, resize, relabel, delete, undo, redo.
 */

import test from "node:test";
import assert from "node:assert";
import {
  editorReducer,
  createInitialEditorState,
  canUndo,
  canRedo,
  clampBBox,
  moveBox,
  type EditorBox,
} from "../src/lib/editorReducer.ts";

const sampleBox1: EditorBox = {
  id: "box-1",
  class_id: 1,
  class_name: "Person",
  confidence: 0.92,
  box_normalized: { ymin: 0.1, xmin: 0.1, ymax: 0.5, xmax: 0.5 },
};

const sampleBox2: EditorBox = {
  id: "box-2",
  class_id: 2,
  class_name: "Laptop",
  confidence: 0.85,
  box_normalized: { ymin: 0.6, xmin: 0.6, ymax: 0.9, xmax: 0.9 },
};

test("createInitialEditorState initializes empty and non-empty states", () => {
  const empty = createInitialEditorState();
  assert.strictEqual(empty.present.length, 0);
  assert.strictEqual(empty.past.length, 0);
  assert.strictEqual(empty.future.length, 0);
  assert.strictEqual(empty.selectedId, null);
  assert.strictEqual(canUndo(empty), false);
  assert.strictEqual(canRedo(empty), false);

  const initial = createInitialEditorState([sampleBox1]);
  assert.strictEqual(initial.present.length, 1);
  assert.strictEqual(initial.present[0].id, "box-1");
});

test("ADD action appends box, selects it, and pushes previous state to past", () => {
  const initial = createInitialEditorState([sampleBox1]);
  const state1 = editorReducer(initial, { type: "ADD", box: sampleBox2 });

  assert.strictEqual(state1.present.length, 2);
  assert.strictEqual(state1.selectedId, "box-2");
  assert.strictEqual(state1.past.length, 1);
  assert.strictEqual(state1.past[0].length, 1);
  assert.strictEqual(canUndo(state1), true);
  assert.strictEqual(canRedo(state1), false);
});

test("MOVE action shifts coordinates and clamps within [0, 1] bounds", () => {
  const initial = createInitialEditorState([sampleBox1]);
  // Move right and down by 0.1
  const moved = editorReducer(initial, {
    type: "MOVE",
    id: "box-1",
    dx: 0.1,
    dy: 0.1,
  });

  assert.strictEqual(moved.past.length, 1);
  const box = moved.present[0];
  assert.strictEqual(box.box_normalized.xmin, 0.2);
  assert.strictEqual(box.box_normalized.ymin, 0.2);
  assert.strictEqual(box.box_normalized.xmax, 0.6);
  assert.strictEqual(box.box_normalized.ymax, 0.6);

  // Attempt to move past boundaries (e.g. +1.0)
  const clamped = editorReducer(initial, {
    type: "MOVE",
    id: "box-1",
    dx: 2.0,
    dy: 2.0,
  });
  const clampedBox = clamped.present[0];
  assert.strictEqual(clampedBox.box_normalized.xmax, 1);
  assert.strictEqual(clampedBox.box_normalized.ymax, 1);
  assert.strictEqual(clampedBox.box_normalized.xmin, 0.6); // 1 - width (0.4)
  assert.strictEqual(clampedBox.box_normalized.ymin, 0.6); // 1 - height (0.4)
});

test("RESIZE action updates bounding box and enforces minimum size", () => {
  const initial = createInitialEditorState([sampleBox1]);
  const resized = editorReducer(initial, {
    type: "RESIZE",
    id: "box-1",
    box_normalized: { ymin: 0.05, xmin: 0.05, ymax: 0.8, xmax: 0.8 },
  });

  assert.strictEqual(resized.past.length, 1);
  assert.strictEqual(resized.present[0].box_normalized.ymin, 0.05);
  assert.strictEqual(resized.present[0].box_normalized.xmax, 0.8);
});

test("RELABEL action updates class_id, class_name, and class colors", () => {
  const initial = createInitialEditorState([sampleBox1]);
  const relabeled = editorReducer(initial, {
    type: "RELABEL",
    id: "box-1",
    class_id: 3,
    class_name: "Chair",
  });

  assert.strictEqual(relabeled.past.length, 1);
  assert.strictEqual(relabeled.present[0].class_id, 3);
  assert.strictEqual(relabeled.present[0].class_name, "Chair");
  assert.strictEqual(relabeled.present[0].color_hex, "#56B4E9");
});

test("DELETE action removes box and updates selection if active", () => {
  const initial = createInitialEditorState([sampleBox1, sampleBox2]);
  const selected = editorReducer(initial, { type: "SELECT", id: "box-1" });
  assert.strictEqual(selected.past.length, 0); // SELECT doesn't pollute past

  const deleted = editorReducer(selected, { type: "DELETE", id: "box-1" });
  assert.strictEqual(deleted.present.length, 1);
  assert.strictEqual(deleted.present[0].id, "box-2");
  assert.strictEqual(deleted.selectedId, null);
  assert.strictEqual(deleted.past.length, 1);
});

test("UNDO and REDO cycle works accurately through multiple changes", () => {
  const initial = createInitialEditorState([sampleBox1]);

  // Step 1: Add box 2
  const s1 = editorReducer(initial, { type: "ADD", box: sampleBox2 });
  // Step 2: Relabel box 1 to Chair
  const s2 = editorReducer(s1, {
    type: "RELABEL",
    id: "box-1",
    class_id: 3,
    class_name: "Chair",
  });
  // Step 3: Move box 2
  const s3 = editorReducer(s2, {
    type: "MOVE",
    id: "box-2",
    dx: 0.05,
    dy: 0.05,
  });

  assert.strictEqual(s3.past.length, 3);
  assert.strictEqual(canUndo(s3), true);
  assert.strictEqual(canRedo(s3), false);

  // Undo step 3 (undo move)
  const u1 = editorReducer(s3, { type: "UNDO" });
  assert.strictEqual(u1.present.find((b) => b.id === "box-2")?.box_normalized.xmin, 0.6);
  assert.strictEqual(canRedo(u1), true);

  // Undo step 2 (undo relabel)
  const u2 = editorReducer(u1, { type: "UNDO" });
  assert.strictEqual(u2.present.find((b) => b.id === "box-1")?.class_name, "Person");

  // Undo step 1 (undo add)
  const u3 = editorReducer(u2, { type: "UNDO" });
  assert.strictEqual(u3.present.length, 1);
  assert.strictEqual(canUndo(u3), false);

  // Redo step 1
  const r1 = editorReducer(u3, { type: "REDO" });
  assert.strictEqual(r1.present.length, 2);

  // Redo step 2
  const r2 = editorReducer(r1, { type: "REDO" });
  assert.strictEqual(r2.present.find((b) => b.id === "box-1")?.class_name, "Chair");

  // New action after partial undo invalidates future stack
  const partialUndo = editorReducer(r2, { type: "UNDO" }); // back to s1
  assert.strictEqual(canRedo(partialUndo), true);
  const branched = editorReducer(partialUndo, { type: "DELETE", id: "box-2" });
  assert.strictEqual(canRedo(branched), false); // future is wiped
});

test("clampBBox handles inverted coords and bounds", () => {
  const inverted = clampBBox({ ymin: 0.8, xmin: 0.8, ymax: 0.2, xmax: 0.3 });
  assert.ok(inverted.ymin < inverted.ymax);
  assert.ok(inverted.xmin < inverted.xmax);
  assert.strictEqual(inverted.ymin, 0.2);
  assert.strictEqual(inverted.ymax, 0.8);
});

test("moveBox shifts and clamps within boundary limits", () => {
  const box = { ymin: 0.1, xmin: 0.2, ymax: 0.4, xmax: 0.5 };
  const shifted = moveBox(box, 0.1, 0.2);
  assert.strictEqual(shifted.xmin, 0.3);
  assert.strictEqual(shifted.ymin, 0.3);
  assert.strictEqual(shifted.xmax, 0.6);
  assert.strictEqual(shifted.ymax, 0.6);

  const overflow = moveBox(box, 0.8, 0.8);
  assert.strictEqual(overflow.xmax, 1);
  assert.strictEqual(overflow.ymax, 1);
  assert.strictEqual(overflow.xmin, 0.7);
  assert.strictEqual(overflow.ymin, 0.7);
});
