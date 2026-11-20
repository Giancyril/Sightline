/**
 * web/src/lib/editorReducer.ts
 * ----------------------------
 * Pure reducer and state machine for the annotation editor.
 * Handles add, move, resize, relabel, delete with undo/redo history stacks.
 */

import type { NormalizedBBox } from "../types.ts";
import { colorForClass } from "./colors.ts";

export interface EditorBox {
  id: string;
  class_id: number;
  class_name: string;
  confidence?: number;
  box_normalized: NormalizedBBox;
  color_hex?: string;
  color_rgb?: [number, number, number];
  isManual?: boolean;
}

export interface EditorState {
  present: EditorBox[];
  past: EditorBox[][];
  future: EditorBox[][];
  selectedId: string | null;
  historyLimit: number;
}

export type EditorAction =
  | { type: "INIT"; boxes: EditorBox[] }
  | { type: "ADD"; box: EditorBox }
  | { type: "MOVE"; id: string; dx: number; dy: number }
  | { type: "RESIZE"; id: string; box_normalized: NormalizedBBox }
  | {
      type: "RELABEL";
      id: string;
      class_id: number;
      class_name: string;
      color_hex?: string;
      color_rgb?: [number, number, number];
    }
  | { type: "DELETE"; id: string }
  | { type: "SELECT"; id: string | null }
  | { type: "UNDO" }
  | { type: "REDO" };

const DEFAULT_HISTORY_LIMIT = 50;
const MIN_BOX_SIZE = 0.005;

/**
 * Clamps coordinate between [min, max].
 */
export function clamp(value: number, min: number = 0, max: number = 1): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Clamps a normalized bounding box strictly within [0, 1] bounds,
 * maintaining valid min < max ordering and minimum dimension.
 */
export function clampBBox(box: NormalizedBBox): NormalizedBBox {
  let ymin = clamp(box.ymin, 0, 1);
  let xmin = clamp(box.xmin, 0, 1);
  let ymax = clamp(box.ymax, 0, 1);
  let xmax = clamp(box.xmax, 0, 1);

  if (ymin > ymax) {
    const tmp = ymin;
    ymin = ymax;
    ymax = tmp;
  }
  if (xmin > xmax) {
    const tmp = xmin;
    xmin = xmax;
    xmax = tmp;
  }

  if (ymax - ymin < MIN_BOX_SIZE) {
    if (ymax + MIN_BOX_SIZE <= 1) {
      ymax = ymin + MIN_BOX_SIZE;
    } else {
      ymin = Math.max(0, ymax - MIN_BOX_SIZE);
    }
  }

  if (xmax - xmin < MIN_BOX_SIZE) {
    if (xmax + MIN_BOX_SIZE <= 1) {
      xmax = xmin + MIN_BOX_SIZE;
    } else {
      xmin = Math.max(0, xmax - MIN_BOX_SIZE);
    }
  }

  return {
    ymin: Number(ymin.toFixed(5)),
    xmin: Number(xmin.toFixed(5)),
    ymax: Number(ymax.toFixed(5)),
    xmax: Number(xmax.toFixed(5)),
  };
}

/**
 * Moves a normalized bounding box by dx and dy while keeping its dimensions
 * intact and clamping entirely inside [0, 1].
 */
export function moveBox(box: NormalizedBBox, dx: number, dy: number): NormalizedBBox {
  const width = box.xmax - box.xmin;
  const height = box.ymax - box.ymin;

  let xmin = box.xmin + dx;
  let ymin = box.ymin + dy;

  // Clamp within image bounds
  if (xmin < 0) xmin = 0;
  if (ymin < 0) ymin = 0;
  if (xmin + width > 1) xmin = Math.max(0, 1 - width);
  if (ymin + height > 1) ymin = Math.max(0, 1 - height);

  const xmax = Math.min(1, xmin + width);
  const ymax = Math.min(1, ymin + height);

  return {
    ymin: Number(ymin.toFixed(5)),
    xmin: Number(xmin.toFixed(5)),
    ymax: Number(ymax.toFixed(5)),
    xmax: Number(xmax.toFixed(5)),
  };
}

/**
 * Creates initial editor state.
 */
export function createInitialEditorState(
  boxes: EditorBox[] = [],
  historyLimit = DEFAULT_HISTORY_LIMIT
): EditorState {
  return {
    present: boxes.map((b) => ({
      ...b,
      box_normalized: clampBBox(b.box_normalized),
    })),
    past: [],
    future: [],
    selectedId: null,
    historyLimit,
  };
}

/**
 * Checks whether undo is available.
 */
export function canUndo(state: EditorState): boolean {
  return state.past.length > 0;
}

/**
 * Checks whether redo is available.
 */
export function canRedo(state: EditorState): boolean {
  return state.future.length > 0;
}

/**
 * Pure reducer function for editor actions.
 */
export function editorReducer(state: EditorState, action: EditorAction): EditorState {
  switch (action.type) {
    case "INIT": {
      return createInitialEditorState(action.boxes, state.historyLimit);
    }

    case "SELECT": {
      if (state.selectedId === action.id) return state;
      return {
        ...state,
        selectedId: action.id,
      };
    }

    case "ADD": {
      const clampedBox: EditorBox = {
        ...action.box,
        box_normalized: clampBBox(action.box.box_normalized),
        color_hex: action.box.color_hex ?? colorForClass(action.box.class_id).hex,
        color_rgb: action.box.color_rgb ?? colorForClass(action.box.class_id).rgb,
      };

      const newPast = [...state.past, state.present].slice(-state.historyLimit);
      return {
        ...state,
        past: newPast,
        present: [...state.present, clampedBox],
        future: [],
        selectedId: clampedBox.id,
      };
    }

    case "MOVE": {
      const targetIndex = state.present.findIndex((b) => b.id === action.id);
      if (targetIndex === -1) return state;

      const current = state.present[targetIndex];
      const moved = moveBox(current.box_normalized, action.dx, action.dy);

      // If nothing changed, return state unchanged
      if (
        moved.ymin === current.box_normalized.ymin &&
        moved.xmin === current.box_normalized.xmin &&
        moved.ymax === current.box_normalized.ymax &&
        moved.xmax === current.box_normalized.xmax
      ) {
        return state;
      }

      const nextBoxes = [...state.present];
      nextBoxes[targetIndex] = {
        ...current,
        box_normalized: moved,
      };

      const newPast = [...state.past, state.present].slice(-state.historyLimit);
      return {
        ...state,
        past: newPast,
        present: nextBoxes,
        future: [],
      };
    }

    case "RESIZE": {
      const targetIndex = state.present.findIndex((b) => b.id === action.id);
      if (targetIndex === -1) return state;

      const current = state.present[targetIndex];
      const resized = clampBBox(action.box_normalized);

      if (
        resized.ymin === current.box_normalized.ymin &&
        resized.xmin === current.box_normalized.xmin &&
        resized.ymax === current.box_normalized.ymax &&
        resized.xmax === current.box_normalized.xmax
      ) {
        return state;
      }

      const nextBoxes = [...state.present];
      nextBoxes[targetIndex] = {
        ...current,
        box_normalized: resized,
      };

      const newPast = [...state.past, state.present].slice(-state.historyLimit);
      return {
        ...state,
        past: newPast,
        present: nextBoxes,
        future: [],
      };
    }

    case "RELABEL": {
      const targetIndex = state.present.findIndex((b) => b.id === action.id);
      if (targetIndex === -1) return state;

      const current = state.present[targetIndex];
      const colorSpec = colorForClass(action.class_id);

      const updated: EditorBox = {
        ...current,
        class_id: action.class_id,
        class_name: action.class_name,
        color_hex: action.color_hex ?? colorSpec.hex,
        color_rgb: action.color_rgb ?? colorSpec.rgb,
      };

      const nextBoxes = [...state.present];
      nextBoxes[targetIndex] = updated;

      const newPast = [...state.past, state.present].slice(-state.historyLimit);
      return {
        ...state,
        past: newPast,
        present: nextBoxes,
        future: [],
      };
    }

    case "DELETE": {
      const targetIndex = state.present.findIndex((b) => b.id === action.id);
      if (targetIndex === -1) return state;

      const nextBoxes = state.present.filter((b) => b.id !== action.id);
      const newPast = [...state.past, state.present].slice(-state.historyLimit);

      return {
        ...state,
        past: newPast,
        present: nextBoxes,
        future: [],
        selectedId: state.selectedId === action.id ? null : state.selectedId,
      };
    }

    case "UNDO": {
      if (state.past.length === 0) return state;

      const previous = state.past[state.past.length - 1];
      const newPast = state.past.slice(0, state.past.length - 1);
      const newFuture = [state.present, ...state.future].slice(0, state.historyLimit);

      // Keep selectedId if still in previous, otherwise deselect
      const nextSelectedId =
        state.selectedId && previous.some((b) => b.id === state.selectedId)
          ? state.selectedId
          : null;

      return {
        ...state,
        past: newPast,
        present: previous,
        future: newFuture,
        selectedId: nextSelectedId,
      };
    }

    case "REDO": {
      if (state.future.length === 0) return state;

      const next = state.future[0];
      const newFuture = state.future.slice(1);
      const newPast = [...state.past, state.present].slice(-state.historyLimit);

      const nextSelectedId =
        state.selectedId && next.some((b) => b.id === state.selectedId)
          ? state.selectedId
          : null;

      return {
        ...state,
        past: newPast,
        present: next,
        future: newFuture,
        selectedId: nextSelectedId,
      };
    }

    default:
      return state;
  }
}
