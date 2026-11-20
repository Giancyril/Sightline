import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useImperativeHandle,
  useReducer,
  forwardRef,
} from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Tag,
  Eye,
  EyeOff,
  GitCompare,
  Pencil,
  Crosshair,
  Trash2,
} from "lucide-react";
import type { DetectionItem, NormalizedBBox } from "../../types.ts";
import { colorForClass } from "../../lib/colors.ts";
import {
  editorReducer,
  createInitialEditorState,
  moveBox,
  clampBBox,
  type EditorBox,
} from "../../lib/editorReducer.ts";
import { Tooltip } from "../ui/Tooltip";
import { cn } from "../../lib/utils";
import { ClassPicker } from "./ClassPicker.tsx";

export interface DetectionCanvasRef {
  exportAnnotatedPng: () => void;
  toggleLabels: () => void;
  toggleBoxes: () => void;
  resetView: () => void;
  toggleEditMode: () => void;
}

export type HandleType = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface HandlePos {
  type: HandleType;
  x: number;
  y: number;
  cursor: string;
}

function getBoxHandles(box: NormalizedBBox, w: number, h: number): HandlePos[] {
  const xmin = box.xmin * w;
  const xmax = box.xmax * w;
  const ymin = box.ymin * h;
  const ymax = box.ymax * h;
  const xmid = (xmin + xmax) / 2;
  const ymid = (ymin + ymax) / 2;

  return [
    { type: "nw", x: xmin, y: ymin, cursor: "nwse-resize" },
    { type: "n", x: xmid, y: ymin, cursor: "ns-resize" },
    { type: "ne", x: xmax, y: ymin, cursor: "nesw-resize" },
    { type: "e", x: xmax, y: ymid, cursor: "ew-resize" },
    { type: "se", x: xmax, y: ymax, cursor: "nwse-resize" },
    { type: "s", x: xmid, y: ymax, cursor: "ns-resize" },
    { type: "sw", x: xmin, y: ymax, cursor: "nesw-resize" },
    { type: "w", x: xmin, y: ymid, cursor: "ew-resize" },
  ];
}

function resizeBoxWithHandle(
  initial: NormalizedBBox,
  handle: HandleType,
  normX: number,
  normY: number,
  isShift: boolean
): NormalizedBBox {
  let { xmin, ymin, xmax, ymax } = initial;
  const initialW = initial.xmax - initial.xmin;
  const initialH = initial.ymax - initial.ymin;
  const ratio = initialW / Math.max(0.001, initialH);

  if (handle === "nw") {
    xmin = normX;
    ymin = normY;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      ymin = ymax - w / ratio;
    }
  } else if (handle === "ne") {
    xmax = normX;
    ymin = normY;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      ymin = ymax - w / ratio;
    }
  } else if (handle === "se") {
    xmax = normX;
    ymax = normY;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      ymax = ymin + w / ratio;
    }
  } else if (handle === "sw") {
    xmin = normX;
    ymax = normY;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      ymax = ymin + w / ratio;
    }
  } else if (handle === "n") {
    ymin = normY;
    if (isShift) {
      const h = Math.max(0.01, ymax - ymin);
      const newW = h * ratio;
      const xmid = (initial.xmin + initial.xmax) / 2;
      xmin = xmid - newW / 2;
      xmax = xmid + newW / 2;
    }
  } else if (handle === "s") {
    ymax = normY;
    if (isShift) {
      const h = Math.max(0.01, ymax - ymin);
      const newW = h * ratio;
      const xmid = (initial.xmin + initial.xmax) / 2;
      xmin = xmid - newW / 2;
      xmax = xmid + newW / 2;
    }
  } else if (handle === "w") {
    xmin = normX;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      const newH = w / ratio;
      const ymid = (initial.ymin + initial.ymax) / 2;
      ymin = ymid - newH / 2;
      ymax = ymid + newH / 2;
    }
  } else if (handle === "e") {
    xmax = normX;
    if (isShift) {
      const w = Math.max(0.01, xmax - xmin);
      const newH = w / ratio;
      const ymid = (initial.ymin + initial.ymax) / 2;
      ymin = ymid - newH / 2;
      ymax = ymid + newH / 2;
    }
  }

  return clampBBox({ ymin, xmin, ymax, xmax });
}

interface DetectionCanvasProps {
  imageSrc: string;
  detections: DetectionItem[];
  selectedIndex: number | null;
  onSelectDetection: (index: number | null) => void;
  hoveredIndex: number | null;
  onHoverDetection: (index: number | null) => void;
  className?: string;
  enableComparison?: boolean;
  isEditMode?: boolean;
  onToggleEditMode?: (editing: boolean) => void;
  onBoxesChange?: (boxes: EditorBox[]) => void;
}

export const DetectionCanvas = forwardRef<DetectionCanvasRef, DetectionCanvasProps>(
  (
    {
      imageSrc,
      detections,
      selectedIndex,
      onSelectDetection,
      hoveredIndex,
      onHoverDetection,
      className,
      enableComparison = false,
      isEditMode,
      onToggleEditMode,
      onBoxesChange,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);

    const [image, setImage] = useState<HTMLImageElement | null>(null);
    const [zoom, setZoom] = useState<number>(1);
    const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
    const [isPanning, setIsPanning] = useState<boolean>(false);
    const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
    const [showLabels, setShowLabels] = useState<boolean>(true);
    const [showBoxes, setShowBoxes] = useState<boolean>(true);
    const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

    // Edit mode state & reducer
    const [internalEditMode, setInternalEditMode] = useState<boolean>(false);
    const isEdit = isEditMode !== undefined ? isEditMode : internalEditMode;

    const [editorState, editorDispatch] = useReducer(
      editorReducer,
      createInitialEditorState()
    );

    const isDraggingBoxRef = useRef<boolean>(false);
    const resizingHandleRef = useRef<HandleType | null>(null);
    const [hoveredHandle, setHoveredHandle] = useState<HandleType | null>(null);
    const dragOriginRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const initialBoxRef = useRef<NormalizedBBox | null>(null);
    const draggingIdRef = useRef<string | null>(null);
    const [dragPreview, setDragPreview] = useState<{
      id: string;
      box_normalized: NormalizedBBox;
    } | null>(null);

    // Draw tool state
    const [isDrawTool, setIsDrawTool] = useState<boolean>(false);
    const [drawClassId, setDrawClassId] = useState<number>(1);
    const isDrawingBoxRef = useRef<boolean>(false);
    const drawStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
    const [drawingBoxPreview, setDrawingBoxPreview] = useState<NormalizedBBox | null>(null);

    const selectedBox =
      isEdit && editorState.selectedId
        ? editorState.present.find((b) => b.id === editorState.selectedId) || null
        : null;

    const toggleEditMode = useCallback(() => {
      const next = !isEdit;
      setInternalEditMode(next);
      onToggleEditMode?.(next);
    }, [isEdit, onToggleEditMode]);

    // Synchronize detections into editor state
    useEffect(() => {
      const boxes: EditorBox[] = detections.map((det, idx) => ({
        id: `det-${idx}`,
        class_id: det.class_id,
        class_name: det.class_name,
        confidence: det.confidence,
        box_normalized: { ...det.box_normalized },
        color_hex: det.color_hex,
        color_rgb: det.color_rgb,
        isManual: false,
      }));
      editorDispatch({ type: "INIT", boxes });
    }, [detections]);

    // Notify parent of boxes changes in edit mode
    useEffect(() => {
      if (isEdit) {
        onBoxesChange?.(editorState.present);
      }
    }, [isEdit, editorState.present, onBoxesChange]);

    // Sync selectedIndex from parent to editor state if external
    useEffect(() => {
      if (selectedIndex !== null && detections[selectedIndex]) {
        editorDispatch({ type: "SELECT", id: `det-${selectedIndex}` });
      } else if (selectedIndex === null && !isEdit) {
        editorDispatch({ type: "SELECT", id: null });
      }
    }, [selectedIndex, detections, isEdit]);

    // Model Comparison feature flag & split slider (0 to 100%)
    const [comparisonActive, setComparisonActive] = useState<boolean>(false);
    const splitPos = 50; // split at center

    // Load image
    useEffect(() => {
      let active = true;
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        if (active) {
          setImage(img);
          setZoom(1);
          setPan({ x: 0, y: 0 });
        }
      };
      img.src = imageSrc;
      return () => {
        active = false;
      };
    }, [imageSrc]);

    // Fullscreen toggle
    const toggleFullscreen = () => {
      if (!containerRef.current) return;
      if (!document.fullscreenElement) {
        void containerRef.current.requestFullscreen();
        setIsFullscreen(true);
      } else {
        void document.exitFullscreen();
        setIsFullscreen(false);
      }
    };

    useEffect(() => {
      const handleFsChange = () => {
        setIsFullscreen(Boolean(document.fullscreenElement));
      };
      document.addEventListener("fullscreenchange", handleFsChange);
      return () => document.removeEventListener("fullscreenchange", handleFsChange);
    }, []);

    // Compute coordinate transform
    const getTransform = useCallback(() => {
      const container = containerRef.current;
      if (!container || !image) {
        return { scale: 1, ox: 0, oy: 0, w: 1, h: 1 };
      }
      const cw = container.clientWidth;
      const ch = container.clientHeight;
      const iw = image.naturalWidth;
      const ih = image.naturalHeight;

      const baseScale = Math.min((cw - 32) / iw, (ch - 32) / ih, 1);
      const totalScale = baseScale * zoom;

      const ox = (cw - iw * totalScale) / 2 + pan.x;
      const oy = (ch - ih * totalScale) / 2 + pan.y;

      return { scale: totalScale, ox, oy, w: iw, h: ih };
    }, [image, zoom, pan]);

    // High-resolution Annotated PNG Export
    const exportAnnotatedPng = useCallback(() => {
      if (!image) return;
      const exportCanvas = document.createElement("canvas");
      const natW = image.naturalWidth;
      const natH = image.naturalHeight;
      exportCanvas.width = natW;
      exportCanvas.height = natH;

      const ctx = exportCanvas.getContext("2d");
      if (!ctx) return;

      // Draw original image
      ctx.drawImage(image, 0, 0, natW, natH);

      // Draw all bounding boxes and labels at full resolution
      detections.forEach((det) => {
        const xmin = det.box_normalized.xmin * natW;
        const ymin = det.box_normalized.ymin * natH;
        const bw = (det.box_normalized.xmax - det.box_normalized.xmin) * natW;
        const bh = (det.box_normalized.ymax - det.box_normalized.ymin) * natH;

        const classCol = colorForClass(det.class_id);

        ctx.lineWidth = 3;
        ctx.strokeStyle = classCol.hex;
        ctx.strokeRect(xmin, ymin, bw, bh);

        // Label
        const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
        const fontSize = Math.max(14, Math.round(natW * 0.018));
        ctx.font = `600 ${fontSize}px "Geist Mono", sans-serif`;

        const metrics = ctx.measureText(labelText);
        const padX = 8;
        const tagH = fontSize + 10;
        const tagW = metrics.width + padX * 2;
        const tagY = ymin >= tagH ? ymin - tagH : ymin;
        const tagX = Math.min(xmin, natW - tagW);

        ctx.fillStyle = classCol.hex;
        ctx.fillRect(tagX, tagY, tagW, tagH);

        ctx.fillStyle = "#ffffff";
        ctx.textBaseline = "middle";
        ctx.fillText(labelText, tagX + padX, tagY + tagH / 2);
      });

      const a = document.createElement("a");
      a.href = exportCanvas.toDataURL("image/png");
      a.download = `vision-ssd-annotated-${Date.now()}.png`;
      a.click();
    }, [image, detections]);

    useImperativeHandle(ref, () => ({
      exportAnnotatedPng,
      toggleLabels: () => setShowLabels((v) => !v),
      toggleBoxes: () => setShowBoxes((v) => !v),
      resetView: () => {
        setZoom(1);
        setPan({ x: 0, y: 0 });
      },
      toggleEditMode,
    }));

    // Render loop
    const render = useCallback(() => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container || !image) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const cw = container.clientWidth;
      const ch = container.clientHeight;

      if (canvas.width !== cw * dpr || canvas.height !== ch * dpr) {
        canvas.width = cw * dpr;
        canvas.height = ch * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, cw, ch);

      const { scale, ox, oy, w, h } = getTransform();

      // Draw Image
      ctx.save();
      ctx.translate(ox, oy);
      ctx.scale(scale, scale);
      ctx.drawImage(image, 0, 0, w, h);

      // Active boxes to render
      const boxesToRender: Array<{
        id: string;
        class_id: number;
        class_name: string;
        confidence?: number;
        box_normalized: NormalizedBBox;
        isManual?: boolean;
      }> = isEdit
        ? editorState.present.map((b) =>
            dragPreview && dragPreview.id === b.id
              ? { ...b, box_normalized: dragPreview.box_normalized }
              : b
          )
        : detections.map((d, i) => ({
            id: `det-${i}`,
            class_id: d.class_id,
            class_name: d.class_name,
            confidence: d.confidence,
            box_normalized: d.box_normalized,
            isManual: false,
          }));

      // Draw Detections
      if (showBoxes) {
        boxesToRender.forEach((det, idx) => {
          // If comparison mode is active, only draw on the left side of split
          if (comparisonActive && !isEdit) {
            const splitXNorm = splitPos / 100;
            if (det.box_normalized.xmin > splitXNorm) return;
          }

          const isSelected = isEdit
            ? editorState.selectedId === det.id
            : selectedIndex === idx;
          const isHovered = hoveredIndex === idx;
          const isActive = isSelected || isHovered;

          const xmin = det.box_normalized.xmin * w;
          const ymin = det.box_normalized.ymin * h;
          const xmax = det.box_normalized.xmax * w;
          const ymax = det.box_normalized.ymax * h;
          const bw = xmax - xmin;
          const bh = ymax - ymin;

          const classCol = colorForClass(det.class_id);
          const strokeHex = classCol.hex;

          const lineWidth = (isSelected ? 3.5 : isActive ? 3 : 2) / scale;
          ctx.lineWidth = lineWidth;
          ctx.strokeStyle = strokeHex;

          if (isActive) {
            ctx.fillStyle = `rgba(${classCol.rgb[0]}, ${classCol.rgb[1]}, ${classCol.rgb[2]}, ${
              isSelected ? 0.3 : 0.2
            })`;
            ctx.fillRect(xmin, ymin, bw, bh);
          }

          ctx.strokeRect(xmin, ymin, bw, bh);

          // Selection indicator dash outline
          if (isSelected) {
            ctx.save();
            ctx.lineWidth = 1.5 / scale;
            ctx.strokeStyle = "#ffffff";
            ctx.setLineDash([4 / scale, 4 / scale]);
            ctx.strokeRect(
              xmin + 1.5 / scale,
              ymin + 1.5 / scale,
              Math.max(1, bw - 3 / scale),
              Math.max(1, bh - 3 / scale)
            );
            ctx.restore();
          }

          // Eight Resize Handles
          if (isSelected && isEdit) {
            const handles = getBoxHandles(det.box_normalized, w, h);
            const handleSize = 7 / scale;
            ctx.save();
            handles.forEach((hPos) => {
              ctx.fillStyle = "#ffffff";
              ctx.strokeStyle = classCol.hex;
              ctx.lineWidth = 1.5 / scale;
              ctx.fillRect(
                hPos.x - handleSize / 2,
                hPos.y - handleSize / 2,
                handleSize,
                handleSize
              );
              ctx.strokeRect(
                hPos.x - handleSize / 2,
                hPos.y - handleSize / 2,
                handleSize,
                handleSize
              );
            });
            ctx.restore();
          }

          // Labels
          if (showLabels) {
            const scorePercent =
              det.confidence !== undefined
                ? `${(det.confidence * 100).toFixed(1)}%`
                : det.isManual
                ? "Manual"
                : "";
            const labelText = scorePercent
              ? `${det.class_name} ${scorePercent}`
              : det.class_name;

            const targetFontSize = 11;
            const fontPx = targetFontSize / scale;
            ctx.font = `500 ${fontPx}px "Geist Mono", monospace`;

            const padX = 5 / scale;
            const padY = 2.5 / scale;
            const metrics = ctx.measureText(labelText);
            const tagW = metrics.width + padX * 2;
            const tagH = fontPx + padY * 2;

            let tagY = ymin - tagH;
            if (tagY < 0) {
              tagY = ymin;
            }
            const tagX = Math.max(0, Math.min(xmin, w - tagW));

            ctx.fillStyle = strokeHex;
            ctx.fillRect(tagX, tagY, tagW, tagH);

            ctx.fillStyle = "#ffffff";
            ctx.textBaseline = "middle";
            ctx.fillText(labelText, tagX + padX, tagY + tagH / 2);
          }
        });
      }

      // Draw live drawing preview box
      if (drawingBoxPreview) {
        const cls = colorForClass(drawClassId);
        const pxMin = drawingBoxPreview.xmin * w;
        const pyMin = drawingBoxPreview.ymin * h;
        const pbw = (drawingBoxPreview.xmax - drawingBoxPreview.xmin) * w;
        const pbh = (drawingBoxPreview.ymax - drawingBoxPreview.ymin) * h;
        ctx.save();
        ctx.strokeStyle = cls.hex;
        ctx.lineWidth = 2.5 / scale;
        ctx.setLineDash([5 / scale, 5 / scale]);
        ctx.strokeRect(pxMin, pyMin, pbw, pbh);
        ctx.fillStyle = `rgba(${cls.rgb[0]}, ${cls.rgb[1]}, ${cls.rgb[2]}, 0.25)`;
        ctx.fillRect(pxMin, pyMin, pbw, pbh);
        ctx.restore();
      }

      ctx.restore();

      // Draw Comparison Split Line & Badges
      if (comparisonActive && !isEdit) {
        const splitX = (cw * splitPos) / 100;
        ctx.strokeStyle = "rgba(0, 112, 243, 0.8)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(splitX, 0);
        ctx.lineTo(splitX, ch);
        ctx.stroke();
      }

      ctx.restore();
    }, [
      image,
      detections,
      isEdit,
      editorState.present,
      editorState.selectedId,
      dragPreview,
      drawingBoxPreview,
      drawClassId,
      selectedIndex,
      hoveredIndex,
      showLabels,
      showBoxes,
      comparisonActive,
      splitPos,
      getTransform,
    ]);

    useEffect(() => {
      render();
    }, [render]);

    useEffect(() => {
      const handleResize = () => render();
      window.addEventListener("resize", handleResize);
      return () => window.removeEventListener("resize", handleResize);
    }, [render]);

    // Keyboard handlers for Esc, Delete/Backspace, D (draw), Arrow keys, and E (edit mode)
    useEffect(() => {
      const handleKeyDown = (e: KeyboardEvent) => {
        const target = e.target as HTMLElement | null;
        const isInputField =
          target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
        if (isInputField) return;

        if (e.key === "Escape") {
          if (isDrawingBoxRef.current) {
            isDrawingBoxRef.current = false;
            setDrawingBoxPreview(null);
          } else if (isDrawTool) {
            setIsDrawTool(false);
          } else if (resizingHandleRef.current) {
            resizingHandleRef.current = null;
            setDragPreview(null);
            initialBoxRef.current = null;
            draggingIdRef.current = null;
          } else if (isDraggingBoxRef.current) {
            isDraggingBoxRef.current = false;
            setDragPreview(null);
            initialBoxRef.current = null;
            draggingIdRef.current = null;
          } else if (editorState.selectedId) {
            editorDispatch({ type: "SELECT", id: null });
            onSelectDetection(null);
          }
        } else if (
          isEdit &&
          editorState.selectedId &&
          (e.key === "Delete" || e.key === "Backspace")
        ) {
          e.preventDefault();
          editorDispatch({ type: "DELETE", id: editorState.selectedId });
          onSelectDetection(null);
        } else if (isEdit && (e.key === "d" || e.key === "D")) {
          setIsDrawTool((prev) => !prev);
        } else if (
          isEdit &&
          editorState.selectedId &&
          ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(e.key)
        ) {
          e.preventDefault();
          const step = e.shiftKey ? 0.02 : 0.005;
          let dx = 0;
          let dy = 0;
          if (e.key === "ArrowLeft") dx = -step;
          if (e.key === "ArrowRight") dx = step;
          if (e.key === "ArrowUp") dy = -step;
          if (e.key === "ArrowDown") dy = step;

          editorDispatch({
            type: "MOVE",
            id: editorState.selectedId,
            dx,
            dy,
          });
        } else if (e.key === "e" || e.key === "E") {
          toggleEditMode();
        }
      };

      window.addEventListener("keydown", handleKeyDown);
      return () => window.removeEventListener("keydown", handleKeyDown);
    }, [
      editorState.selectedId,
      isEdit,
      isDrawTool,
      onSelectDetection,
      toggleEditMode,
    ]);

    // Mouse move: detect hover box & handle box dragging / handle resizing / draw box
    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isPanning) {
        const nextX = e.clientX - panStart.x;
        const nextY = e.clientY - panStart.y;
        const maxPan = 600;
        setPan({
          x: Math.max(-maxPan, Math.min(maxPan, nextX)),
          y: Math.max(-maxPan, Math.min(maxPan, nextY)),
        });
        return;
      }

      const container = containerRef.current;
      if (!container || !image || !showBoxes) return;

      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;

      const { scale, ox, oy, w, h } = getTransform();
      const imgX = (mx - ox) / scale;
      const imgY = (my - oy) / scale;

      const normX = imgX / w;
      const normY = imgY / h;

      // Handle active draw new box
      if (isEdit && isDrawTool && isDrawingBoxRef.current) {
        const xmin = Math.min(drawStartRef.current.x, normX);
        const xmax = Math.max(drawStartRef.current.x, normX);
        const ymin = Math.min(drawStartRef.current.y, normY);
        const ymax = Math.max(drawStartRef.current.y, normY);
        setDrawingBoxPreview(clampBBox({ ymin, xmin, ymax, xmax }));
        return;
      }

      // Handle active resize drag
      if (
        isEdit &&
        resizingHandleRef.current &&
        initialBoxRef.current &&
        draggingIdRef.current
      ) {
        const resized = resizeBoxWithHandle(
          initialBoxRef.current,
          resizingHandleRef.current,
          normX,
          normY,
          e.shiftKey
        );
        setDragPreview({ id: draggingIdRef.current, box_normalized: resized });
        return;
      }

      // Handle active box move drag
      if (
        isEdit &&
        isDraggingBoxRef.current &&
        initialBoxRef.current &&
        draggingIdRef.current
      ) {
        const dx = normX - dragOriginRef.current.x;
        const dy = normY - dragOriginRef.current.y;
        const moved = moveBox(initialBoxRef.current, dx, dy);
        setDragPreview({ id: draggingIdRef.current, box_normalized: moved });
        return;
      }

      // Check if mouse is hovering over any handle of selected box
      if (isEdit && editorState.selectedId) {
        const selected = editorState.present.find(
          (b) => b.id === editorState.selectedId
        );
        if (selected) {
          const handles = getBoxHandles(selected.box_normalized, w, h);
          const hitRadius = 8 / scale;
          const hit = handles.find(
            (hPos) => Math.hypot(imgX - hPos.x, imgY - hPos.y) <= hitRadius
          );
          if (hit) {
            setHoveredHandle(hit.type);
            onHoverDetection(null);
            return;
          }
        }
      }
      setHoveredHandle(null);

      const currentBoxes = isEdit ? editorState.present : detections;
      for (let i = currentBoxes.length - 1; i >= 0; i--) {
        const b = currentBoxes[i].box_normalized;
        if (
          normX >= b.xmin &&
          normX <= b.xmax &&
          normY >= b.ymin &&
          normY <= b.ymax
        ) {
          onHoverDetection(i);
          return;
        }
      }
      onHoverDetection(null);
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
      const container = containerRef.current;
      if (!container || !image) return;

      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const { scale, ox, oy, w, h } = getTransform();
      const imgX = (mx - ox) / scale;
      const imgY = (my - oy) / scale;
      const normX = imgX / w;
      const normY = imgY / h;

      // Draw tool click starts drawing
      if (isEdit && isDrawTool && e.button === 0 && !e.shiftKey) {
        isDrawingBoxRef.current = true;
        drawStartRef.current = { x: normX, y: normY };
        setDrawingBoxPreview({
          ymin: normY,
          xmin: normX,
          ymax: normY,
          xmax: normX,
        });
        return;
      }

      if (isEdit && e.button === 0 && !e.shiftKey) {
        // First check if clicked on any handle of the currently selected box
        if (editorState.selectedId) {
          const selected = editorState.present.find(
            (b) => b.id === editorState.selectedId
          );
          if (selected) {
            const handles = getBoxHandles(selected.box_normalized, w, h);
            const hitRadius = 8 / scale;
            const hit = handles.find(
              (hPos) => Math.hypot(imgX - hPos.x, imgY - hPos.y) <= hitRadius
            );
            if (hit) {
              resizingHandleRef.current = hit.type;
              initialBoxRef.current = { ...selected.box_normalized };
              draggingIdRef.current = selected.id;
              setDragPreview({
                id: selected.id,
                box_normalized: selected.box_normalized,
              });
              return;
            }
          }
        }

        // Otherwise find clicked box
        for (let i = editorState.present.length - 1; i >= 0; i--) {
          const b = editorState.present[i].box_normalized;
          if (
            normX >= b.xmin &&
            normX <= b.xmax &&
            normY >= b.ymin &&
            normY <= b.ymax
          ) {
            const target = editorState.present[i];
            editorDispatch({ type: "SELECT", id: target.id });
            onSelectDetection(i);
            isDraggingBoxRef.current = true;
            dragOriginRef.current = { x: normX, y: normY };
            initialBoxRef.current = { ...target.box_normalized };
            draggingIdRef.current = target.id;
            setDragPreview({ id: target.id, box_normalized: target.box_normalized });
            return;
          }
        }
        // Clicked outside boxes in edit mode -> deselect
        editorDispatch({ type: "SELECT", id: null });
        onSelectDetection(null);
        return;
      }

      if (e.button === 1 || e.shiftKey || hoveredIndex === null) {
        setIsPanning(true);
        setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      }
    };

    const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
      // Finalize draw tool
      if (isEdit && isDrawTool && isDrawingBoxRef.current) {
        isDrawingBoxRef.current = false;
        if (drawingBoxPreview) {
          const bw = drawingBoxPreview.xmax - drawingBoxPreview.xmin;
          const bh = drawingBoxPreview.ymax - drawingBoxPreview.ymin;
          if (bw >= 0.01 && bh >= 0.01) {
            const cls = colorForClass(drawClassId);
            const newBox: EditorBox = {
              id: `manual-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
              class_id: drawClassId,
              class_name: cls.name,
              confidence: 1.0,
              box_normalized: drawingBoxPreview,
              color_hex: cls.hex,
              color_rgb: cls.rgb,
              isManual: true,
            };
            editorDispatch({ type: "ADD", box: newBox });
            setIsDrawTool(false);
          }
        }
        setDrawingBoxPreview(null);
        return;
      }

      // Finalize resize drag
      if (resizingHandleRef.current) {
        resizingHandleRef.current = null;
        if (dragPreview && draggingIdRef.current) {
          editorDispatch({
            type: "RESIZE",
            id: draggingIdRef.current,
            box_normalized: dragPreview.box_normalized,
          });
        }
        setDragPreview(null);
        initialBoxRef.current = null;
        draggingIdRef.current = null;
        return;
      }

      // Finalize move drag
      if (isDraggingBoxRef.current) {
        isDraggingBoxRef.current = false;
        if (dragPreview && initialBoxRef.current && draggingIdRef.current) {
          const dx = dragPreview.box_normalized.xmin - initialBoxRef.current.xmin;
          const dy = dragPreview.box_normalized.ymin - initialBoxRef.current.ymin;
          if (Math.abs(dx) > 0.0001 || Math.abs(dy) > 0.0001) {
            editorDispatch({ type: "MOVE", id: draggingIdRef.current, dx, dy });
          }
        }
        setDragPreview(null);
        initialBoxRef.current = null;
        draggingIdRef.current = null;
        return;
      }

      if (isPanning) {
        setIsPanning(false);
      } else if (e.button === 0 && !isEdit) {
        onSelectDetection(hoveredIndex);
      }
    };

    const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      setZoom((z) => Math.min(Math.max(0.5, z * factor), 5));
    };

    const handleResetZoom = () => {
      setZoom(1);
      setPan({ x: 0, y: 0 });
    };

    return (
      <div
        ref={containerRef}
        role="region"
        aria-label="Object detection workspace canvas"
        aria-keyshortcuts="Space U B F Esc E D Delete Backspace ArrowLeft ArrowRight ArrowUp ArrowDown"
        className={cn(
          "relative w-full h-full min-h-[380px] bg-canvas border border-border-subtle rounded-panel overflow-hidden flex items-center justify-center select-none",
          isFullscreen && "fixed inset-0 z-50 rounded-none border-none",
          className
        )}
      >
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Object detection canvas: ${detections.length} objects detected`}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            onHoverDetection(null);
            setHoveredHandle(null);
            setIsPanning(false);
            if (isDrawingBoxRef.current) {
              isDrawingBoxRef.current = false;
              setDrawingBoxPreview(null);
            }
          }}
          onWheel={handleWheel}
          className={cn(
            "w-full h-full block touch-none",
            isDrawTool
              ? "cursor-crosshair"
              : resizingHandleRef.current || hoveredHandle
              ? hoveredHandle === "nw" || hoveredHandle === "se"
                ? "cursor-nwse-resize"
                : hoveredHandle === "ne" || hoveredHandle === "sw"
                ? "cursor-nesw-resize"
                : hoveredHandle === "n" || hoveredHandle === "s"
                ? "cursor-ns-resize"
                : "cursor-ew-resize"
              : isPanning || isDraggingBoxRef.current
              ? "cursor-grabbing"
              : isEdit &&
                hoveredIndex !== null &&
                editorState.selectedId === editorState.present[hoveredIndex]?.id
              ? "cursor-move"
              : hoveredIndex !== null
              ? "cursor-pointer"
              : zoom > 1
              ? "cursor-grab"
              : "cursor-default"
          )}
        />

        {/* Edit Mode Active Banner */}
        {isEdit && (
          <div
            className="absolute top-3 left-3 z-10 flex items-center gap-1.5 px-2.5 py-1 rounded-chip bg-amber-500/10 border border-amber-500/30 text-amber-400 text-[11px] font-mono font-medium backdrop-blur-md"
            aria-live="polite"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
            {isDrawTool
              ? "Drawing Tool Active: Click and drag on image to create box • Esc to cancel"
              : selectedBox
              ? `Selected: ${selectedBox.class_name} • Del to delete • Relabel via picker • Drag to move`
              : "Edit Mode: Click box to select/relabel/delete • D to draw new box • Arrow keys to nudge"}
          </div>
        )}

        {/* Comparison Labels Header Overlay */}
        {comparisonActive && (
          <div className="absolute top-3 inset-x-3 z-10 flex items-center justify-between pointer-events-none">
            <span className="px-2 py-0.5 rounded-chip bg-card/90 border border-border-subtle text-[11px] font-mono text-brand font-medium">
              Fine-Tuned (mAP 26.4%)
            </span>
            <span className="px-2 py-0.5 rounded-chip bg-card/90 border border-border-subtle text-[11px] font-mono text-text-tertiary font-medium">
              Baseline COCO (Simulated)
            </span>
          </div>
        )}

        {/* Floating Toolbar Controls */}
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1 bg-card/90 backdrop-blur-md border border-border-subtle rounded-control p-1 shadow-sm max-w-[calc(100%-24px)] overflow-x-auto">
          <Tooltip content={isEdit ? "Exit edit mode (E / Esc)" : "Edit annotations (E)"}>
            <button
              type="button"
              onClick={toggleEditMode}
              className={cn(
                "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand shrink-0",
                isEdit
                  ? "text-amber-400 bg-amber-500/20 ring-1 ring-amber-500/50"
                  : "text-text-tertiary hover:text-text-primary hover:bg-elevated"
              )}
              aria-label="Toggle edit mode"
              aria-pressed={isEdit}
            >
              <Pencil className="w-3.5 h-3.5" />
            </button>
          </Tooltip>

          {isEdit && (
            <>
              <Tooltip content={isDrawTool ? "Cancel drawing (D / Esc)" : "Draw new box (D)"}>
                <button
                  type="button"
                  onClick={() => setIsDrawTool((v) => !v)}
                  className={cn(
                    "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand shrink-0",
                    isDrawTool
                      ? "text-emerald-400 bg-emerald-500/20 ring-1 ring-emerald-500/50"
                      : "text-text-tertiary hover:text-text-primary hover:bg-elevated"
                  )}
                  aria-label="Draw new box"
                  aria-pressed={isDrawTool}
                >
                  <Crosshair className="w-3.5 h-3.5" />
                </button>
              </Tooltip>

              {/* Class picker for relabeling selected box or setting draw class */}
              {selectedBox ? (
                <div className="flex items-center gap-1 pl-0.5 shrink-0">
                  <span className="text-[10px] text-text-tertiary hidden sm:inline">Relabel:</span>
                  <ClassPicker
                    size="sm"
                    selectedClassId={selectedBox.class_id}
                    onSelectClass={(newId, newName) => {
                      editorDispatch({
                        type: "RELABEL",
                        id: selectedBox.id,
                        class_id: newId,
                        class_name: newName,
                      });
                    }}
                  />
                  <Tooltip content="Delete selected box (Del / Backspace)">
                    <button
                      type="button"
                      onClick={() => {
                        editorDispatch({ type: "DELETE", id: selectedBox.id });
                        onSelectDetection(null);
                      }}
                      className="p-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-red-400 shrink-0"
                      aria-label="Delete selected box"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </Tooltip>
                </div>
              ) : isDrawTool ? (
                <div className="flex items-center gap-1 pl-0.5 shrink-0">
                  <span className="text-[10px] text-text-tertiary hidden sm:inline">Class:</span>
                  <ClassPicker
                    size="sm"
                    selectedClassId={drawClassId}
                    onSelectClass={(newId) => setDrawClassId(newId)}
                  />
                </div>
              ) : null}
            </>
          )}

          <div className="h-3.5 w-px bg-border-subtle mx-0.5" />
          <Tooltip content="Zoom in (+)">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.min(z * 1.25, 5))}
              className="p-1.5 text-text-secondary hover:text-text-primary hover:bg-elevated rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Zoom in"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
          </Tooltip>

          <Tooltip content="Zoom out (-)">
            <button
              type="button"
              onClick={() => setZoom((z) => Math.max(z / 1.25, 0.5))}
              className="p-1.5 text-text-secondary hover:text-text-primary hover:bg-elevated rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Zoom out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
          </Tooltip>

          <span className="font-mono text-[10px] text-text-tertiary px-1 tabular-nums min-w-[38px] text-center">
            {Math.round(zoom * 100)}%
          </span>

          <Tooltip content="Reset view">
            <button
              type="button"
              onClick={handleResetZoom}
              className="p-1.5 text-text-secondary hover:text-text-primary hover:bg-elevated rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Reset zoom and position"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </Tooltip>

          <div className="h-3.5 w-px bg-border-subtle mx-0.5" />

          <Tooltip content={showLabels ? "Hide labels (Space)" : "Show labels (Space)"}>
            <button
              type="button"
              onClick={() => setShowLabels((v) => !v)}
              className={cn(
                "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                showLabels
                  ? "text-brand bg-brand-subtle"
                  : "text-text-tertiary hover:text-text-primary hover:bg-elevated"
              )}
              aria-label="Toggle labels"
            >
              <Tag className="w-3.5 h-3.5" />
            </button>
          </Tooltip>

          <Tooltip content={showBoxes ? "Hide bounding boxes (B)" : "Show bounding boxes (B)"}>
            <button
              type="button"
              onClick={() => setShowBoxes((v) => !v)}
              className={cn(
                "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                showBoxes
                  ? "text-brand bg-brand-subtle"
                  : "text-text-tertiary hover:text-text-primary hover:bg-elevated"
              )}
              aria-label="Toggle bounding boxes"
            >
              {showBoxes ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            </button>
          </Tooltip>

          {enableComparison && (
            <>
              <div className="h-3.5 w-px bg-border-subtle mx-0.5" />
              <Tooltip content={comparisonActive ? "Exit model comparison" : "Compare with baseline model"}>
                <button
                  type="button"
                  onClick={() => setComparisonActive((v) => !v)}
                  className={cn(
                    "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                    comparisonActive
                      ? "text-brand bg-brand-subtle"
                      : "text-text-tertiary hover:text-text-primary hover:bg-elevated"
                  )}
                  aria-label="Toggle model comparison"
                >
                  <GitCompare className="w-3.5 h-3.5" />
                </button>
              </Tooltip>
            </>
          )}

          <div className="h-3.5 w-px bg-border-subtle mx-0.5" />

          <Tooltip content={isFullscreen ? "Exit fullscreen (F)" : "Fullscreen (F)"}>
            <button
              type="button"
              onClick={toggleFullscreen}
              className="p-1.5 text-text-secondary hover:text-text-primary hover:bg-elevated rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label="Toggle fullscreen"
            >
              {isFullscreen ? (
                <Minimize2 className="w-3.5 h-3.5" />
              ) : (
                <Maximize2 className="w-3.5 h-3.5" />
              )}
            </button>
          </Tooltip>
        </div>
      </div>
    );
  }
);

DetectionCanvas.displayName = "DetectionCanvas";
