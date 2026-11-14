import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Upload,
  SplitSquareVertical,
  Columns2,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sparkles,
  RefreshCw,
  Cpu,
  Layers,
  Clock,
  Box,
} from "lucide-react";
import { compareModels, fetchModels } from "../api";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { Skeleton } from "../components/ui/Skeleton";
import { Slider } from "../components/ui/Slider";
import { Tooltip } from "../components/ui/Tooltip";
import { colorForClass } from "../lib/colors";
import { cn } from "../lib/utils";
import type { CompareResponse, DetectionItem, ModelRegistryItem } from "../types";

export type CompareDisplayMode = "slider" | "side-by-side";

export const CompareView: React.FC = () => {
  // Model selection state
  const [modelA, setModelA] = useState<string>("fine-tuned");
  const [modelB, setModelB] = useState<string>("coco-baseline");
  const [registeredModels, setRegisteredModels] = useState<ModelRegistryItem[]>([]);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);

  // Threshold states
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [nmsThreshold, setNmsThreshold] = useState<number>(0.45);

  // Image & Execution state
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [imageBlob, setImageBlob] = useState<Blob | null>(null);
  const [imageElement, setImageElement] = useState<HTMLImageElement | null>(null);
  const [isComparing, setIsComparing] = useState<boolean>(false);
  const [compareResult, setCompareResult] = useState<CompareResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // View presentation state
  const [displayMode, setDisplayMode] = useState<CompareDisplayMode>("slider");
  const [sliderPos, setSliderPos] = useState<number>(50); // percentage 0 to 100
  const [isDraggingSlider, setIsDraggingSlider] = useState<boolean>(false);

  // Synchronized Zoom & Pan state
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Refs for rendering
  const sliderContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const canvasRefA = useRef<HTMLCanvasElement>(null);
  const canvasRefB = useRef<HTMLCanvasElement>(null);
  const sliderCanvasRef = useRef<HTMLCanvasElement>(null);

  // Fetch available models
  useEffect(() => {
    let active = true;
    fetchModels()
      .then((res) => {
        if (active) {
          setRegisteredModels(res.models);
          setLoadingModels(false);
        }
      })
      .catch(() => {
        if (active) {
          setLoadingModels(false);
        }
      });
    return () => {
      active = false;
    };
  }, []);

  // Filter detections client-side for threshold slider interactivity
  const filteredDetectionsA = useMemo<DetectionItem[]>(() => {
    if (!compareResult) return [];
    return compareResult.model_a_result.detections.filter(
      (d) => d.confidence >= confThreshold
    );
  }, [compareResult, confThreshold]);

  const filteredDetectionsB = useMemo<DetectionItem[]>(() => {
    if (!compareResult) return [];
    return compareResult.model_b_result.detections.filter(
      (d) => d.confidence >= confThreshold
    );
  }, [compareResult, confThreshold]);

  // Execute comparison
  const runComparison = useCallback(
    async (blob: Blob, modA = modelA, modB = modelB, nms = nmsThreshold) => {
      try {
        setIsComparing(true);
        setErrorMsg(null);
        const res = await compareModels(blob, modA, modB, 0.05, nms);
        setCompareResult(res);
      } catch (err) {
        setErrorMsg((err as Error).message || "Failed to run comparison");
      } finally {
        setIsComparing(false);
      }
    },
    [modelA, modelB, nmsThreshold]
  );

  // Handle image upload
  const handleFileChange = (file: File | Blob) => {
    setImageBlob(file);
    const url = URL.createObjectURL(file);
    setImageSrc(url);

    const img = new Image();
    img.src = url;
    img.onload = () => {
      setImageElement(img);
      setZoom(1);
      setPan({ x: 0, y: 0 });
      void runComparison(file);
    };
  };

  // Synchronized canvas drawer
  const drawDetections = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      cw: number,
      ch: number,
      dets: DetectionItem[],
      clipRect?: { left: number; right: number }
    ) => {
      if (!imageElement) return;

      ctx.clearRect(0, 0, cw, ch);

      // Save state for clipping
      ctx.save();
      if (clipRect) {
        ctx.beginPath();
        ctx.rect(clipRect.left, 0, clipRect.right - clipRect.left, ch);
        ctx.clip();
      }

      // Compute coordinate transform with zoom & pan
      const baseScale = Math.min(cw / imageElement.width, ch / imageElement.height);
      const totalScale = baseScale * zoom;
      const ox = (cw - imageElement.width * totalScale) / 2 + pan.x;
      const oy = (ch - imageElement.height * totalScale) / 2 + pan.y;

      // Draw background image
      ctx.drawImage(
        imageElement,
        ox,
        oy,
        imageElement.width * totalScale,
        imageElement.height * totalScale
      );

      // Draw bounding boxes
      for (const det of dets) {
        const col = colorForClass(det.class_id);
        const x = ox + det.box_normalized.xmin * imageElement.width * totalScale;
        const y = oy + det.box_normalized.ymin * imageElement.height * totalScale;
        const w = (det.box_normalized.xmax - det.box_normalized.xmin) * imageElement.width * totalScale;
        const h = (det.box_normalized.ymax - det.box_normalized.ymin) * imageElement.height * totalScale;

        // Box stroke
        ctx.strokeStyle = col.hex;
        ctx.lineWidth = Math.max(1.5, 2 * (totalScale / baseScale));
        ctx.strokeRect(x, y, w, h);

        // Label tag
        const score = Math.round(det.confidence * 100);
        const labelText = `${det.class_name} ${score}%`;
        ctx.font = "bold 11px monospace";
        const textMetrics = ctx.measureText(labelText);
        const tagH = 16;
        const tagW = textMetrics.width + 8;
        const tagY = Math.max(0, y - tagH);

        ctx.fillStyle = col.hex;
        ctx.fillRect(x, tagY, tagW, tagH);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, x + 4, tagY + 12);
      }

      ctx.restore();
    },
    [imageElement, zoom, pan]
  );

  // Render loop when in Side-by-side mode
  useEffect(() => {
    if (displayMode !== "side-by-side") return;

    if (canvasRefA.current && imageElement) {
      const c = canvasRefA.current;
      c.width = c.clientWidth * window.devicePixelRatio;
      c.height = c.clientHeight * window.devicePixelRatio;
      const ctx = c.getContext("2d");
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        drawDetections(ctx, c.clientWidth, c.clientHeight, filteredDetectionsA);
      }
    }

    if (canvasRefB.current && imageElement) {
      const c = canvasRefB.current;
      c.width = c.clientWidth * window.devicePixelRatio;
      c.height = c.clientHeight * window.devicePixelRatio;
      const ctx = c.getContext("2d");
      if (ctx) {
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        drawDetections(ctx, c.clientWidth, c.clientHeight, filteredDetectionsB);
      }
    }
  }, [displayMode, drawDetections, filteredDetectionsA, filteredDetectionsB, imageElement]);

  // Render loop when in Slider mode
  useEffect(() => {
    if (displayMode !== "slider" || !sliderCanvasRef.current || !imageElement) return;

    const c = sliderCanvasRef.current;
    c.width = c.clientWidth * window.devicePixelRatio;
    c.height = c.clientHeight * window.devicePixelRatio;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const cw = c.clientWidth;
    const ch = c.clientHeight;

    const splitPx = (cw * sliderPos) / 100;

    // Draw Model A on left half
    drawDetections(ctx, cw, ch, filteredDetectionsA, { left: 0, right: splitPx });

    // Draw Model B on right half
    drawDetections(ctx, cw, ch, filteredDetectionsB, { left: splitPx, right: cw });

    // Draw vertical divider
    ctx.strokeStyle = "#0070f3";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(splitPx, 0);
    ctx.lineTo(splitPx, ch);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [displayMode, sliderPos, drawDetections, filteredDetectionsA, filteredDetectionsB, imageElement]);

  // Synchronized Pan interaction handlers
  const handlePointerDown = (e: React.PointerEvent) => {
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (!isPanning) return;
    setPan({
      x: e.clientX - panStart.x,
      y: e.clientY - panStart.y,
    });
  };

  const handlePointerUp = () => {
    setIsPanning(false);
  };

  // Synchronized Zoom on mouse wheel
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((z) => Math.min(Math.max(0.5, z * factor), 5.0));
  };

  // Slider Dragging handler
  const handleSliderPointerDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    setIsDraggingSlider(true);
  };

  useEffect(() => {
    if (!isDraggingSlider) return;

    const onPointerMove = (e: PointerEvent) => {
      if (!sliderContainerRef.current) return;
      const rect = sliderContainerRef.current.getBoundingClientRect();
      const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
      setSliderPos(pct);
    };

    const onPointerUp = () => {
      setIsDraggingSlider(false);
    };

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
    };
  }, [isDraggingSlider]);

  return (
    <div className="flex flex-col flex-1 h-full min-h-[500px] gap-4">
      {/* Top Controls Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-card border border-border-subtle rounded-panel">
        {/* Model Selection Cluster */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <Cpu className="w-4 h-4 text-brand" />
            <span className="font-semibold text-text-primary">Model A:</span>
            <select
              value={modelA}
              aria-label="Model A for comparison"
              disabled={isComparing || loadingModels}
              onChange={(e) => {
                setModelA(e.target.value);
                if (imageBlob) void runComparison(imageBlob, e.target.value, modelB);
              }}
              className="h-7 px-2 text-xs bg-elevated text-text-primary border border-border-subtle rounded-control cursor-pointer focus:outline-none focus:ring-1 focus:ring-brand"
            >
              {registeredModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name} ({m.version})
                </option>
              ))}
            </select>
          </div>

          <span className="text-text-tertiary text-xs">vs</span>

          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <span className="font-semibold text-text-primary">Model B:</span>
            <select
              value={modelB}
              aria-label="Model B for comparison"
              disabled={isComparing || loadingModels}
              onChange={(e) => {
                setModelB(e.target.value);
                if (imageBlob) void runComparison(imageBlob, modelA, e.target.value);
              }}
              className="h-7 px-2 text-xs bg-elevated text-text-primary border border-border-subtle rounded-control cursor-pointer focus:outline-none focus:ring-1 focus:ring-brand"
            >
              {registeredModels.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name} ({m.version})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* View Mode & Zoom Controls */}
        <div className="flex items-center gap-2">
          {/* Display Mode Toggle */}
          <div className="flex items-center rounded-control border border-border-subtle bg-elevated p-0.5">
            <button
              type="button"
              onClick={() => setDisplayMode("slider")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand",
                displayMode === "slider"
                  ? "bg-card text-text-primary shadow-sm"
                  : "text-text-tertiary hover:text-text-primary"
              )}
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              <span>Slider</span>
            </button>
            <button
              type="button"
              onClick={() => setDisplayMode("side-by-side")}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brand",
                displayMode === "side-by-side"
                  ? "bg-card text-text-primary shadow-sm"
                  : "text-text-tertiary hover:text-text-primary"
              )}
            >
              <Columns2 className="w-3.5 h-3.5" />
              <span>Side-by-side</span>
            </button>
          </div>

          <div className="h-4 w-px bg-border-subtle mx-1" />

          {/* Synchronized Zoom Controls */}
          <div className="flex items-center gap-1 bg-elevated border border-border-subtle rounded-control p-0.5">
            <Tooltip content="Zoom in">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.min(z * 1.25, 5.0))}
                className="p-1 text-text-secondary hover:text-text-primary rounded-[4px] transition-colors"
                aria-label="Zoom in"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>
            </Tooltip>
            <Tooltip content="Zoom out">
              <button
                type="button"
                onClick={() => setZoom((z) => Math.max(z / 1.25, 0.5))}
                className="p-1 text-text-secondary hover:text-text-primary rounded-[4px] transition-colors"
                aria-label="Zoom out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>
            </Tooltip>
            <span className="font-mono text-[10px] text-text-tertiary px-1 tabular-nums min-w-[32px] text-center">
              {Math.round(zoom * 100)}%
            </span>
            <Tooltip content="Reset view">
              <button
                type="button"
                onClick={() => {
                  setZoom(1);
                  setPan({ x: 0, y: 0 });
                }}
                className="p-1 text-text-secondary hover:text-text-primary rounded-[4px] transition-colors"
                aria-label="Reset zoom and position"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </Tooltip>
          </div>

          {/* Upload Button */}
          <input
            type="file"
            ref={fileInputRef}
            className="hidden"
            accept="image/jpeg,image/png,image/webp"
            onChange={(e) => {
              if (e.target.files?.[0]) handleFileChange(e.target.files[0]);
            }}
          />
          <Button
            variant="secondary"
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            leftIcon={<Upload className="w-3.5 h-3.5" />}
          >
            {imageSrc ? "Change Image" : "Upload Image"}
          </Button>
        </div>
      </div>

      {/* Main Comparison Canvas Area */}
      {!imageSrc ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 border border-dashed border-border-subtle rounded-panel bg-card text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-control bg-elevated border border-border-subtle text-brand mb-3">
            <SplitSquareVertical className="w-6 h-6" />
          </div>
          <h3 className="text-sm font-semibold text-text-primary mb-1">
            Compare Models Side-by-Side
          </h3>
          <p className="text-xs text-text-tertiary max-w-sm mb-4 leading-relaxed">
            Upload an image to compare detections, confidence scores, and inference latency across two models simultaneously.
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="primary"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              leftIcon={<Upload className="w-3.5 h-3.5" />}
            >
              Select Image
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={async () => {
                const res = await fetch("/sample.jpg");
                const b = await res.blob();
                handleFileChange(b);
              }}
              leftIcon={<Sparkles className="w-3.5 h-3.5" />}
            >
              Use Sample
            </Button>
          </div>
        </div>
      ) : isComparing ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 bg-card border border-border-subtle rounded-panel">
          <Skeleton className="w-full h-80 max-w-2xl rounded-panel" />
          <p className="text-xs text-text-tertiary mt-4 flex items-center gap-2">
            <RefreshCw className="w-3.5 h-3.5 animate-spin text-brand" />
            Running parallel inference across both models…
          </p>
        </div>
      ) : errorMsg ? (
        <div className="flex-1 flex flex-col items-center justify-center p-8 bg-card border border-border-subtle rounded-panel text-center">
          <p className="text-sm font-semibold text-status-danger mb-2">Comparison Failed</p>
          <p className="text-xs text-text-tertiary mb-4">{errorMsg}</p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => imageBlob && void runComparison(imageBlob)}
          >
            Retry Comparison
          </Button>
        </div>
      ) : (
        <div className="flex-1 flex flex-col min-h-0 gap-3">
          {/* Canvas Viewport */}
          <div
            className="flex-1 relative overflow-hidden bg-canvas border border-border-subtle rounded-panel select-none touch-none"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onWheel={handleWheel}
          >
            {displayMode === "slider" ? (
              /* Draggable Slider Mode */
              <div ref={sliderContainerRef} className="relative w-full h-full">
                <canvas ref={sliderCanvasRef} className="w-full h-full block" />

                {/* Left/Right Identification Badges */}
                <div className="absolute top-3 left-3 pointer-events-none">
                  <Badge variant="brand" mono>
                    Model A: {modelA} ({filteredDetectionsA.length} boxes)
                  </Badge>
                </div>
                <div className="absolute top-3 right-3 pointer-events-none">
                  <Badge variant="neutral" mono>
                    Model B: {modelB} ({filteredDetectionsB.length} boxes)
                  </Badge>
                </div>

                {/* Draggable Divider Line & Handle */}
                <div
                  className="absolute top-0 bottom-0 w-8 -ml-4 flex items-center justify-center cursor-ew-resize z-10"
                  style={{ left: `${sliderPos}%` }}
                  onPointerDown={handleSliderPointerDown}
                >
                  <div className="w-0.5 h-full bg-brand" />
                  <div className="absolute flex items-center justify-center w-6 h-6 rounded-full bg-card border border-brand text-brand shadow-md">
                    <SplitSquareVertical className="w-3 h-3" />
                  </div>
                </div>
              </div>
            ) : (
              /* Side-by-Side Mode */
              <div className="grid grid-cols-2 w-full h-full divide-x divide-border-subtle">
                <div className="relative w-full h-full">
                  <canvas ref={canvasRefA} className="w-full h-full block" />
                  <div className="absolute top-3 left-3 pointer-events-none">
                    <Badge variant="brand" mono>
                      Model A: {modelA} ({filteredDetectionsA.length} boxes)
                    </Badge>
                  </div>
                </div>
                <div className="relative w-full h-full">
                  <canvas ref={canvasRefB} className="w-full h-full block" />
                  <div className="absolute top-3 left-3 pointer-events-none">
                    <Badge variant="neutral" mono>
                      Model B: {modelB} ({filteredDetectionsB.length} boxes)
                    </Badge>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Interactive Threshold & Metadata Strip */}
          <div className="p-3 bg-card border border-border-subtle rounded-panel flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-6 flex-1 min-w-[280px]">
              <div className="flex-1 max-w-xs">
                <Slider
                  label="Confidence threshold"
                  value={confThreshold}
                  min={0.05}
                  max={0.95}
                  step={0.05}
                  onValueChange={setConfThreshold}
                />
              </div>
              <div className="flex-1 max-w-xs">
                <Slider
                  label="NMS IoU threshold"
                  value={nmsThreshold}
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  onValueChange={(val) => {
                    setNmsThreshold(val);
                    if (imageBlob) void runComparison(imageBlob, modelA, modelB, val);
                  }}
                />
              </div>
            </div>

            {compareResult && (
              <div className="flex items-center gap-4 text-xs font-mono text-text-tertiary">
                <span className="flex items-center gap-1.5">
                  <Box className="w-3.5 h-3.5 text-brand" />
                  Model A: {filteredDetectionsA.length} objs ({compareResult.model_a_result.inference_time_ms.toFixed(1)}ms)
                </span>
                <span>·</span>
                <span className="flex items-center gap-1.5">
                  <Layers className="w-3.5 h-3.5 text-text-secondary" />
                  Model B: {filteredDetectionsB.length} objs ({compareResult.model_b_result.inference_time_ms.toFixed(1)}ms)
                </span>
                <span>·</span>
                <span className="flex items-center gap-1.5">
                  <Clock className="w-3.5 h-3.5" />
                  Parallel: {compareResult.total_time_ms.toFixed(1)}ms
                </span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
