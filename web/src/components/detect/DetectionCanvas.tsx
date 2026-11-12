import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useImperativeHandle,
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
} from "lucide-react";
import type { DetectionItem } from "../../types";
import { colorForClass } from "../../lib/colors";
import { Tooltip } from "../ui/Tooltip";
import { cn } from "../../lib/utils";

export interface DetectionCanvasRef {
  exportAnnotatedPng: () => void;
  toggleLabels: () => void;
  toggleBoxes: () => void;
  resetView: () => void;
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

      // Draw Detections
      if (showBoxes) {
        detections.forEach((det, idx) => {
          // If comparison mode is active, only draw on the left side of split
          if (comparisonActive) {
            const splitXNorm = splitPos / 100;
            if (det.box_normalized.xmin > splitXNorm) return;
          }

          const isSelected = selectedIndex === idx;
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

          const lineWidth = (isActive ? 3 : 2) / scale;
          ctx.lineWidth = lineWidth;
          ctx.strokeStyle = strokeHex;

          if (isActive) {
            ctx.fillStyle = `rgba(${classCol.rgb[0]}, ${classCol.rgb[1]}, ${classCol.rgb[2]}, 0.22)`;
            ctx.fillRect(xmin, ymin, bw, bh);
          }

          ctx.strokeRect(xmin, ymin, bw, bh);

          // Labels
          if (showLabels) {
            const scorePercent = `${(det.confidence * 100).toFixed(1)}%`;
            const labelText = `${det.class_name} ${scorePercent}`;

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

      ctx.restore();

      // Draw Comparison Split Line & Badges
      if (comparisonActive) {
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

    // Mouse move: detect hover box
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

      for (let i = detections.length - 1; i >= 0; i--) {
        const b = detections[i].box_normalized;
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
      if (e.button === 1 || e.shiftKey || hoveredIndex === null) {
        setIsPanning(true);
        setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      }
    };

    const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (isPanning) {
        setIsPanning(false);
      } else if (e.button === 0) {
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
        aria-keyshortcuts="Space U B F Esc ArrowLeft ArrowRight"
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
            setIsPanning(false);
          }}
          onWheel={handleWheel}
          className={cn(
            "w-full h-full block touch-none",
            isPanning
              ? "cursor-grabbing"
              : hoveredIndex !== null
              ? "cursor-pointer"
              : zoom > 1
              ? "cursor-grab"
              : "cursor-default"
          )}
        />

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
        <div className="absolute bottom-3 left-3 z-10 flex items-center gap-1 bg-card/90 backdrop-blur-md border border-border-subtle rounded-control p-1 shadow-sm">
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
