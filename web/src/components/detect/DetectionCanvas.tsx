import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Tag,
  Eye,
  EyeOff,
} from "lucide-react";
import type { DetectionItem } from "../../types";
import { colorForClass } from "../../lib/colors";
import { Tooltip } from "../ui/Tooltip";
import { cn } from "../../lib/utils";

interface DetectionCanvasProps {
  imageSrc: string;
  detections: DetectionItem[];
  selectedIndex: number | null;
  onSelectDetection: (index: number | null) => void;
  className?: string;
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  imageSrc,
  detections,
  selectedIndex,
  onSelectDetection,
  className,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState<boolean>(false);
  const [panStart, setPanStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

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

  // Compute transform: scale and offsets to center and zoom
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

  // Render canvas
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

    // 1. Draw Image
    ctx.save();
    ctx.translate(ox, oy);
    ctx.scale(scale, scale);
    ctx.drawImage(image, 0, 0, w, h);

    // 2. Draw Detections
    if (showBoxes) {
      detections.forEach((det, idx) => {
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

        // Constant screen stroke of 2px (or 3px when active)
        const lineWidth = (isActive ? 3 : 2) / scale;
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeHex;

        // Subtle fill on active
        if (isActive) {
          ctx.fillStyle = `rgba(${classCol.rgb[0]}, ${classCol.rgb[1]}, ${classCol.rgb[2]}, 0.20)`;
          ctx.fillRect(xmin, ymin, bw, bh);
        }

        ctx.strokeRect(xmin, ymin, bw, bh);

        // 3. Small Label Chip
        if (showLabels) {
          const scorePercent = `${(det.confidence * 100).toFixed(1)}%`;
          const labelText = `${det.class_name} ${scorePercent}`;

          // Keep label font size constant on screen (~11px)
          const targetFontSize = 11;
          const fontPx = targetFontSize / scale;
          ctx.font = `500 ${fontPx}px "Geist Mono", monospace`;

          const padX = 5 / scale;
          const padY = 2.5 / scale;
          const metrics = ctx.measureText(labelText);
          const tagW = metrics.width + padX * 2;
          const tagH = fontPx + padY * 2;

          // Position chip at top-left of box (inside or above)
          let tagY = ymin - tagH;
          if (tagY < 0) {
            tagY = ymin;
          }
          const tagX = Math.max(0, Math.min(xmin, w - tagW));

          // Chip background
          ctx.fillStyle = strokeHex;
          ctx.fillRect(tagX, tagY, tagW, tagH);

          // Chip text
          ctx.fillStyle = "#ffffff";
          ctx.textBaseline = "middle";
          ctx.fillText(labelText, tagX + padX, tagY + tagH / 2);
        }
      });
    }

    ctx.restore();
    ctx.restore();
  }, [image, detections, selectedIndex, hoveredIndex, showLabels, showBoxes, getTransform]);

  useEffect(() => {
    render();
  }, [render]);

  // Window resize handler
  useEffect(() => {
    const handleResize = () => render();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, [render]);

  // Mouse move: detect hover box
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
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

    // Check in reverse order
    for (let i = detections.length - 1; i >= 0; i--) {
      const b = detections[i].box_normalized;
      if (
        normX >= b.xmin &&
        normX <= b.xmax &&
        normY >= b.ymin &&
        normY <= b.ymax
      ) {
        setHoveredIndex(i);
        return;
      }
    }
    setHoveredIndex(null);
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    // Middle click or space+click or drag on background to pan
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

  // Wheel zoom
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.87;
    setZoom((z) => Math.min(Math.max(0.5, z * zoomFactor), 5));
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  return (
    <div
      ref={containerRef}
      className={cn(
        "relative w-full h-full min-h-[380px] bg-canvas border border-border-subtle rounded-panel overflow-hidden flex items-center justify-center select-none",
        isFullscreen && "fixed inset-0 z-50 rounded-none border-none",
        className
      )}
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          setHoveredIndex(null);
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

        <Tooltip content={showLabels ? "Hide labels" : "Show labels"}>
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

        <Tooltip content={showBoxes ? "Hide bounding boxes" : "Show bounding boxes"}>
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

        <div className="h-3.5 w-px bg-border-subtle mx-0.5" />

        <Tooltip content={isFullscreen ? "Exit fullscreen" : "Fullscreen"}>
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
};
