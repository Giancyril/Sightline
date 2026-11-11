import React, { useEffect, useRef, useState, useCallback } from "react";
import type { DetectionItem } from "../types";

interface DetectionCanvasProps {
  imageSrc: string;
  detections: DetectionItem[];
  selectedIndex: number | null;
  onSelectDetection: (index: number | null) => void;
}

export const DetectionCanvas: React.FC<DetectionCanvasProps> = ({
  imageSrc,
  detections,
  selectedIndex,
  onSelectDetection,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Load image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => setImage(img);
    img.src = imageSrc;
  }, [imageSrc]);

  // Render canvas
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const naturalW = image.naturalWidth;
    const naturalH = image.naturalHeight;

    canvas.width = naturalW;
    canvas.height = naturalH;

    // Draw background image
    ctx.drawImage(image, 0, 0, naturalW, naturalH);

    // Draw detections
    detections.forEach((det, idx) => {
      const isSelected = selectedIndex === idx;
      const isHovered = hoveredIndex === idx;
      const isActive = isSelected || isHovered;

      const ymin = det.box_normalized.ymin * naturalH;
      const xmin = det.box_normalized.xmin * naturalW;
      const ymax = det.box_normalized.ymax * naturalH;
      const xmax = det.box_normalized.xmax * naturalW;

      const width = xmax - xmin;
      const height = ymax - ymin;

      const strokeColor = det.color_hex;

      // 1. Box outline and fill
      ctx.lineWidth = isActive ? 4 : 2;
      ctx.strokeStyle = strokeColor;
      ctx.strokeRect(xmin, ymin, width, height);

      // Subtle fill
      ctx.fillStyle = isActive
        ? `rgba(${det.color_rgb.join(",")}, 0.28)`
        : `rgba(${det.color_rgb.join(",")}, 0.12)`;
      ctx.fillRect(xmin, ymin, width, height);

      // 2. Label tag
      const labelText = `${det.class_name} ${(det.confidence * 100).toFixed(1)}%`;
      const fontSize = Math.max(12, Math.min(18, Math.round(naturalW * 0.022)));
      ctx.font = `600 ${fontSize}px sans-serif`;

      const textMetrics = ctx.measureText(labelText);
      const paddingX = 6;
      const tagHeight = fontSize + 8;
      const tagWidth = textMetrics.width + paddingX * 2;

      // Ensure tag stays within image bounds
      const tagY = ymin >= tagHeight ? ymin - tagHeight : ymin;
      const tagX = Math.min(xmin, naturalW - tagWidth);

      // Tag background
      ctx.fillStyle = strokeColor;
      ctx.fillRect(tagX, tagY, tagWidth, tagHeight);

      // Tag text
      ctx.fillStyle = "#ffffff";
      ctx.textBaseline = "middle";
      ctx.fillText(labelText, tagX + paddingX, tagY + tagHeight / 2);
    });
  }, [image, detections, selectedIndex, hoveredIndex]);

  useEffect(() => {
    renderCanvas();
  }, [renderCanvas]);

  // Mouse interaction to detect hovered box
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const mouseX = (e.clientX - rect.left) * scaleX;
    const mouseY = (e.clientY - rect.top) * scaleY;

    // Check hit in reverse order (top-most box first)
    for (let i = detections.length - 1; i >= 0; i--) {
      const det = detections[i];
      const ymin = det.box_normalized.ymin * canvas.height;
      const xmin = det.box_normalized.xmin * canvas.width;
      const ymax = det.box_normalized.ymax * canvas.height;
      const xmax = det.box_normalized.xmax * canvas.width;

      if (mouseX >= xmin && mouseX <= xmax && mouseY >= ymin && mouseY <= ymax) {
        setHoveredIndex(i);
        return;
      }
    }
    setHoveredIndex(null);
  };

  const handleClick = () => {
    onSelectDetection(hoveredIndex);
  };

  return (
    <div
      ref={containerRef}
      className="relative w-full overflow-hidden rounded-xl bg-slate-950 flex items-center justify-center border border-slate-800"
    >
      <canvas
        ref={canvasRef}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredIndex(null)}
        onClick={handleClick}
        className="max-h-[600px] w-auto max-w-full object-contain cursor-crosshair"
      />
    </div>
  );
};

