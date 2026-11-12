import React, { useRef, useEffect } from "react";
import { Upload, Camera, Image as ImageIcon } from "lucide-react";
import { Kbd } from "../ui/Kbd";
import { Button } from "../ui/Button";
import { cn } from "../../lib/utils";

interface SampleImage {
  id: string;
  title: string;
  subtitle: string;
  src: string;
}

const SAMPLE_IMAGES: SampleImage[] = [
  {
    id: "sample-workspace",
    title: "Office Desk",
    subtitle: "Laptop, Microphone",
    src: "/samples/sample-workspace.jpg",
  },
  {
    id: "sample-desk",
    title: "Study Table",
    subtitle: "Book, Stapler",
    src: "/samples/sample-desk.jpg",
  },
  {
    id: "sample-street",
    title: "Urban Street",
    subtitle: "Clock, Backpack",
    src: "/samples/sample-street.jpg",
  },
];

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

interface EmptyStateDropzoneProps {
  onFileSelect: (file: File | Blob) => void;
  onSelectSample: (sampleSrc: string) => void;
  onSwitchToWebcam: () => void;
  className?: string;
}

export const EmptyStateDropzone: React.FC<EmptyStateDropzoneProps> = ({
  onFileSelect,
  onSelectSample,
  onSwitchToWebcam,
  className,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropzoneRef = useRef<HTMLDivElement>(null);

  // Paste from clipboard support
  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile();
          if (file) {
            onFileSelect(file);
            break;
          }
        }
      }
    };
    window.addEventListener("paste", handlePaste);
    return () => window.removeEventListener("paste", handlePaste);
  }, [onFileSelect]);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      onFileSelect(file);
    }
  };

  return (
    <div
      ref={dropzoneRef}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      className={cn(
        "flex flex-col items-center justify-center w-full max-w-xl mx-auto p-6 sm:p-8 bg-card border border-border-subtle rounded-panel text-center transition-all",
        className
      )}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file) onFileSelect(file);
        }}
        className="hidden"
      />

      {/* Refined Drop Area */}
      <div className="flex h-12 w-12 items-center justify-center rounded-control bg-elevated border border-border-subtle text-text-primary mb-3.5">
        <Upload className="w-5 h-5 text-text-secondary" />
      </div>

      <h3 className="text-base font-semibold tracking-tight text-text-primary mb-1">
        Select an image to detect
      </h3>
      <p className="text-xs text-text-secondary max-w-sm mb-4 leading-relaxed">
        Drag and drop your file here, browse from disk, or paste directly from clipboard.
      </p>

      {/* Primary Actions */}
      <div className="flex items-center gap-2 mb-4">
        <Button
          variant="primary"
          size="sm"
          onClick={() => fileInputRef.current?.click()}
          leftIcon={<ImageIcon className="w-3.5 h-3.5" />}
        >
          Browse image
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={onSwitchToWebcam}
          leftIcon={<Camera className="w-3.5 h-3.5" />}
        >
          Use webcam
        </Button>
      </div>

      {/* Keyboard and metadata hints */}
      <div className="flex items-center gap-2 text-[11px] text-text-tertiary mb-6">
        <span>JPEG, PNG, WebP · Max 10 MB</span>
        <span>·</span>
        <span className="flex items-center gap-1">
          <Kbd>{isMac ? "⌘V" : "Ctrl+V"}</Kbd> paste
        </span>
      </div>

      {/* 3 Sample Images */}
      <div className="w-full pt-4 border-t border-border-subtle">
        <p className="text-[11px] font-medium text-text-tertiary mb-3 uppercase tracking-wider text-left">
          Or try with a sample image
        </p>
        <div className="grid grid-cols-3 gap-2.5">
          {SAMPLE_IMAGES.map((sample) => (
            <button
              key={sample.id}
              type="button"
              onClick={() => onSelectSample(sample.src)}
              className="group flex flex-col overflow-hidden rounded-control border border-border-subtle bg-elevated text-left transition hover:border-border-strong focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
            >
              <div className="relative aspect-video w-full overflow-hidden bg-subtle">
                <img
                  src={sample.src}
                  alt={sample.title}
                  className="h-full w-full object-cover transition duration-150 group-hover:scale-105"
                  loading="lazy"
                />
              </div>
              <div className="p-2">
                <p className="text-xs font-medium text-text-primary group-hover:text-brand truncate">
                  {sample.title}
                </p>
                <p className="text-[10px] text-text-tertiary font-mono truncate mt-0.5">
                  {sample.subtitle}
                </p>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
