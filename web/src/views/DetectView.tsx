import React, { useState, useRef, useMemo, useCallback } from "react";
import { Upload, Camera, RefreshCw, X } from "lucide-react";
import { DetectionCanvas } from "../components/detect/DetectionCanvas";
import { EmptyStateDropzone } from "../components/detect/EmptyStateDropzone";
import { Inspector } from "../components/detect/Inspector";
import { WebcamCapture } from "../components/WebcamCapture";
import { Button } from "../components/ui/Button";
import { detectImage } from "../api";
import type { DetectionResponse, HistoryRecord } from "../types";

interface DetectViewProps {
  onSaveHistory: (record: HistoryRecord) => void;
}

export const DetectView: React.FC<DetectViewProps> = ({ onSaveHistory }) => {
  const [mode, setMode] = useState<"upload" | "webcam">("upload");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null);

  // Upload progress & state
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isApiOffline, setIsApiOffline] = useState<boolean>(false);

  // Raw API response
  const [rawResult, setRawResult] = useState<DetectionResponse | null>(null);
  const [selectedDetectionIdx, setSelectedDetectionIdx] = useState<number | null>(null);

  // Sliders
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [nmsThreshold, setNmsThreshold] = useState<number>(0.45);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Process detection with API
  const runDetection = useCallback(
    async (blob: Blob, nms: number = nmsThreshold) => {
      try {
        setIsDetecting(true);
        setErrorMsg(null);
        setIsApiOffline(false);
        setSelectedDetectionIdx(null);

        // Call backend API (we query at confidence 0.05 so client can filter up to 0.95 instantly)
        const res = await detectImage(blob, 0.05, nms);
        setRawResult(res);

        // Save to session history
        onSaveHistory({
          id: Math.random().toString(36).substring(2, 9),
          timestamp: new Date().toLocaleTimeString(),
          previewUrl: URL.createObjectURL(blob),
          result: res,
        });
      } catch (err) {
        const message = (err as Error).message;
        if (
          message.includes("Failed to fetch") ||
          message.includes("NetworkError") ||
          message.includes("status 502") ||
          message.includes("status 503")
        ) {
          setIsApiOffline(true);
        } else {
          setErrorMsg(message);
        }
      } finally {
        setIsDetecting(false);
      }
    },
    [nmsThreshold, onSaveHistory]
  );

  // Determinate upload flow
  const handleProcessFile = useCallback(
    (file: Blob | File) => {
      setCurrentBlob(file);
      setUploadProgress(0);

      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);

      // Simulate rapid progress for visual feedback
      const timer1 = setTimeout(() => setUploadProgress(45), 40);
      const timer2 = setTimeout(() => setUploadProgress(90), 100);
      const timer3 = setTimeout(() => {
        setUploadProgress(100);
        setTimeout(() => {
          setUploadProgress(null);
          void runDetection(file);
        }, 120);
      }, 160);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer3);
      };
    },
    [runDetection]
  );

  // Load sample image
  const handleSelectSample = async (sampleSrc: string) => {
    try {
      const response = await fetch(sampleSrc);
      const blob = await response.blob();
      handleProcessFile(blob);
    } catch (err) {
      setErrorMsg(`Failed to load sample image: ${(err as Error).message}`);
    }
  };

  // Re-run detection on current image (e.g. on retry or NMS change)
  const handleRetry = () => {
    if (currentBlob) {
      void runDetection(currentBlob, nmsThreshold);
    }
  };

  const handleNmsChange = (newNms: number) => {
    setNmsThreshold(newNms);
    if (currentBlob) {
      void runDetection(currentBlob, newNms);
    }
  };

  // Clear current image and reset
  const handleClear = () => {
    setImagePreview(null);
    setCurrentBlob(null);
    setRawResult(null);
    setSelectedDetectionIdx(null);
    setErrorMsg(null);
    setUploadProgress(null);
  };

  // Instant Client-Filtered Detections based on live confidence threshold
  const clientFilteredDetections = useMemo(() => {
    if (!rawResult) return [];
    return rawResult.detections.filter((d) => d.confidence >= confThreshold);
  }, [rawResult, confThreshold]);

  return (
    <div className="flex flex-col w-full h-[calc(100vh-140px)] min-h-[580px] max-[899px]:h-auto gap-3">
      {/* Top Workspace Toolbar */}
      <div className="flex items-center justify-between pb-2 border-b border-border-subtle shrink-0">
        <div className="flex items-center gap-2">
          {/* Mode Switcher */}
          <div className="inline-flex p-0.5 rounded-control bg-elevated border border-border-subtle">
            <button
              type="button"
              onClick={() => {
                setMode("upload");
                if (!imagePreview) handleClear();
              }}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand ${
                mode === "upload"
                  ? "bg-card text-text-primary shadow-sm border border-border-subtle"
                  : "text-text-tertiary hover:text-text-primary"
              }`}
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Image</span>
            </button>
            <button
              type="button"
              onClick={() => setMode("webcam")}
              className={`flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand ${
                mode === "webcam"
                  ? "bg-card text-text-primary shadow-sm border border-border-subtle"
                  : "text-text-tertiary hover:text-text-primary"
              }`}
            >
              <Camera className="w-3.5 h-3.5" />
              <span>Webcam</span>
            </button>
          </div>

          {imagePreview && (
            <span className="text-[11px] font-mono text-text-tertiary hidden sm:inline">
              {rawResult?.image_dimensions
                ? `${rawResult.image_dimensions.width}×${rawResult.image_dimensions.height} px`
                : "Active image"}
            </span>
          )}
        </div>

        {/* Right actions when image is active */}
        {imagePreview && (
          <div className="flex items-center gap-2">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleProcessFile(file);
              }}
              className="hidden"
            />
            <Button
              variant="secondary"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              leftIcon={<RefreshCw className="w-3.5 h-3.5" />}
            >
              Change image
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={handleClear}
              leftIcon={<X className="w-3.5 h-3.5" />}
              aria-label="Clear image"
            >
              Clear
            </Button>
          </div>
        )}
      </div>

      {/* Determinate upload progress bar */}
      {uploadProgress !== null && (
        <div className="w-full bg-elevated h-1 rounded-full overflow-hidden shrink-0">
          <div
            className="bg-brand h-full transition-all duration-150"
            style={{ width: `${uploadProgress}%` }}
          />
        </div>
      )}

      {/* Main Workspace Body */}
      {mode === "webcam" ? (
        <div className="flex-1 min-h-[460px] flex items-center justify-center p-4 bg-card border border-border-subtle rounded-panel">
          <WebcamCapture onCapture={handleProcessFile} isProcessing={isDetecting} />
        </div>
      ) : !imagePreview ? (
        /* Empty State */
        <div className="flex-1 flex items-center justify-center py-8">
          <EmptyStateDropzone
            onFileSelect={handleProcessFile}
            onSelectSample={handleSelectSample}
            onSwitchToWebcam={() => setMode("webcam")}
          />
        </div>
      ) : (
        /* Two-Pane Workspace: Canvas (~68%) + Inspector (~32%) */
        <div className="flex-1 flex flex-col min-[900px]:flex-row gap-4 min-h-0 overflow-hidden max-[899px]:overflow-visible">
          {/* Left: Canvas Area (about 68%) */}
          <div className="flex-[68] min-w-0 h-full max-[899px]:h-[440px] flex flex-col overflow-hidden">
            <DetectionCanvas
              imageSrc={imagePreview}
              detections={clientFilteredDetections}
              selectedIndex={selectedDetectionIdx}
              onSelectDetection={setSelectedDetectionIdx}
            />
          </div>

          {/* Right: Inspector Pane (about 32%) */}
          <div className="flex-[32] min-w-[300px] max-w-[420px] max-[899px]:max-w-none max-[899px]:w-full h-full max-[899px]:h-[480px] shrink-0 overflow-hidden">
            <Inspector
              detections={clientFilteredDetections}
              allDetections={rawResult?.detections || []}
              inferenceMs={rawResult?.inference_time_ms ?? null}
              confThreshold={confThreshold}
              nmsThreshold={nmsThreshold}
              onConfChange={setConfThreshold}
              onNmsChange={handleNmsChange}
              selectedIndex={selectedDetectionIdx}
              onSelectIndex={setSelectedDetectionIdx}
              isLoading={isDetecting}
              errorMsg={errorMsg}
              onRetry={handleRetry}
              isApiOffline={isApiOffline}
            />
          </div>
        </div>
      )}
    </div>
  );
};
