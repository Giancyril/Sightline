import React, {
  useState,
  useRef,
  useMemo,
  useEffect,
  Suspense,
  lazy,
} from "react";
import { Upload, Camera, RefreshCw, X, HelpCircle } from "lucide-react";
import { DetectionCanvas, type DetectionCanvasRef } from "../components/detect/DetectionCanvas";
import { EmptyStateDropzone } from "../components/detect/EmptyStateDropzone";
import { Inspector } from "../components/detect/Inspector";
import { ExportMenu } from "../components/detect/ExportMenu";
import { ShortcutsDialog } from "../components/detect/ShortcutsDialog";
import { Button } from "../components/ui/Button";
import { Skeleton } from "../components/ui/Skeleton";
import { detectImage } from "../api";
import type { DetectionResponse, HistoryRecord } from "../types";

// Lazy-load webcam workspace module
const WebcamWorkspace = lazy(() => import("../components/detect/WebcamWorkspace"));

interface DetectViewProps {
  onSaveHistory: (record: HistoryRecord) => void;
  initialRecord?: HistoryRecord | null;
}

export const DetectView: React.FC<DetectViewProps> = ({
  onSaveHistory,
  initialRecord,
}) => {
  const [mode, setMode] = useState<"upload" | "webcam">("upload");
  const [imagePreview, setImagePreview] = useState<string | null>(
    () => initialRecord?.thumbnail || initialRecord?.previewUrl || null
  );
  const [currentBlob, setCurrentBlob] = useState<Blob | null>(null);

  // Upload progress & inference state
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isApiOffline, setIsApiOffline] = useState<boolean>(false);

  // Results & Interaction state
  const [rawResult, setRawResult] = useState<DetectionResponse | null>(
    () => initialRecord?.result || null
  );
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  // Sliders
  const [confThreshold, setConfThreshold] = useState<number>(
    () => initialRecord?.result?.thresholds_applied?.confidence_threshold || 0.25
  );
  const [nmsThreshold, setNmsThreshold] = useState<number>(
    () => initialRecord?.result?.thresholds_applied?.nms_iou_threshold || 0.45
  );

  // Shortcuts Dialog state
  const [shortcutsOpen, setShortcutsOpen] = useState<boolean>(false);

  // Accessibility aria-live announcement
  const [announcement, setAnnouncement] = useState<string>("");

  const canvasRef = useRef<DetectionCanvasRef>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Run backend detection
  const runDetection = async (blob: Blob, nms: number = nmsThreshold) => {
    try {
      setIsDetecting(true);
      setErrorMsg(null);
      setIsApiOffline(false);
      setSelectedIndex(null);
      setHoveredIndex(null);

      // Fetch at baseline 0.05 to allow zero-latency client-side re-filtering
      const res = await detectImage(blob, 0.05, nms);
      setRawResult(res);

      const count = res.detections.filter((d) => d.confidence >= confThreshold).length;
      setAnnouncement(
        `Detection complete: ${count} objects detected in ${res.inference_time_ms.toFixed(1)} milliseconds.`
      );

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
        setAnnouncement("API inference server is offline or unreachable.");
      } else {
        setErrorMsg(message);
        setAnnouncement(`Error during inference: ${message}`);
      }
    } finally {
      setIsDetecting(false);
    }
  };

  // Determinate upload flow
  const handleProcessFile = (file: Blob | File) => {
    setCurrentBlob(file);
    setUploadProgress(0);

    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);

    setTimeout(() => setUploadProgress(40), 40);
    setTimeout(() => setUploadProgress(85), 100);
    setTimeout(() => {
      setUploadProgress(100);
      setTimeout(() => {
        setUploadProgress(null);
        void runDetection(file);
      }, 100);
    }, 150);
  };

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

  const handleClear = () => {
    setImagePreview(null);
    setCurrentBlob(null);
    setRawResult(null);
    setSelectedIndex(null);
    setHoveredIndex(null);
    setErrorMsg(null);
    setUploadProgress(null);
    setAnnouncement("Workspace cleared.");
  };

  // Instant Client-Filtered Detections based on live confidence threshold
  const clientFilteredDetections = useMemo(() => {
    if (!rawResult) return [];
    return rawResult.detections.filter((d) => d.confidence >= confThreshold);
  }, [rawResult, confThreshold]);

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (
        target?.tagName === "INPUT" ||
        target?.tagName === "TEXTAREA" ||
        target?.isContentEditable
      ) {
        return;
      }

      if (e.key === "u" || e.key === "U") {
        e.preventDefault();
        fileInputRef.current?.click();
        return;
      }

      if (e.code === "Space" && imagePreview) {
        e.preventDefault();
        canvasRef.current?.toggleLabels();
        return;
      }

      if ((e.key === "b" || e.key === "B") && imagePreview) {
        e.preventDefault();
        canvasRef.current?.toggleBoxes();
        return;
      }

      if (e.key === "Escape") {
        setSelectedIndex(null);
        setHoveredIndex(null);
        return;
      }

      if (e.key === "?" && !e.ctrlKey && !e.metaKey) {
        e.preventDefault();
        setShortcutsOpen((o) => !o);
        return;
      }

      if (clientFilteredDetections.length > 0) {
        if (e.key === "ArrowDown" || e.key === "ArrowRight") {
          e.preventDefault();
          setSelectedIndex((prev) => {
            if (prev === null || prev >= clientFilteredDetections.length - 1) return 0;
            return prev + 1;
          });
        } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
          e.preventDefault();
          setSelectedIndex((prev) => {
            if (prev === null || prev <= 0) return clientFilteredDetections.length - 1;
            return prev - 1;
          });
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [imagePreview, clientFilteredDetections.length]);

  return (
    <div className="flex flex-col w-full h-[calc(100vh-140px)] min-h-[580px] max-[899px]:h-auto gap-3">
      {/* Screen-reader aria-live announcement */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </div>

      {/* Hidden File Input */}
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

      {/* Top Workspace Toolbar */}
      <div className="flex items-center justify-between pb-2 border-b border-border-subtle shrink-0">
        <div className="flex items-center gap-2">
          {/* Mode Selector */}
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

        {/* Right Action Cluster */}
        <div className="flex items-center gap-2">
          {/* Export Menu */}
          <ExportMenu
            detections={clientFilteredDetections}
            rawResult={rawResult}
            confThreshold={confThreshold}
            nmsThreshold={nmsThreshold}
            onExportAnnotatedImage={() => canvasRef.current?.exportAnnotatedPng()}
            disabled={!imagePreview || clientFilteredDetections.length === 0}
          />

          {imagePreview && (
            <>
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
            </>
          )}

          {/* Keyboard Shortcuts Trigger */}
          <button
            type="button"
            onClick={() => setShortcutsOpen(true)}
            className="flex h-7 w-7 items-center justify-center rounded-control border border-border-subtle bg-card text-text-tertiary hover:text-text-primary hover:bg-elevated transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
            title="Keyboard shortcuts (?)"
            aria-label="Keyboard shortcuts"
          >
            <HelpCircle className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Determinate Upload Progress Bar */}
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
        /* Lazy-loaded Webcam Workspace */
        <Suspense
          fallback={
            <div className="flex-1 flex items-center justify-center p-8 bg-card border border-border-subtle rounded-panel">
              <Skeleton className="w-full h-80 max-w-xl" />
            </div>
          }
        >
          <WebcamWorkspace
            onSaveHistory={onSaveHistory}
            onExitWebcam={() => setMode("upload")}
          />
        </Suspense>
      ) : !imagePreview ? (
        /* Empty State Dropzone */
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
          {/* Left: Canvas Area (~68%) */}
          <div className="flex-[68] min-w-0 h-full max-[899px]:h-[440px] flex flex-col overflow-hidden">
            <DetectionCanvas
              ref={canvasRef}
              imageSrc={imagePreview}
              detections={clientFilteredDetections}
              selectedIndex={selectedIndex}
              onSelectDetection={setSelectedIndex}
              hoveredIndex={hoveredIndex}
              onHoverDetection={setHoveredIndex}
              enableComparison={true}
            />
          </div>

          {/* Right: Inspector Pane (~32%) */}
          <div className="flex-[32] min-w-[300px] max-w-[420px] max-[899px]:max-w-none max-[899px]:w-full h-full max-[899px]:h-[480px] shrink-0 overflow-hidden">
            <Inspector
              detections={clientFilteredDetections}
              allDetections={rawResult?.detections || []}
              inferenceMs={rawResult?.inference_time_ms ?? null}
              confThreshold={confThreshold}
              nmsThreshold={nmsThreshold}
              onConfChange={setConfThreshold}
              onNmsChange={handleNmsChange}
              selectedIndex={selectedIndex}
              onSelectIndex={setSelectedIndex}
              hoveredIndex={hoveredIndex}
              onHoverIndex={setHoveredIndex}
              isLoading={isDetecting}
              errorMsg={errorMsg}
              onRetry={handleRetry}
              isApiOffline={isApiOffline}
            />
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Dialog */}
      <ShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
};
