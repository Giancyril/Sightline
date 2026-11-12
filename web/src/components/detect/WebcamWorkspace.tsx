import React, { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  Camera,
  CameraOff,
  Pause,
  Play,
  RotateCcw,
  Zap,
  Activity,
  Layers,
  Box,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Slider } from "../ui/Slider";
import { detectImage } from "../../api";
import { colorForClass } from "../../lib/colors";
import type { DetectionItem, DetectionResponse, HistoryRecord } from "../../types";

interface WebcamWorkspaceProps {
  onSaveHistory: (record: HistoryRecord) => void;
  onExitWebcam: () => void;
}

export const WebcamWorkspace: React.FC<WebcamWorkspaceProps> = ({
  onSaveHistory,
  onExitWebcam,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [permissionState, setPermissionState] = useState<"idle" | "granted" | "denied">("idle");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);

  // Metrics
  const [fps, setFps] = useState<number>(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [detections, setDetections] = useState<DetectionItem[]>([]);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  // Sliders
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [nmsThreshold, setNmsThreshold] = useState<number>(0.45);

  const lastFrameTimeRef = useRef<number>(0);
  const loopRef = useRef<number | null>(null);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (loopRef.current) {
      cancelAnimationFrame(loopRef.current);
      loopRef.current = null;
    }
    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setPermissionState("idle");
  }, []);

  // Request & start camera
  const startCamera = useCallback(async () => {
    try {
      setErrorMsg(null);
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
        setPermissionState("granted");
        setIsPaused(false);
      }
    } catch (err) {
      setPermissionState("denied");
      setErrorMsg((err as Error).message);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    const init = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: { ideal: 640 }, height: { ideal: 480 } },
          audio: false,
        });

        if (mounted && videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          await videoRef.current.play();
          setPermissionState("granted");
          setIsPaused(false);
        }
      } catch (err) {
        if (mounted) {
          setPermissionState("denied");
          setErrorMsg((err as Error).message);
        }
      }
    };

    void init();
    return () => {
      mounted = false;
      stopCamera();
    };
  }, [stopCamera]);

  // Capture frame as blob
  const getFrameBlob = useCallback(async (): Promise<Blob | null> => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return null;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = video.videoWidth || 640;
    tempCanvas.height = video.videoHeight || 480;
    const ctx = tempCanvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

    return new Promise((resolve) => {
      tempCanvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.85);
    });
  }, []);

  // Periodic detection run
  const runFrameDetection = useCallback(async () => {
    if (isProcessing || isPaused || permissionState !== "granted") return;

    const blob = await getFrameBlob();
    if (!blob) return;

    const start = performance.now();
    try {
      setIsProcessing(true);
      const res: DetectionResponse = await detectImage(blob, confThreshold, nmsThreshold);
      setDetections(res.detections);
      setLatencyMs(Math.round(performance.now() - start));
    } catch {
      // Quiet fail on dropped frame
    } finally {
      setIsProcessing(false);
    }
  }, [isProcessing, isPaused, permissionState, getFrameBlob, confThreshold, nmsThreshold]);

  // Main animation / render loop
  useEffect(() => {
    if (permissionState !== "granted") return;

    let timer: ReturnType<typeof setInterval>;

    const renderLoop = () => {
      const now = performance.now();
      if (lastFrameTimeRef.current > 0) {
        const delta = now - lastFrameTimeRef.current;
        if (delta > 0) {
          setFps(Math.round(1000 / delta));
        }
      }
      lastFrameTimeRef.current = now;

      // Draw video and overlays to canvas
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.readyState >= 2 && !isPaused) {
        const ctx = canvas.getContext("2d");
        if (ctx) {
          if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Draw detections
          detections.forEach((det, idx) => {
            const isSelected = selectedIdx === idx;
            const x = det.box_normalized.xmin * canvas.width;
            const y = det.box_normalized.ymin * canvas.height;
            const w = (det.box_normalized.xmax - det.box_normalized.xmin) * canvas.width;
            const h = (det.box_normalized.ymax - det.box_normalized.ymin) * canvas.height;

            const classCol = colorForClass(det.class_id);

            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeStyle = classCol.hex;
            ctx.strokeRect(x, y, w, h);

            // Label tag
            const label = `${det.class_name} ${(det.confidence * 100).toFixed(0)}%`;
            ctx.font = '600 12px "Geist Mono", monospace';
            const textMetrics = ctx.measureText(label);
            const tagW = textMetrics.width + 8;
            const tagH = 18;
            const tagY = y >= tagH ? y - tagH : y;

            ctx.fillStyle = classCol.hex;
            ctx.fillRect(x, tagY, tagW, tagH);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(label, x + 4, tagY + 13);
          });
        }
      }

      loopRef.current = requestAnimationFrame(renderLoop);
    };

    loopRef.current = requestAnimationFrame(renderLoop);

    // Call inference every 800ms
    timer = setInterval(() => {
      void runFrameDetection();
    }, 800);

    return () => {
      if (loopRef.current) cancelAnimationFrame(loopRef.current);
      clearInterval(timer);
    };
  }, [permissionState, isPaused, detections, selectedIdx, runFrameDetection]);

  // Snapshot to History
  const handleSnapshot = async () => {
    const blob = await getFrameBlob();
    if (!blob) return;

    try {
      setIsProcessing(true);
      const res = await detectImage(blob, confThreshold, nmsThreshold);
      const previewUrl = URL.createObjectURL(blob);

      onSaveHistory({
        id: Math.random().toString(36).substring(2, 9),
        timestamp: new Date().toLocaleTimeString(),
        previewUrl,
        result: res,
      });
    } catch (err) {
      setErrorMsg(`Snapshot failed: ${(err as Error).message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const uniqueClassesCount = useMemo(() => {
    return new Set(detections.map((d) => d.class_id)).size;
  }, [detections]);

  // Permission Denied State
  if (permissionState === "denied") {
    return (
      <div className="flex flex-col items-center justify-center h-full min-h-[460px] p-8 bg-card border border-border-subtle rounded-panel text-center">
        <div className="flex h-12 w-12 items-center justify-center rounded-control bg-status-danger-subtle text-status-danger mb-3">
          <CameraOff className="w-6 h-6" />
        </div>
        <h3 className="text-base font-semibold text-text-primary mb-1">
          Camera Access Blocked
        </h3>
        <p className="text-xs text-text-secondary max-w-sm mb-4 leading-relaxed">
          {errorMsg || "Camera permission was denied. Enable camera access in your browser settings to continue."}
        </p>
        <div className="flex items-center gap-2">
          <Button
            variant="primary"
            size="sm"
            onClick={startCamera}
            leftIcon={<RotateCcw className="w-3.5 h-3.5" />}
          >
            Try again
          </Button>
          <Button variant="secondary" size="sm" onClick={onExitWebcam}>
            Back to images
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-[900px]:flex-row gap-4 min-h-0 overflow-hidden max-[899px]:overflow-visible">
      {/* Hidden Video element providing frames */}
      <video ref={videoRef} playsInline muted className="hidden" />

      {/* Left Pane: Camera Canvas (68%) */}
      <div className="flex-[68] min-w-0 h-full max-[899px]:h-[440px] flex flex-col bg-canvas border border-border-subtle rounded-panel overflow-hidden relative">
        <canvas
          ref={canvasRef}
          className="w-full h-full object-contain block bg-canvas"
        />

        {/* Floating Controls Overlay */}
        <div className="absolute top-3 left-3 flex items-center gap-2 bg-card/90 backdrop-blur-md border border-border-subtle rounded-control px-2.5 py-1 text-xs">
          <span className="flex items-center gap-1.5 font-medium text-text-primary">
            <span
              className={`w-2 h-2 rounded-full ${
                !isPaused ? "bg-status-success animate-pulse" : "bg-text-tertiary"
              }`}
            />
            {!isPaused ? "Live feed" : "Paused"}
          </span>
          <span className="text-text-tertiary">·</span>
          <span className="font-mono text-[11px] text-text-secondary tabular-nums">
            {fps} FPS
          </span>
          {latencyMs !== null && (
            <>
              <span className="text-text-tertiary">·</span>
              <span className="font-mono text-[11px] text-text-secondary tabular-nums">
                {latencyMs} ms
              </span>
            </>
          )}
        </div>

        {/* Bottom Floating Action Bar */}
        <div className="absolute bottom-3 left-3 right-3 flex items-center justify-between">
          <div className="flex items-center gap-1.5 bg-card/90 backdrop-blur-md border border-border-subtle rounded-control p-1">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setIsPaused((p) => !p)}
              leftIcon={
                isPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />
              }
            >
              {isPaused ? "Resume" : "Pause"}
            </Button>
            <Button
              variant="primary"
              size="sm"
              onClick={handleSnapshot}
              isLoading={isProcessing}
              leftIcon={<Camera className="w-3.5 h-3.5" />}
            >
              Snapshot to history
            </Button>
          </div>

          <Button variant="secondary" size="sm" onClick={onExitWebcam}>
            Exit webcam
          </Button>
        </div>
      </div>

      {/* Right Pane: Live Inspector (32%) */}
      <div className="flex-[32] min-w-[300px] max-w-[420px] max-[899px]:max-w-none max-[899px]:w-full h-full max-[899px]:h-[480px] shrink-0 overflow-hidden flex flex-col bg-card border border-border-subtle rounded-panel">
        {/* Summary Strip */}
        <div className="p-3.5 border-b border-border-subtle bg-subtle/50">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-text-tertiary">
              Live Inspector
            </span>
            <span className="text-[11px] font-mono text-brand flex items-center gap-1">
              <Activity className="w-3 h-3" /> Streaming
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2">
            <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
              <span className="text-[10px] text-text-tertiary flex items-center gap-1">
                <Box className="w-3 h-3 opacity-60" /> Objects
              </span>
              <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
                {detections.length}
              </span>
            </div>

            <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
              <span className="text-[10px] text-text-tertiary flex items-center gap-1">
                <Layers className="w-3 h-3 opacity-60" /> Classes
              </span>
              <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
                {uniqueClassesCount}
              </span>
            </div>

            <div className="flex flex-col p-2 rounded-control bg-card border border-border-subtle">
              <span className="text-[10px] text-text-tertiary flex items-center gap-1">
                <Zap className="w-3 h-3 opacity-60" /> Latency
              </span>
              <span className="text-sm font-semibold font-mono text-text-primary tabular-nums mt-0.5">
                {latencyMs !== null ? `${latencyMs} ms` : "—"}
              </span>
            </div>
          </div>
        </div>

        {/* Sliders */}
        <div className="p-3.5 border-b border-border-subtle flex flex-col gap-3">
          <Slider
            label="Confidence threshold"
            value={confThreshold}
            min={0.05}
            max={0.95}
            step={0.05}
            onValueChange={setConfThreshold}
          />
          <Slider
            label="NMS IoU threshold"
            value={nmsThreshold}
            min={0.10}
            max={0.90}
            step={0.05}
            onValueChange={setNmsThreshold}
          />
        </div>

        {/* Live Detections List */}
        <div className="flex-1 overflow-y-auto p-2">
          {detections.length === 0 ? (
            <div className="py-8 text-center text-xs text-text-tertiary">
              No objects in current frame above {(confThreshold * 100).toFixed(0)}%
            </div>
          ) : (
            <div className="flex flex-col gap-1">
              {detections.map((det, idx) => {
                const classCol = colorForClass(det.class_id);
                const isSelected = selectedIdx === idx;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setSelectedIdx(isSelected ? null : idx)}
                    className={`flex items-center justify-between w-full px-2.5 py-1.5 rounded-control text-left transition-colors border select-none ${
                      isSelected
                        ? "bg-elevated border-brand shadow-sm"
                        : "bg-card border-transparent hover:bg-subtle hover:border-border-subtle"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: classCol.hex }}
                      />
                      <span className="text-xs font-medium text-text-primary">
                        {det.class_name}
                      </span>
                    </div>
                    <span className="font-mono text-xs font-medium text-text-primary tabular-nums">
                      {(det.confidence * 100).toFixed(1)}%
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default WebcamWorkspace;
