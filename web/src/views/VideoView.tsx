import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  Upload,
  Video,
  Play,
  RotateCcw,
  AlertCircle,
  CheckCircle2,
  Layers,
  Cpu,
  Download,
  FileText,
  Sliders,
  TrendingUp,
  ArrowDownToLine,
  ArrowUpFromLine,
  BarChart2,
} from "lucide-react";
import {
  uploadVideo,
  fetchVideoJobStatus,
  cancelVideoJob,
  fetchModels,
  fetchVideoSummary,
  API_BASE_URL,
} from "../api";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { EmptyState } from "../components/ui/EmptyState";
import { Slider } from "../components/ui/Slider";
import { cn } from "../lib/utils";
import type { ModelRegistryItem, VideoJobStatus, VideoSummary } from "../types";

export const VideoView: React.FC = () => {
  // Config
  const [modelId, setModelId] = useState<string>("fine-tuned");
  const [registeredModels, setRegisteredModels] = useState<ModelRegistryItem[]>([]);
  const [frameStride, setFrameStride] = useState<number>(3);
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [nmsThreshold, setNmsThreshold] = useState<number>(0.45);

  // Counting line (normalized 0-1 coords)
  const [lineEnabled, setLineEnabled] = useState<boolean>(false);
  const [lineP1, setLineP1] = useState<{ x: number; y: number }>({ x: 0.1, y: 0.5 });
  const [lineP2, setLineP2] = useState<{ x: number; y: number }>({ x: 0.9, y: 0.5 });
  const [draggingPoint, setDraggingPoint] = useState<"p1" | "p2" | null>(null);
  const lineCanvasRef = useRef<HTMLCanvasElement>(null);

  // Upload & Job state
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [job, setJob] = useState<VideoJobStatus | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isCancelled, setIsCancelled] = useState<boolean>(false);
  const [summary, setSummary] = useState<VideoSummary | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Load registered models
  useEffect(() => {
    let mounted = true;
    fetchModels()
      .then((res) => {
        if (mounted) {
          setRegisteredModels(res.models);
        }
      })
      .catch(() => {
        // Fallback default
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Polling for job status
  useEffect(() => {
    if (!job || job.state === "done" || job.state === "failed") {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
      return;
    }

    const poll = () => {
      fetchVideoJobStatus(job.job_id)
        .then((latest) => {
          setJob(latest);
          if (latest.state === "failed" && latest.error) {
            setErrorMsg(latest.error);
          }
        })
        .catch((err: Error) => {
          setErrorMsg(err.message || "Failed polling job status.");
        });
    };

    pollingTimerRef.current = setInterval(poll, 1500);

    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
        pollingTimerRef.current = null;
      }
    };
  }, [job]);

  // Fetch summary when job completes
  useEffect(() => {
    if (job?.state === "done" && !summary) {
      fetchVideoSummary(job.job_id)
        .then((s) => setSummary(s))
        .catch(() => {
          // summary is optional; fail silently
        });
    }
  }, [job, summary]);

  // Draw counting line on canvas
  useEffect(() => {
    const canvas = lineCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (!lineEnabled) return;

    const x1 = lineP1.x * W;
    const y1 = lineP1.y * H;
    const x2 = lineP2.x * W;
    const y2 = lineP2.y * H;

    ctx.strokeStyle = "#6366f1";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw handles
    for (const [hx, hy] of [[x1, y1], [x2, y2]]) {
      ctx.fillStyle = "#6366f1";
      ctx.beginPath();
      ctx.arc(hx, hy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [lineEnabled, lineP1, lineP2]);

  const handleCanvasMouseDown = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = lineCanvasRef.current;
    if (!canvas || !lineEnabled) return;
    const rect = canvas.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const dist1 = Math.hypot(nx - lineP1.x, ny - lineP1.y);
    const dist2 = Math.hypot(nx - lineP2.x, ny - lineP2.y);
    if (dist1 < 0.08) setDraggingPoint("p1");
    else if (dist2 < 0.08) setDraggingPoint("p2");
  }, [lineEnabled, lineP1, lineP2]);

  const handleCanvasMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = lineCanvasRef.current;
    if (!canvas || !draggingPoint) return;
    const rect = canvas.getBoundingClientRect();
    const nx = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const ny = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    if (draggingPoint === "p1") setLineP1({ x: nx, y: ny });
    else setLineP2({ x: nx, y: ny });
  }, [draggingPoint]);

  const handleCanvasMouseUp = useCallback(() => {
    setDraggingPoint(null);
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    setErrorMsg(null);
    setIsCancelled(false);

    // Validate size (max 50 MB)
    if (file.size > 50 * 1024 * 1024) {
      setErrorMsg(`File size ${(file.size / (1024 * 1024)).toFixed(1)} MB exceeds maximum limit of 50 MB.`);
      return;
    }

    // Validate type
    const lowerName = file.name.toLowerCase();
    if (!lowerName.endsWith(".mp4") && !lowerName.endsWith(".webm")) {
      setErrorMsg("Unsupported format. Please upload an MP4 or WebM video file.");
      return;
    }

    setSelectedFile(file);
    setJob(null);
  }, []);

  const handleStartProcessing = async () => {
    if (!selectedFile) return;

    setIsUploading(true);
    setErrorMsg(null);
    setIsCancelled(false);
    setSummary(null);

    try {
      const uploadRes = await uploadVideo(selectedFile, {
        model: modelId,
        frameStride,
        confidenceThreshold: confThreshold,
        nmsIouThreshold: nmsThreshold,
        countingLine: lineEnabled
          ? { p1: lineP1, p2: lineP2 }
          : undefined,
      });

      // Initialize job state from response
      setJob({
        job_id: uploadRes.job_id,
        state: "running",
        progress: 0.0,
        frames_processed: 0,
        total_frames_to_process: uploadRes.total_frames_to_process,
        fps: uploadRes.fps,
        duration_s: uploadRes.duration_s,
        width: uploadRes.width,
        height: uploadRes.height,
        has_results: false,
      });
    } catch (err: unknown) {
      setErrorMsg((err as Error).message || "Failed to upload video clip.");
    } finally {
      setIsUploading(false);
    }
  };

  const handleCancelJob = async () => {
    if (!job) return;
    try {
      await cancelVideoJob(job.job_id);
      setIsCancelled(true);
      setJob((prev) => (prev ? { ...prev, state: "failed", error: "Cancelled by user." } : null));
    } catch {
      setIsCancelled(true);
    }
  };

  const handleReset = () => {
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
    setJob(null);
    setSelectedFile(null);
    setErrorMsg(null);
    setIsCancelled(false);
    setSummary(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      {/* Left / Main Workspace */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        {/* Empty State / Upload Dropzone when no job or file */}
        {!job && !selectedFile && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            className={cn(
              "relative flex flex-col items-center justify-center p-12 rounded-panel border-2 border-dashed transition-all select-none min-h-[380px]",
              isDragging
                ? "border-brand bg-brand-subtle/20"
                : "border-border-subtle bg-subtle/40 hover:bg-subtle/70 hover:border-border-muted"
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/mp4,video/webm"
              className="hidden"
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileSelect(e.target.files[0]);
                }
              }}
            />

            <EmptyState
              icon={<Video className="w-8 h-8 text-brand" />}
              title="Drop a video clip to begin"
              description="Upload an MP4 or WebM video (up to 30s, 50MB) for per-frame detection, multi-object tracking and directional line counting."
              action={
                <Button
                  variant="primary"
                  onClick={() => fileInputRef.current?.click()}
                  leftIcon={<Upload className="w-4 h-4" />}
                >
                  Select Video File
                </Button>
              }
            />

            <div className="mt-6 flex flex-wrap items-center justify-center gap-4 text-xs text-text-tertiary">
              <span className="flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5 text-status-success" /> MP4 & WebM container
              </span>
              <span>·</span>
              <span>Max duration: 30 seconds</span>
              <span>·</span>
              <span>Max file size: 50 MB</span>
            </div>
          </div>
        )}

        {/* Selected file preview & ready-to-run state */}
        {!job && selectedFile && (
          <div className="rounded-panel border border-border-subtle bg-card p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between border-b border-border-subtle pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2.5 rounded-control bg-elevated text-brand border border-border-subtle">
                  <Video className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-text-primary">
                    {selectedFile.name}
                  </h3>
                  <p className="text-xs text-text-secondary">
                    {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB · Ready for analysis
                  </p>
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={handleReset}>
                Change File
              </Button>
            </div>

            <div className="flex items-center justify-between pt-2">
              <p className="text-xs text-text-tertiary">
                Processing configured model: <span className="font-semibold text-text-secondary">{modelId}</span> (stride: {frameStride})
              </p>
              <Button
                variant="primary"
                onClick={handleStartProcessing}
                isLoading={isUploading}
                leftIcon={<Play className="w-4 h-4" />}
              >
                Start Processing
              </Button>
            </div>
          </div>
        )}

        {/* Active Job Processing State (Queued or Running) */}
        {job && (job.state === "queued" || job.state === "running") && (
          <div className="rounded-panel border border-border-subtle bg-card p-8 flex flex-col items-center justify-center text-center gap-5 min-h-[320px]">
            <div className="relative flex items-center justify-center w-16 h-16 rounded-full bg-brand-subtle/30 text-brand animate-pulse">
              <Video className="w-8 h-8" />
            </div>

            <div className="max-w-md w-full">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Badge variant="brand">
                    {job.state === "queued" ? "Queued" : "Processing"}
                  </Badge>
                  <span className="text-xs text-text-secondary font-medium">
                    Job #{job.job_id.slice(0, 8)}
                  </span>
                </div>
                <span className="text-xs font-mono font-semibold text-brand">
                  {Math.round(job.progress * 100)}%
                </span>
              </div>

              {/* Determinate Progress Bar */}
              <div className="w-full h-2 rounded-full bg-elevated overflow-hidden border border-border-subtle">
                <div
                  className="h-full bg-brand transition-all duration-300 ease-out rounded-full"
                  style={{ width: `${Math.max(4, Math.round(job.progress * 100))}%` }}
                />
              </div>

              <div className="flex items-center justify-between mt-2 text-[11px] text-text-tertiary font-mono">
                <span>
                  Frames: {job.frames_processed} / {job.total_frames_to_process}
                </span>
                <span>
                  {job.fps > 0 ? `${job.fps.toFixed(1)} fps` : "reading..."} · {job.duration_s ? `${job.duration_s.toFixed(1)}s` : ""}
                </span>
              </div>
            </div>

            <Button variant="danger" size="sm" onClick={handleCancelJob}>
              Cancel Job
            </Button>
          </div>
        )}

        {/* Error or Cancelled state */}
        {errorMsg && (
          <div className="rounded-panel border border-status-danger/30 bg-status-danger-subtle/20 p-5 flex flex-col gap-3">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-status-danger flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h4 className="text-sm font-semibold text-status-danger">
                  {isCancelled ? "Processing Cancelled" : "Video Processing Error"}
                </h4>
                <p className="text-xs text-text-secondary mt-1">{errorMsg}</p>
              </div>
            </div>
            <div className="flex justify-end gap-2 pt-2">
              <Button variant="secondary" size="sm" onClick={handleReset}>
                Upload Another Video
              </Button>
            </div>
          </div>
        )}

        {/* Job Done State */}
        {job && job.state === "done" && (
          <div className="rounded-panel border border-border-subtle bg-card p-6 flex flex-col gap-5">
            <div className="flex items-center justify-between border-b border-border-subtle pb-4">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-control bg-status-success-subtle text-status-success border border-status-success/20">
                  <CheckCircle2 className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-text-primary">
                    Video Analysis Complete
                  </h3>
                  <p className="text-xs text-text-secondary">
                    Processed {job.frames_processed} frames across {job.duration_s.toFixed(1)}s video
                  </p>
                </div>
              </div>
              <Button variant="secondary" size="sm" onClick={handleReset} leftIcon={<RotateCcw className="w-3.5 h-3.5" />}>
                New Video
              </Button>
            </div>

            {/* Annotated video player */}
            <div className="rounded-control overflow-hidden bg-black border border-border-subtle">
              <video
                controls
                className="w-full max-h-[400px] object-contain"
                src={`${API_BASE_URL}/v1/video/${job.job_id}/annotated`}
              >
                Your browser does not support the video element.
              </video>
            </div>

            {/* Summary stats panel */}
            {summary && (
              <div className="flex flex-col gap-4">
                {/* KPI row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="rounded-control bg-elevated border border-border-subtle p-3 flex flex-col gap-0.5">
                    <span className="text-[10px] uppercase tracking-wide text-text-tertiary font-medium">Unique Objects</span>
                    <span className="text-xl font-bold text-text-primary font-mono">{summary.unique_counts}</span>
                  </div>
                  <div className="rounded-control bg-elevated border border-border-subtle p-3 flex flex-col gap-0.5">
                    <span className="text-[10px] uppercase tracking-wide text-text-tertiary font-medium flex items-center gap-1"><ArrowDownToLine className="w-3 h-3" />Line IN</span>
                    <span className="text-xl font-bold text-status-success font-mono">{summary.line_counts.total_in}</span>
                  </div>
                  <div className="rounded-control bg-elevated border border-border-subtle p-3 flex flex-col gap-0.5">
                    <span className="text-[10px] uppercase tracking-wide text-text-tertiary font-medium flex items-center gap-1"><ArrowUpFromLine className="w-3 h-3" />Line OUT</span>
                    <span className="text-xl font-bold text-brand font-mono">{summary.line_counts.total_out}</span>
                  </div>
                  <div className="rounded-control bg-elevated border border-border-subtle p-3 flex flex-col gap-0.5">
                    <span className="text-[10px] uppercase tracking-wide text-text-tertiary font-medium flex items-center gap-1"><TrendingUp className="w-3 h-3" />FPS Processed</span>
                    <span className="text-xl font-bold text-text-primary font-mono">{summary.fps_processed.toFixed(1)}</span>
                  </div>
                </div>

                {/* Counts per class bar chart */}
                {Object.keys(summary.counts_per_class).length > 0 && (
                  <div className="rounded-control bg-elevated border border-border-subtle p-4 flex flex-col gap-3">
                    <h4 className="text-xs font-semibold text-text-secondary flex items-center gap-1.5">
                      <BarChart2 className="w-3.5 h-3.5 text-brand" />
                      Unique Objects by Class
                    </h4>
                    <div className="flex flex-col gap-2">
                      {Object.entries(summary.counts_per_class)
                        .sort(([, a], [, b]) => b - a)
                        .map(([cls, cnt]) => {
                          const maxVal = Math.max(...Object.values(summary.counts_per_class));
                          const pct = maxVal > 0 ? (cnt / maxVal) * 100 : 0;
                          return (
                            <div key={cls} className="flex items-center gap-2">
                              <span className="text-[11px] text-text-secondary w-24 truncate shrink-0 capitalize">{cls}</span>
                              <div className="flex-1 h-2 rounded-full bg-elevated-2 overflow-hidden border border-border-subtle">
                                <div
                                  className="h-full bg-brand rounded-full transition-all duration-500"
                                  style={{ width: `${pct}%` }}
                                />
                              </div>
                              <span className="text-[11px] font-mono font-semibold text-text-primary w-6 text-right">{cnt}</span>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Results download action buttons */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              <a
                href={`${API_BASE_URL}/v1/video/${job.job_id}/annotated`}
                download={`annotated_${job.job_id}.mp4`}
                className="flex items-center justify-center gap-2 p-3 rounded-control border border-border-subtle bg-elevated hover:bg-subtle text-xs font-medium text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              >
                <Video className="w-4 h-4 text-brand" />
                <span>Annotated MP4</span>
                <Download className="w-3.5 h-3.5 text-text-tertiary ml-auto" />
              </a>

              <a
                href={`${API_BASE_URL}/v1/video/${job.job_id}/tracks`}
                download={`tracks_${job.job_id}.csv`}
                className="flex items-center justify-center gap-2 p-3 rounded-control border border-border-subtle bg-elevated hover:bg-subtle text-xs font-medium text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              >
                <FileText className="w-4 h-4 text-brand" />
                <span>Tracks CSV</span>
                <Download className="w-3.5 h-3.5 text-text-tertiary ml-auto" />
              </a>

              <a
                href={`${API_BASE_URL}/v1/video/${job.job_id}/summary`}
                download={`summary_${job.job_id}.json`}
                className="flex items-center justify-center gap-2 p-3 rounded-control border border-border-subtle bg-elevated hover:bg-subtle text-xs font-medium text-text-primary transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              >
                <Layers className="w-4 h-4 text-brand" />
                <span>Summary JSON</span>
                <Download className="w-3.5 h-3.5 text-text-tertiary ml-auto" />
              </a>
            </div>
          </div>
        )}
      </div>

      {/* Right / Configuration Inspector Panel */}
      <div className="w-full lg:w-80 flex-shrink-0 flex flex-col gap-4">
        <div className="rounded-panel border border-border-subtle bg-card p-5 flex flex-col gap-5">
          <div className="flex items-center justify-between border-b border-border-subtle pb-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5" />
              Video Settings
            </h3>
            <Badge variant="neutral">Config</Badge>
          </div>

          {/* Model Selector */}
          <div className="flex flex-col gap-1.5">
            <label className="text-xs font-medium text-text-secondary flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-brand" />
              Detection Model
            </label>
            <select
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              disabled={!!job && (job.state === "queued" || job.state === "running")}
              className="w-full text-xs rounded-control border border-border-subtle bg-elevated px-3 py-2 text-text-primary focus:outline-none focus:ring-2 focus:ring-brand"
            >
              {registeredModels.length > 0 ? (
                registeredModels.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name} ({m.version})
                  </option>
                ))
              ) : (
                <>
                  <option value="fine-tuned">SSD MobileNet V2 (Open Images 20-Class)</option>
                  <option value="coco-baseline">SSD MobileNet V2 (COCO 80-Class Baseline)</option>
                </>
              )}
            </select>
          </div>

          {/* Frame Stride Slider */}
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-text-secondary">Frame Stride</span>
              <span className="font-mono text-text-primary font-medium">Every {frameStride} frames</span>
            </div>
            <Slider
              min={1}
              max={10}
              step={1}
              value={frameStride}
              onValueChange={(val) => setFrameStride(val)}
              disabled={!!job && (job.state === "queued" || job.state === "running")}
            />
            <p className="text-[11px] text-text-tertiary">
              Higher stride processes faster but samples fewer video frames.
            </p>
          </div>

          {/* Confidence Threshold Slider */}
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-text-secondary">Confidence Filter</span>
              <span className="font-mono text-text-primary font-medium">{Math.round(confThreshold * 100)}%</span>
            </div>
            <Slider
              min={0.05}
              max={0.95}
              step={0.05}
              value={confThreshold}
              onValueChange={(val) => setConfThreshold(val)}
              disabled={!!job && (job.state === "queued" || job.state === "running")}
            />
          </div>

          {/* IoU NMS Threshold Slider */}
          <div className="flex flex-col gap-2">
            <div className="flex justify-between items-center text-xs">
              <span className="text-text-secondary">NMS IoU Threshold</span>
              <span className="font-mono text-text-primary font-medium">{Math.round(nmsThreshold * 100)}%</span>
            </div>
            <Slider
              min={0.1}
              max={0.9}
              step={0.05}
              value={nmsThreshold}
              onValueChange={(val) => setNmsThreshold(val)}
              disabled={!!job && (job.state === "queued" || job.state === "running")}
            />
          </div>

          {/* Counting Line Editor */}
          <div className="border-t border-border-subtle pt-4 flex flex-col gap-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-medium text-text-secondary flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={lineEnabled}
                  onChange={(e) => setLineEnabled(e.target.checked)}
                  disabled={!!job && (job.state === "queued" || job.state === "running")}
                  className="rounded border-border-subtle text-brand focus:ring-brand"
                />
                Counting Line
              </label>
              {lineEnabled && (
                <span className="text-[10px] text-brand font-medium">Active</span>
              )}
            </div>

            {lineEnabled && (
              <div className="flex flex-col gap-2">
                <p className="text-[11px] text-text-tertiary">
                  Drag the two blue handles to place a directional counting tripwire across the frame.
                </p>
                <div className="relative rounded-control overflow-hidden border border-border-subtle bg-elevated aspect-[4/3] flex items-center justify-center">
                  <canvas
                    ref={lineCanvasRef}
                    width={280}
                    height={210}
                    onMouseDown={handleCanvasMouseDown}
                    onMouseMove={handleCanvasMouseMove}
                    onMouseUp={handleCanvasMouseUp}
                    onMouseLeave={handleCanvasMouseUp}
                    className="w-full h-full cursor-crosshair"
                  />
                  <div className="absolute bottom-1 right-2 text-[10px] font-mono text-text-tertiary pointer-events-none">
                    P1: ({lineP1.x.toFixed(2)}, {lineP1.y.toFixed(2)}) | P2: ({lineP2.x.toFixed(2)}, {lineP2.y.toFixed(2)})
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
