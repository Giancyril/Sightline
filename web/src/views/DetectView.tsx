import React, { useState, useRef } from "react";
import { Upload, Sliders, Zap, CheckCircle2, AlertCircle } from "lucide-react";
import { DetectionCanvas } from "../components/DetectionCanvas";
import { WebcamCapture } from "../components/WebcamCapture";
import { detectImage } from "../api";
import type { DetectionResponse, HistoryRecord } from "../types";

interface DetectViewProps {
  onSaveHistory: (record: HistoryRecord) => void;
}

export const DetectView: React.FC<DetectViewProps> = ({ onSaveHistory }) => {
  const [mode, setMode] = useState<"upload" | "webcam">("upload");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDetecting, setIsDetecting] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [selectedDetectionIdx, setSelectedDetectionIdx] = useState<number | null>(null);

  // Settings
  const [confThreshold, setConfThreshold] = useState<number>(0.25);
  const [nmsThreshold, setNmsThreshold] = useState<number>(0.45);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const processDetection = async (blob: Blob) => {
    try {
      setIsDetecting(true);
      setErrorMsg(null);
      setSelectedDetectionIdx(null);

      // Create preview URL
      const previewUrl = URL.createObjectURL(blob);
      setImagePreview(previewUrl);

      const res = await detectImage(blob, confThreshold, nmsThreshold);
      setResult(res);

      // Save to history
      onSaveHistory({
        id: Math.random().toString(36).substring(2, 9),
        timestamp: new Date().toLocaleTimeString(),
        previewUrl,
        result: res,
      });
    } catch (err) {
      setErrorMsg((err as Error).message);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processDetection(file);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      processDetection(file);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Column: Input and Canvas */}
      <div className="lg:col-span-8 flex flex-col gap-5">
        {/* Mode Selector */}
        <div className="flex items-center gap-2 bg-slate-900 p-1.5 rounded-xl border border-slate-800 w-fit">
          <button
            type="button"
            onClick={() => setMode("upload")}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
              mode === "upload"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Upload Image
          </button>
          <button
            type="button"
            onClick={() => setMode("webcam")}
            className={`px-4 py-1.5 rounded-lg text-sm font-medium transition ${
              mode === "webcam"
                ? "bg-sky-600 text-white shadow-sm"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            Live Webcam
          </button>
        </div>

        {/* Upload Zone / Webcam Box */}
        {mode === "upload" ? (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-sky-500 rounded-xl p-8 bg-slate-900/60 hover:bg-slate-900 text-center cursor-pointer transition flex flex-col items-center justify-center gap-3"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileUpload}
              className="hidden"
            />
            <div className="w-12 h-12 rounded-full bg-sky-500/10 flex items-center justify-center text-sky-400">
              <Upload className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">
                Click to browse or drop an image
              </p>
              <p className="text-xs text-slate-400 mt-1">
                Supports JPEG, PNG, WebP (up to 10MB)
              </p>
            </div>
          </div>
        ) : (
          <WebcamCapture onCapture={processDetection} isProcessing={isDetecting} />
        )}

        {errorMsg && (
          <div className="p-4 rounded-xl bg-red-950/40 border border-red-800/80 flex items-center gap-3 text-red-300 text-sm">
            <AlertCircle className="w-5 h-5 flex-shrink-0 text-red-400" />
            <span>{errorMsg}</span>
          </div>
        )}

        {/* Canvas Display */}
        {imagePreview && (
          <div className="flex flex-col gap-3">
            <div className="flex items-center justify-between text-xs text-slate-400">
              <span>Interactive Detection Overlay (Click box to inspect)</span>
              {result && (
                <div className="flex items-center gap-3">
                  <span className="flex items-center gap-1 text-emerald-400 font-medium">
                    <CheckCircle2 className="w-4 h-4" />
                    {result.total_detections} detected
                  </span>
                  <span className="flex items-center gap-1 text-sky-400 font-mono">
                    <Zap className="w-4 h-4" />
                    {result.inference_time_ms} ms
                  </span>
                </div>
              )}
            </div>
            <DetectionCanvas
              imageSrc={imagePreview}
              detections={result?.detections || []}
              selectedIndex={selectedDetectionIdx}
              onSelectDetection={setSelectedDetectionIdx}
            />
          </div>
        )}
      </div>

      {/* Right Column: Controls & Detections List */}
      <div className="lg:col-span-4 flex flex-col gap-5">
        {/* Parameter Sliders */}
        <div className="p-5 bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col gap-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200 border-b border-slate-800 pb-3">
            <Sliders className="w-4 h-4 text-sky-400" />
            <span>Inference Settings</span>
          </div>

          <div>
            <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
              <span>Confidence Threshold</span>
              <span className="font-mono text-sky-400">{(confThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.05"
              value={confThreshold}
              onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
              className="w-full accent-sky-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Minimum probability required to retain a detection.
            </p>
          </div>

          <div>
            <div className="flex justify-between text-xs font-medium text-slate-300 mb-1.5">
              <span>NMS IoU Threshold</span>
              <span className="font-mono text-sky-400">{(nmsThreshold * 100).toFixed(0)}%</span>
            </div>
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={nmsThreshold}
              onChange={(e) => setNmsThreshold(parseFloat(e.target.value))}
              className="w-full accent-sky-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Intersection-over-Union threshold for deduplicating boxes.
            </p>
          </div>
        </div>

        {/* Detections List */}
        <div className="p-5 bg-slate-900/90 rounded-xl border border-slate-800 flex flex-col gap-3 min-h-[300px]">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200">
              Detected Objects
            </h3>
            <span className="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 font-mono">
              {result?.detections.length || 0}
            </span>
          </div>

          {result && result.detections.length > 0 ? (
            <div className="flex flex-col gap-2 overflow-y-auto max-h-[460px] pr-1">
              {result.detections.map((det, idx) => {
                const isSelected = selectedDetectionIdx === idx;
                return (
                  <div
                    key={idx}
                    onClick={() => setSelectedDetectionIdx(isSelected ? null : idx)}
                    className={`p-3 rounded-lg border text-left cursor-pointer transition flex flex-col gap-2 ${
                      isSelected
                        ? "bg-slate-800 border-sky-500 shadow-sm"
                        : "bg-slate-950/60 border-slate-800/80 hover:border-slate-700"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span
                          className="w-3 h-3 rounded-full flex-shrink-0"
                          style={{ backgroundColor: det.color_hex }}
                        />
                        <span className="text-sm font-semibold text-slate-200">
                          {det.class_name}
                        </span>
                      </div>
                      <span className="text-xs font-mono font-bold text-sky-400">
                        {(det.confidence * 100).toFixed(1)}%
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-400 font-mono bg-slate-900/60 p-2 rounded">
                      <div>
                        Px: [{det.box_pixel.xmin}, {det.box_pixel.ymin}]
                      </div>
                      <div>
                        Size: {det.box_pixel.width}×{det.box_pixel.height}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-400 text-sm">
              {isDetecting ? "Running detector model..." : "No objects detected yet."}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

