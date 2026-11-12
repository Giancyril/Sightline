import React, { useState, useRef } from "react";
import {
  Play,
  Upload,
  Copy,
  Check,
  Sparkles,
  Loader2,
  Clock,
  FileText,
  AlertCircle,
} from "lucide-react";
import { Slider } from "../ui/Slider";
import { Button } from "../ui/Button";
import { Badge } from "../ui/Badge";
import { detectImage, fetchClasses, fetchModelInfo, checkHealth } from "../../api";
import { cn } from "../../lib/utils";

interface SampleImageOption {
  id: string;
  name: string;
  src: string;
}

const SAMPLE_OPTIONS: SampleImageOption[] = [
  { id: "workspace", name: "Office Workspace", src: "/samples/sample-workspace.jpg" },
  { id: "desk", name: "Study Desk", src: "/samples/sample-desk.jpg" },
  { id: "street", name: "Urban Street", src: "/samples/sample-street.jpg" },
];

export const TryItConsole: React.FC = () => {
  const [activeEndpoint, setActiveEndpoint] = useState<"detect" | "classes" | "model" | "health">(
    "detect"
  );
  const [selectedSample, setSelectedSample] = useState<string>("/samples/sample-workspace.jpg");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("/samples/sample-workspace.jpg");
  const [confidence, setConfidence] = useState<number>(0.25);
  const [nmsIou, setNmsIou] = useState<number>(0.45);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [status, setStatus] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [responseJson, setResponseJson] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copied, setCopied] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSampleChange = (src: string) => {
    setSelectedSample(src);
    setUploadedFile(null);
    setPreviewUrl(src);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      setSelectedSample("");
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const executeRequest = async () => {
    setIsLoading(true);
    setStatus(null);
    setErrorMsg(null);
    setResponseJson(null);
    const start = performance.now();

    try {
      if (activeEndpoint === "detect") {
        let blobToSend: Blob;
        if (uploadedFile) {
          blobToSend = uploadedFile;
        } else {
          // Fetch the bundled sample image as a blob
          const res = await fetch(selectedSample);
          if (!res.ok) throw new Error("Failed to load sample image file");
          blobToSend = await res.blob();
        }

        const data = await detectImage(blobToSend, confidence, nmsIou);
        const duration = Math.round(performance.now() - start);
        setStatus("200 OK");
        setLatency(duration);
        setResponseJson(JSON.stringify(data, null, 2));
      } else if (activeEndpoint === "classes") {
        const data = await fetchClasses();
        const duration = Math.round(performance.now() - start);
        setStatus("200 OK");
        setLatency(duration);
        setResponseJson(JSON.stringify(data, null, 2));
      } else if (activeEndpoint === "model") {
        const data = await fetchModelInfo();
        const duration = Math.round(performance.now() - start);
        setStatus("200 OK");
        setLatency(duration);
        setResponseJson(JSON.stringify(data, null, 2));
      } else if (activeEndpoint === "health") {
        const data = await checkHealth();
        const duration = Math.round(performance.now() - start);
        setStatus("200 OK");
        setLatency(duration);
        setResponseJson(JSON.stringify(data, null, 2));
      }
    } catch (err) {
      const duration = Math.round(performance.now() - start);
      setLatency(duration);
      const msg = err instanceof Error ? err.message : "Inference request failed";
      setStatus("Error");
      setErrorMsg(msg);
      setResponseJson(
        JSON.stringify(
          {
            error: true,
            status: "request_failed",
            message: msg,
            hint: "Verify that the FastAPI inference service is running and reachable.",
          },
          null,
          2
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const copyResponse = () => {
    if (responseJson) {
      void navigator.clipboard.writeText(responseJson);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div className="rounded-panel border border-border-subtle bg-card overflow-hidden flex flex-col shadow-sm">
      {/* Console Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 bg-subtle px-4 py-3 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-brand" />
          <h4 className="text-sm font-semibold text-text-primary">
            Interactive API Console
          </h4>
          <span className="text-xs text-text-muted hidden sm:inline">&middot; Live Execution</span>
        </div>

        {/* Endpoint Selector Tabs */}
        <div className="flex items-center rounded-control bg-elevated p-0.5 border border-border-subtle text-xs">
          <button
            type="button"
            onClick={() => setActiveEndpoint("detect")}
            className={cn(
              "px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium transition-colors",
              activeEndpoint === "detect"
                ? "bg-card text-brand font-semibold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            )}
          >
            POST /v1/detect
          </button>
          <button
            type="button"
            onClick={() => setActiveEndpoint("classes")}
            className={cn(
              "px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium transition-colors",
              activeEndpoint === "classes"
                ? "bg-card text-brand font-semibold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            )}
          >
            GET /v1/classes
          </button>
          <button
            type="button"
            onClick={() => setActiveEndpoint("model")}
            className={cn(
              "px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium transition-colors",
              activeEndpoint === "model"
                ? "bg-card text-brand font-semibold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            )}
          >
            GET /v1/model
          </button>
          <button
            type="button"
            onClick={() => setActiveEndpoint("health")}
            className={cn(
              "px-2.5 py-1 rounded-[4px] font-mono text-[11px] font-medium transition-colors",
              activeEndpoint === "health"
                ? "bg-card text-brand font-semibold shadow-sm"
                : "text-text-secondary hover:text-text-primary"
            )}
          >
            GET /healthz
          </button>
        </div>
      </div>

      {/* Main Console Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 divide-y lg:divide-y-0 lg:divide-x divide-border-subtle">
        {/* Left Column: Request Inputs */}
        <div className="lg:col-span-5 p-4 sm:p-5 flex flex-col justify-between gap-5 bg-card">
          <div className="flex flex-col gap-4">
            {activeEndpoint === "detect" ? (
              <>
                {/* Image Selection */}
                <div className="flex flex-col gap-2">
                  <label className="text-xs font-semibold text-text-primary flex items-center justify-between">
                    <span>Input Image Payload</span>
                    <span className="text-[11px] font-normal text-text-muted">JPEG, PNG, WebP</span>
                  </label>

                  {/* Preset Samples */}
                  <div className="grid grid-cols-3 gap-1.5">
                    {SAMPLE_OPTIONS.map((sample) => (
                      <button
                        key={sample.id}
                        type="button"
                        onClick={() => handleSampleChange(sample.src)}
                        className={cn(
                          "flex flex-col items-center gap-1 p-1.5 rounded-control border text-left transition-colors",
                          selectedSample === sample.src && !uploadedFile
                            ? "border-brand bg-brand-subtle/40 text-text-primary"
                            : "border-border-subtle bg-subtle hover:bg-elevated text-text-secondary"
                        )}
                      >
                        <img
                          src={sample.src}
                          alt={sample.name}
                          className="w-full h-12 object-cover rounded-[3px]"
                        />
                        <span className="text-[10px] font-medium truncate w-full text-center">
                          {sample.name}
                        </span>
                      </button>
                    ))}
                  </div>

                  {/* Or Custom Upload */}
                  <div className="mt-1">
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/jpeg,image/png,image/webp"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                    <button
                      type="button"
                      onClick={() => fileInputRef.current?.click()}
                      className={cn(
                        "w-full flex items-center justify-center gap-2 py-2 px-3 border border-dashed rounded-control text-xs transition-colors",
                        uploadedFile
                          ? "border-brand bg-brand-subtle/20 text-brand font-medium"
                          : "border-border-subtle bg-subtle hover:bg-elevated text-text-secondary"
                      )}
                    >
                      <Upload className="w-3.5 h-3.5" />
                      <span>{uploadedFile ? uploadedFile.name : "Upload Custom Image…"}</span>
                    </button>
                  </div>

                  {/* Active Preview Strip */}
                  {previewUrl && (
                    <div className="flex items-center gap-2.5 p-2 rounded-control bg-subtle border border-border-subtle text-[11px] text-text-muted">
                      <img
                        src={previewUrl}
                        alt="Preview"
                        className="w-8 h-8 object-cover rounded-sm border border-border-subtle shrink-0"
                      />
                      <div className="truncate">
                        <span className="text-text-primary font-medium block truncate">
                          {uploadedFile ? uploadedFile.name : "Sample: " + selectedSample.split("/").pop()}
                        </span>
                        <span>{uploadedFile ? `${(uploadedFile.size / 1024).toFixed(1)} KB` : "Bundled fixture"}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Threshold Sliders */}
                <div className="flex flex-col gap-3 pt-2 border-t border-border-subtle">
                  <Slider
                    label="confidence_threshold"
                    value={confidence}
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    onValueChange={setConfidence}
                    formatValue={(v) => v.toFixed(2)}
                  />
                  <Slider
                    label="nms_iou_threshold"
                    value={nmsIou}
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    onValueChange={setNmsIou}
                    formatValue={(v) => v.toFixed(2)}
                  />
                </div>
              </>
            ) : (
              <div className="p-4 rounded-control bg-subtle border border-border-subtle text-xs text-text-secondary space-y-2">
                <div className="font-semibold text-text-primary">No Body Parameters Required</div>
                <p className="leading-relaxed text-[11px]">
                  This endpoint accepts a standard HTTP GET request with no query parameters or payload body. Click &quot;Send Request&quot; below to execute a real-time call against the active backend.
                </p>
              </div>
            )}
          </div>

          {/* Action Trigger */}
          <Button
            variant="primary"
            size="md"
            onClick={executeRequest}
            disabled={isLoading}
            className="w-full gap-2 shadow-sm"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Executing Request…</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Send Request</span>
              </>
            )}
          </Button>
        </div>

        {/* Right Column: Response Output */}
        <div className="lg:col-span-7 flex flex-col bg-canvas">
          {/* Response Telemetry Bar */}
          <div className="flex items-center justify-between px-4 py-2.5 bg-subtle/80 border-b border-border-subtle text-xs">
            <div className="flex items-center gap-3">
              <span className="text-text-muted font-medium">Response:</span>
              {status ? (
                <Badge
                  variant={status.startsWith("2") ? "success" : "danger"}
                  mono
                  className="text-[10px]"
                >
                  {status}
                </Badge>
              ) : (
                <span className="text-text-muted font-mono text-[11px]">Awaiting trigger</span>
              )}

              {latency !== null && (
                <div className="flex items-center gap-1 text-[11px] font-mono text-text-secondary">
                  <Clock className="w-3 h-3 text-text-tertiary" />
                  <span>{latency} ms</span>
                </div>
              )}
            </div>

            {responseJson && (
              <button
                type="button"
                onClick={copyResponse}
                className="flex items-center gap-1 text-xs text-text-secondary hover:text-text-primary font-medium transition-colors"
                title="Copy JSON response"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5 text-status-success" />
                    <span className="text-status-success font-mono">Copied</span>
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5 text-text-tertiary" />
                    <span>Copy JSON</span>
                  </>
                )}
              </button>
            )}
          </div>

          {/* Response Payload Screen */}
          <div className="flex-grow p-4 min-h-[300px] max-h-[460px] overflow-auto">
            {errorMsg && (
              <div className="mb-3 p-2.5 rounded-control bg-status-danger-subtle border border-status-danger/30 text-status-danger text-xs flex items-center gap-2">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}
            {isLoading ? (
              <div className="h-full min-h-[260px] flex flex-col items-center justify-center gap-3 text-text-muted">
                <Loader2 className="w-6 h-6 animate-spin text-brand" />
                <span className="text-xs font-mono">Sending request to inference service…</span>
              </div>
            ) : responseJson ? (
              <pre className="font-mono text-xs text-text-primary leading-relaxed whitespace-pre select-text">
                <code>{responseJson}</code>
              </pre>
            ) : (
              <div className="h-full min-h-[260px] flex flex-col items-center justify-center gap-2 text-text-muted text-center px-4">
                <FileText className="w-8 h-8 opacity-40 mb-1" />
                <div className="text-xs font-medium text-text-secondary">Ready to test</div>
                <p className="text-[11px] max-w-xs">
                  Choose a sample image and click &quot;Send Request&quot; to inspect live JSON responses directly in this console.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
