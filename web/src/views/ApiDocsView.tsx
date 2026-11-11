import React, { useState, useEffect } from "react";
import { Terminal, Copy, Check, Activity } from "lucide-react";
import { checkHealth } from "../api";

export const ApiDocsView: React.FC = () => {
  const [activeSnippetTab, setActiveSnippetTab] = useState<"curl" | "js" | "python">("curl");
  const [copied, setCopied] = useState<boolean>(false);
  const [healthStatus, setHealthStatus] = useState<{ status: string; version: string } | null>(
    null
  );

  useEffect(() => {
    checkHealth()
      .then((res) => setHealthStatus({ status: res.status, version: res.version }))
      .catch(() => setHealthStatus({ status: "offline", version: "unknown" }));
  }, []);

  const snippets = {
    curl: `curl -X POST "http://localhost:8000/v1/detect?confidence_threshold=0.25&nms_iou_threshold=0.45" \\
  -H "Accept: application/json" \\
  -F "file=@/path/to/image.jpg"`,
    js: `const formData = new FormData();
formData.append("file", imageFile);

const response = await fetch("http://localhost:8000/v1/detect?confidence_threshold=0.25", {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(result.detections);`,
    python: `import requests

url = "http://localhost:8000/v1/detect"
params = {"confidence_threshold": 0.25, "nms_iou_threshold": 0.45}

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, params=params, files=files)

data = response.json()
print(f"Total detected: {data['total_detections']}")
for det in data["detections"]:
    print(f"- {det['class_name']}: {det['confidence']*100:.1f}%")`,
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(snippets[activeSnippetTab]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col gap-8 max-w-4xl">
      {/* Header with Health Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-xl font-bold text-slate-100">REST API Documentation & Console</h2>
          <p className="text-sm text-slate-400 mt-1">
            FastAPI OpenAPI 3.1 specification for programmatic object detection.
          </p>
        </div>

        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 w-fit">
          <Activity
            className={`w-4 h-4 ${
              healthStatus?.status === "ok" ? "text-emerald-400 animate-pulse" : "text-amber-400"
            }`}
          />
          <span className="text-xs font-mono font-medium text-slate-300">
            API {healthStatus?.status === "ok" ? "Online" : "Connecting..."} (
            {healthStatus?.version || "v1.0.0"})
          </span>
        </div>
      </div>

      {/* Code Snippets Box */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden flex flex-col">
        <div className="flex items-center justify-between bg-slate-950 px-4 py-3 border-b border-slate-800">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-sky-400" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              POST /v1/detect Quickstart
            </span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex items-center bg-slate-900 rounded-md p-1 border border-slate-800 text-xs">
              <button
                type="button"
                onClick={() => setActiveSnippetTab("curl")}
                className={`px-2.5 py-1 rounded transition ${
                  activeSnippetTab === "curl"
                    ? "bg-sky-600 text-white font-medium"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                cURL
              </button>
              <button
                type="button"
                onClick={() => setActiveSnippetTab("js")}
                className={`px-2.5 py-1 rounded transition ${
                  activeSnippetTab === "js"
                    ? "bg-sky-600 text-white font-medium"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                JavaScript
              </button>
              <button
                type="button"
                onClick={() => setActiveSnippetTab("python")}
                className={`px-2.5 py-1 rounded transition ${
                  activeSnippetTab === "python"
                    ? "bg-sky-600 text-white font-medium"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Python
              </button>
            </div>

            <button
              type="button"
              onClick={copyToClipboard}
              className="p-1.5 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 transition"
              title="Copy snippet"
            >
              {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <pre className="p-4 text-xs font-mono text-slate-200 bg-slate-950/70 overflow-x-auto leading-relaxed">
          <code>{snippets[activeSnippetTab]}</code>
        </pre>
      </div>

      {/* Endpoints Table */}
      <div className="flex flex-col gap-4">
        <h3 className="text-base font-bold text-slate-200">Endpoint Reference</h3>

        <div className="flex flex-col gap-3">
          {/* /v1/detect */}
          <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-sky-500/20 text-sky-400 border border-sky-500/30">
                POST
              </span>
              <span className="font-mono text-sm font-semibold text-slate-200">/v1/detect</span>
            </div>
            <p className="text-xs text-slate-400">
              Run object detection on multipart image payload. Supports query parameters{" "}
              <code className="text-sky-300">confidence_threshold</code> (default 0.25) and{" "}
              <code className="text-sky-300">nms_iou_threshold</code> (default 0.45).
            </p>
          </div>

          {/* /v1/classes */}
          <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                GET
              </span>
              <span className="font-mono text-sm font-semibold text-slate-200">/v1/classes</span>
            </div>
            <p className="text-xs text-slate-400">
              Returns the full 20-class target taxonomy with categorical color codes and
              descriptions.
            </p>
          </div>

          {/* /v1/model */}
          <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                GET
              </span>
              <span className="font-mono text-sm font-semibold text-slate-200">/v1/model</span>
            </div>
            <p className="text-xs text-slate-400">
              Returns model specification, input dimensions (300×300), and benchmark metrics
              (mAP@0.5, latency SLAs).
            </p>
          </div>

          {/* /healthz */}
          <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded text-xs font-bold font-mono bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                GET
              </span>
              <span className="font-mono text-sm font-semibold text-slate-200">/healthz</span>
            </div>
            <p className="text-xs text-slate-400">
              Readiness and liveness probe returning API version and model load state.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
