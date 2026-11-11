import { useState } from "react";
import { Scan, History, Cpu, Terminal, Eye } from "lucide-react";
import { DetectView } from "./views/DetectView";
import { HistoryView } from "./views/HistoryView";
import { ModelView } from "./views/ModelView";
import { ApiDocsView } from "./views/ApiDocsView";
import type { HistoryRecord } from "./types";

export function App() {
  const [activeTab, setActiveTab] = useState<"detect" | "history" | "model" | "api">("detect");
  const [history, setHistory] = useState<HistoryRecord[]>([]);

  const handleSaveHistory = (record: HistoryRecord) => {
    setHistory((prev) => [record, ...prev].slice(0, 30));
  };

  const handleSelectHistoryRecord = (record: HistoryRecord) => {
    // Switch to detect tab
    setActiveTab("detect");
    // History can be viewed in history view or re-inspected
    console.log("Selected record:", record.id);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* Navigation Bar */}
      <header className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-950">
              <Eye className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-slate-100 tracking-tight">VisionSSD</h1>
                <span className="text-[10px] font-mono font-bold px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30">
                  v1.0.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 -mt-0.5 hidden sm:block">
                SSD MobileNet V2 · 20-Class Open Images Detector
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              type="button"
              onClick={() => setActiveTab("detect")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition ${
                activeTab === "detect"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Scan className="w-4 h-4" />
              <span>Detect</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("history")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition relative ${
                activeTab === "history"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <History className="w-4 h-4" />
              <span>History</span>
              {history.length > 0 && (
                <span className="ml-1 text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-slate-800 text-sky-300">
                  {history.length}
                </span>
              )}
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("model")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition ${
                activeTab === "model"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Cpu className="w-4 h-4" />
              <span className="hidden sm:inline">Model & Metrics</span>
              <span className="sm:hidden">Model</span>
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("api")}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-sm font-medium transition ${
                activeTab === "api"
                  ? "bg-sky-600 text-white shadow-sm"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/60"
              }`}
            >
              <Terminal className="w-4 h-4" />
              <span className="hidden sm:inline">API Docs</span>
              <span className="sm:hidden">API</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow max-w-7xl mx-auto w-full px-4 sm:px-6 py-8">
        {activeTab === "detect" && <DetectView onSaveHistory={handleSaveHistory} />}
        {activeTab === "history" && (
          <HistoryView
            history={history}
            onSelectRecord={handleSelectHistoryRecord}
            onClearHistory={handleClearHistory}
          />
        )}
        {activeTab === "model" && <ModelView />}
        {activeTab === "api" && <ApiDocsView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 py-6 text-center text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 VisionSSD System · Fine-Tuned PyTorch SSD MobileNet V2</p>
          <div className="flex items-center gap-4 text-slate-400">
            <span>FastAPI Backend :8000</span>
            <span>·</span>
            <span>React + Vite Frontend :5173</span>
            <span>·</span>
            <span>mAP@0.5: 26.44%</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
