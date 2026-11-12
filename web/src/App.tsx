import { useState, useEffect } from "react";
import { Scan, History, Cpu, Terminal, Eye, Palette } from "lucide-react";
import { DetectView } from "./views/DetectView";
import { HistoryView } from "./views/HistoryView";
import { ModelView } from "./views/ModelView";
import { ApiDocsView } from "./views/ApiDocsView";
import { DevComponentsView } from "./views/DevComponentsView";
import { ThemeToggle } from "./components/ui/ThemeToggle";
import type { HistoryRecord } from "./types";

export function App() {
  const [activeTab, setActiveTab] = useState<
    "detect" | "history" | "model" | "api" | "dev"
  >(() => {
    if (typeof window !== "undefined") {
      if (
        window.location.pathname === "/dev/components" ||
        window.location.pathname.startsWith("/dev") ||
        window.location.hash === "#/dev/components"
      ) {
        return "dev";
      }
    }
    return "detect";
  });

  const [history, setHistory] = useState<HistoryRecord[]>([]);

  useEffect(() => {
    const handlePopState = () => {
      if (window.location.pathname === "/dev/components") {
        setActiveTab("dev");
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const handleTabChange = (tab: "detect" | "history" | "model" | "api" | "dev") => {
    setActiveTab(tab);
    if (tab === "dev") {
      window.history.pushState(null, "", "/dev/components");
    } else if (window.location.pathname === "/dev/components") {
      window.history.pushState(null, "", "/");
    }
  };

  const handleSaveHistory = (record: HistoryRecord) => {
    setHistory((prev) => [record, ...prev].slice(0, 30));
  };

  const handleSelectHistoryRecord = (record: HistoryRecord) => {
    setActiveTab("detect");
    console.log("Selected record:", record.id);
  };

  const handleClearHistory = () => {
    setHistory([]);
  };

  return (
    <div className="min-h-screen bg-canvas text-text-primary flex flex-col font-sans transition-colors duration-150">
      {/* Navigation Bar */}
      <header className="sticky top-0 z-50 bg-overlay backdrop-blur-md border-b border-border-subtle">
        <div className="max-w-container mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          {/* Logo and Title */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-control bg-elevated border border-border-subtle flex items-center justify-center text-text-primary">
              <Eye className="w-4 h-4" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold tracking-tight text-text-primary">
                  VisionSSD
                </span>
                <span className="text-[10px] font-mono font-medium px-1.5 py-0.2 rounded-chip bg-elevated text-text-secondary border border-border-subtle">
                  v1.0.0
                </span>
              </div>
              <p className="text-[11px] text-text-tertiary -mt-0.5 hidden sm:block">
                SSD MobileNet V2 · 20-Class Open Images Detector
              </p>
            </div>
          </div>

          {/* Navigation Tabs & Actions */}
          <div className="flex items-center gap-2 sm:gap-3">
            <nav className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => handleTabChange("detect")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium transition ${
                  activeTab === "detect"
                    ? "bg-elevated text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary hover:bg-subtle"
                }`}
              >
                <Scan className="w-3.5 h-3.5" />
                <span>Detect</span>
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("history")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium transition relative ${
                  activeTab === "history"
                    ? "bg-elevated text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary hover:bg-subtle"
                }`}
              >
                <History className="w-3.5 h-3.5" />
                <span>History</span>
                {history.length > 0 && (
                  <span className="ml-1 text-[10px] font-mono px-1 py-0.2 rounded-chip bg-subtle text-text-secondary border border-border-subtle">
                    {history.length}
                  </span>
                )}
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("model")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium transition ${
                  activeTab === "model"
                    ? "bg-elevated text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary hover:bg-subtle"
                }`}
              >
                <Cpu className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">Model & Metrics</span>
                <span className="sm:hidden">Model</span>
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("api")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-control text-xs font-medium transition ${
                  activeTab === "api"
                    ? "bg-elevated text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary hover:bg-subtle"
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">API Docs</span>
                <span className="sm:hidden">API</span>
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("dev")}
                title="Design System Components (/dev/components)"
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-control text-xs font-medium transition ${
                  activeTab === "dev"
                    ? "bg-brand-subtle text-brand border border-brand/30 shadow-sm"
                    : "text-text-tertiary hover:text-text-primary hover:bg-subtle"
                }`}
              >
                <Palette className="w-3.5 h-3.5" />
                <span className="hidden md:inline">Components</span>
              </button>
            </nav>

            <div className="h-4 w-px bg-border-subtle hidden sm:block" />

            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow w-full">
        {activeTab === "detect" && (
          <div className="max-w-container mx-auto px-4 sm:px-6 py-6">
            <DetectView onSaveHistory={handleSaveHistory} />
          </div>
        )}
        {activeTab === "history" && (
          <div className="max-w-container mx-auto px-4 sm:px-6 py-6">
            <HistoryView
              history={history}
              onSelectRecord={handleSelectHistoryRecord}
              onClearHistory={handleClearHistory}
            />
          </div>
        )}
        {activeTab === "model" && (
          <div className="max-w-container mx-auto px-4 sm:px-6 py-6">
            <ModelView />
          </div>
        )}
        {activeTab === "api" && (
          <div className="max-w-container mx-auto px-4 sm:px-6 py-6">
            <ApiDocsView />
          </div>
        )}
        {activeTab === "dev" && <DevComponentsView />}
      </main>

      {/* Footer */}
      <footer className="border-t border-border-subtle bg-subtle py-5 text-center text-xs text-text-tertiary mt-auto">
        <div className="max-w-container mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <p>© 2026 VisionSSD System · Fine-Tuned PyTorch SSD MobileNet V2</p>
          <div className="flex items-center gap-4 text-text-secondary font-mono text-[11px]">
            <span>FastAPI Backend</span>
            <span>·</span>
            <span>React + Vite</span>
            <span>·</span>
            <span className="tabular-nums">mAP@0.5: 26.44%</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
