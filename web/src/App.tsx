import { useState, useEffect } from "react";
import { Search } from "lucide-react";
import { DetectView } from "./views/DetectView";
import { CompareView } from "./views/CompareView";
import { HistoryView } from "./views/HistoryView";
import { ModelView } from "./views/ModelView";
import { ApiDocsView } from "./views/ApiDocsView";
import { DevComponentsView } from "./views/DevComponentsView";
import { ThemeToggle } from "./components/ui/ThemeToggle";
import { ApiHealthDot } from "./components/shell/ApiHealthDot";
import { CommandPalette, type NavTab } from "./components/shell/CommandPalette";
import { MobileNav } from "./components/shell/MobileNav";
import { PageContainer, PageHeader } from "./components/shell/PageContainer";
import { useApiHealth } from "./hooks/useApiHealth";
import {
  getStoredHistory,
  saveHistoryItem,
  deleteHistoryItem,
  clearAllHistory,
  createLightweightThumbnail,
} from "./lib/historyStorage";
import { cn } from "./lib/utils";
import type { HistoryRecord } from "./types";

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export function App() {
  const [activeTab, setActiveTab] = useState<NavTab>(() => {
    if (typeof window !== "undefined") {
      const urlTab = new URLSearchParams(window.location.search).get("tab") as NavTab | null;
      if (urlTab && ["detect", "compare", "history", "model", "api", "dev"].includes(urlTab)) {
        return urlTab;
      }
      if (
        window.location.pathname === "/dev/components" ||
        window.location.pathname.startsWith("/dev") ||
        window.location.hash === "#/dev/components"
      ) {
        return "dev";
      }
      if (window.location.hash === "#/history" || window.location.pathname === "/history") {
        return "history";
      }
    }
    return "detect";
  });

  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [activeDetectionRecord, setActiveDetectionRecord] = useState<HistoryRecord | null>(null);
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
  const apiHealth = useApiHealth();

  // Load persistent history from IndexedDB on startup
  useEffect(() => {
    let mounted = true;
    void getStoredHistory().then((stored) => {
      if (mounted && stored.length > 0) {
        setHistory(stored);
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const handlePopState = () => {
      if (window.location.pathname === "/dev/components") {
        setActiveTab("dev");
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  // Global keyboard shortcut for Cmd/Ctrl+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandPaletteOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const handleTabChange = (tab: NavTab) => {
    setActiveTab(tab);
    if (tab === "dev") {
      window.history.pushState(null, "", "/dev/components");
    } else if (window.location.pathname === "/dev/components") {
      window.history.pushState(null, "", "/");
    }
  };

  const handleSaveHistory = async (record: HistoryRecord) => {
    const createdAt = record.createdAt || Date.now();
    let thumbnail = record.thumbnail || record.previewUrl;

    try {
      thumbnail = await createLightweightThumbnail(record.previewUrl, record.result);
    } catch {
      // Fallback to original previewUrl if canvas render fails
    }

    const itemToStore = {
      ...record,
      createdAt,
      thumbnail,
    };

    setHistory((prev) => [itemToStore, ...prev.filter((i) => i.id !== itemToStore.id)].slice(0, 200));
    void saveHistoryItem({
      id: record.id,
      createdAt,
      timestamp: record.timestamp,
      thumbnail,
      previewUrl: record.previewUrl,
      result: record.result,
    });
  };

  const handleSelectHistoryRecord = (record: HistoryRecord) => {
    setActiveDetectionRecord(record);
    setActiveTab("detect");
  };

  const handleDeleteRecord = (id: string) => {
    setHistory((prev) => prev.filter((i) => i.id !== id));
    void deleteHistoryItem(id);
  };

  const handleClearHistory = () => {
    setHistory([]);
    void clearAllHistory();
  };

  return (
    <div className="min-h-screen bg-canvas text-text-primary flex flex-col font-sans transition-colors duration-150">
      {/* Accessible skip link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:px-3 focus:py-1.5 focus:rounded-control focus:bg-card focus:text-brand focus:border focus:border-brand focus:text-xs focus:font-medium focus:shadow-md"
      >
        Skip to content
      </a>

      {/* 48px Top Bar */}
      <header className="sticky top-0 z-40 h-12 bg-overlay backdrop-blur-md border-b border-border-subtle">
        <div className="max-w-container mx-auto px-6 h-full flex items-center justify-between gap-4">
          {/* Left: Type-based wordmark + quiet version badge */}
          <div className="flex items-center gap-6">
            <button
              type="button"
              onClick={() => handleTabChange("detect")}
              className="flex items-center gap-2 select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand rounded-control py-1 px-0.5 text-left"
            >
              <span className="text-sm font-semibold tracking-tight text-text-primary">
                VisionSSD
              </span>
              <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded-chip bg-elevated text-text-tertiary border border-border-subtle leading-none">
                v1.0.0
              </span>
            </button>

            {/* Desktop Navigation Tabs as underlined text links */}
            <nav
              className="hidden min-[720px]:flex items-center h-12 gap-1"
              aria-label="Main Navigation"
            >
              <button
                type="button"
                onClick={() => handleTabChange("detect")}
                className={cn(
                  "relative h-12 flex items-center px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "detect"
                    ? "text-text-primary font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                Detect
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("compare")}
                className={cn(
                  "relative h-12 flex items-center px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "compare"
                    ? "text-text-primary font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                Compare
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("history")}
                className={cn(
                  "relative h-12 flex items-center gap-1.5 px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "history"
                    ? "text-text-primary font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                <span>History</span>
                {history.length > 0 && (
                  <span className="font-mono text-[10px] px-1 py-0.2 rounded-chip bg-elevated text-text-secondary border border-border-subtle leading-none">
                    {history.length}
                  </span>
                )}
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("model")}
                className={cn(
                  "relative h-12 flex items-center px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "model"
                    ? "text-text-primary font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                Model
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("api")}
                className={cn(
                  "relative h-12 flex items-center px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "api"
                    ? "text-text-primary font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                API
              </button>

              <button
                type="button"
                onClick={() => handleTabChange("dev")}
                className={cn(
                  "relative h-12 flex items-center px-2.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  activeTab === "dev"
                    ? "text-brand font-semibold after:absolute after:bottom-0 after:left-0 after:right-0 after:h-[2px] after:bg-brand"
                    : "text-text-tertiary hover:text-text-primary"
                )}
              >
                Components
              </button>
            </nav>
          </div>

          {/* Right utility cluster: Command Palette, Theme, Health, Mobile menu */}
          <div className="flex items-center gap-2 sm:gap-2.5">
            {/* Command Palette Button */}
            <button
              type="button"
              onClick={() => setCommandPaletteOpen(true)}
              className="flex items-center gap-2 h-7 px-2 text-xs text-text-secondary bg-elevated hover:bg-subtle hover:text-text-primary border border-border-subtle rounded-control transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
              aria-label={`Open command palette (${isMac ? "Cmd+K" : "Ctrl+K"})`}
            >
              <Search className="w-3.5 h-3.5 text-text-tertiary" />
              <span className="hidden sm:inline text-text-tertiary text-xs">Search…</span>
              <kbd className="inline-flex items-center rounded-chip bg-card border border-border-muted px-1 py-0.2 text-[10px] font-mono text-text-tertiary leading-none">
                {isMac ? "⌘K" : "Ctrl+K"}
              </kbd>
            </button>

            <div className="h-4 w-px bg-border-subtle" />

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* API Health Dot with Tooltip */}
            <ApiHealthDot health={apiHealth} />

            {/* Mobile Nav for < 720px */}
            <MobileNav
              activeTab={activeTab}
              onSelectTab={handleTabChange}
              historyCount={history.length}
            />
          </div>
        </div>
      </header>

      {/* Main Content with uniform PageContainer */}
      <main id="main-content" tabIndex={-1} className="flex-grow w-full outline-none">
        {activeTab === "detect" && (
          <PageContainer>
            <PageHeader
              title="Detect"
              description="Real-time object detection with MobileNet V2 SSD across 20 object classes."
            />
            <DetectView
              key={activeDetectionRecord?.id || "fresh"}
              onSaveHistory={handleSaveHistory}
              initialRecord={activeDetectionRecord}
            />
          </PageContainer>
        )}
        {activeTab === "compare" && (
          <PageContainer>
            <PageHeader
              title="Model Comparison"
              description="Benchmark two object detection models concurrently with synchronized zoom and pan."
            />
            <CompareView />
          </PageContainer>
        )}
        {activeTab === "history" && (
          <PageContainer>
            <PageHeader
              title="History"
              description="Review past detection results, confidence scores, and timings from this browser."
            />
            <HistoryView
              history={history}
              onSelectRecord={handleSelectHistoryRecord}
              onClearHistory={handleClearHistory}
              onDeleteRecord={handleDeleteRecord}
              onNavigateToDetect={() => setActiveTab("detect")}
            />
          </PageContainer>
        )}
        {activeTab === "model" && (
          <PageContainer>
            <PageHeader
              title="Model & Metrics"
              description="SSD MobileNet V2 architecture specifications, validation benchmarks, and class taxonomy."
            />
            <ModelView />
          </PageContainer>
        )}
        {activeTab === "api" && (
          <PageContainer>
            <PageHeader
              title="API Reference"
              description="REST API documentation, request schemas, and code samples for programmatic inference."
            />
            <ApiDocsView />
          </PageContainer>
        )}
        {activeTab === "dev" && <DevComponentsView />}
      </main>

      {/* Slim Footer: app version, model name, link to model page (no ports or stack details) */}
      <footer className="border-t border-border-subtle bg-subtle py-3 text-xs text-text-tertiary mt-auto">
        <div className="max-w-container mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="font-medium text-text-secondary">VisionSSD v1.0.0</span>
            <span>·</span>
            <span>SSD MobileNet V2</span>
          </div>
          <button
            type="button"
            onClick={() => handleTabChange("model")}
            className="text-text-secondary hover:text-brand transition-colors text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand rounded-control px-1"
          >
            Model specifications & benchmarks →
          </button>
        </div>
      </footer>

      {/* Global Command Palette */}
      <CommandPalette
        open={commandPaletteOpen}
        onOpenChange={setCommandPaletteOpen}
        onNavigate={handleTabChange}
      />
    </div>
  );
}

export default App;
