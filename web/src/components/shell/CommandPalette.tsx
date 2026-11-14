import React, { useEffect, useRef, useState, useMemo } from "react";
import {
  Scan,
  History,
  Cpu,
  Terminal,
  Sun,
  Moon,
  Laptop,
  Search,
  Palette,
  ArrowRight,
  Upload,
  Tag,
  HelpCircle,
  XCircle,
  Download,
} from "lucide-react";
import * as RadixDialog from "@radix-ui/react-dialog";
import { cn } from "../../lib/utils";
import { useTheme, type Theme } from "../../lib/theme";

export type NavTab = "detect" | "compare" | "history" | "model" | "api" | "dev";

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon: React.ReactNode;
  group: "Navigate" | "Actions" | "Theme";
  keywords?: string[];
  shortcut?: string;
  action: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate: (tab: NavTab) => void;
}

const isMac =
  typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform);

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  open,
  onOpenChange,
  onNavigate,
}) => {
  const { setTheme } = useTheme();
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const allItems: CommandItem[] = useMemo(
    () => [
      // Navigation
      {
        id: "nav-detect",
        label: "Detect",
        description: "Run object detection on an image",
        icon: <Scan className="w-4 h-4" />,
        group: "Navigate",
        keywords: ["upload", "image", "run", "detect", "inference"],
        action: () => {
          onNavigate("detect");
          onOpenChange(false);
        },
      },
      {
        id: "nav-history",
        label: "History",
        description: "View past detection results",
        icon: <History className="w-4 h-4" />,
        group: "Navigate",
        keywords: ["past", "results", "log", "previous"],
        action: () => {
          onNavigate("history");
          onOpenChange(false);
        },
      },
      {
        id: "nav-model",
        label: "Model & Metrics",
        description: "Model info, accuracy and benchmarks",
        icon: <Cpu className="w-4 h-4" />,
        group: "Navigate",
        keywords: ["model", "metrics", "accuracy", "map", "benchmark", "performance"],
        action: () => {
          onNavigate("model");
          onOpenChange(false);
        },
      },
      {
        id: "nav-api",
        label: "API Docs",
        description: "REST API reference and examples",
        icon: <Terminal className="w-4 h-4" />,
        group: "Navigate",
        keywords: ["api", "docs", "reference", "rest", "curl", "endpoint"],
        action: () => {
          onNavigate("api");
          onOpenChange(false);
        },
      },
      {
        id: "nav-dev",
        label: "Component Library",
        description: "Design system component showcase",
        icon: <Palette className="w-4 h-4" />,
        group: "Navigate",
        keywords: ["dev", "components", "design", "system", "tokens"],
        action: () => {
          onNavigate("dev");
          onOpenChange(false);
        },
      },

      // Actions
      {
        id: "act-upload",
        label: "Upload new image",
        description: "Browse filesystem for an image",
        icon: <Upload className="w-4 h-4" />,
        group: "Actions",
        shortcut: "U",
        keywords: ["browse", "upload", "file", "open", "image"],
        action: () => {
          onNavigate("detect");
          onOpenChange(false);
          setTimeout(() => {
            window.dispatchEvent(new KeyboardEvent("keydown", { key: "u" }));
          }, 100);
        },
      },
      {
        id: "act-toggle-labels",
        label: "Toggle label chips",
        description: "Show or hide labels on bounding boxes",
        icon: <Tag className="w-4 h-4" />,
        group: "Actions",
        shortcut: "Space",
        keywords: ["labels", "tags", "scores", "hide", "show"],
        action: () => {
          onNavigate("detect");
          onOpenChange(false);
          setTimeout(() => {
            window.dispatchEvent(new KeyboardEvent("keydown", { code: "Space" }));
          }, 100);
        },
      },
      {
        id: "act-clear-selection",
        label: "Clear active selection",
        description: "Unpin and deselect active detection",
        icon: <XCircle className="w-4 h-4" />,
        group: "Actions",
        shortcut: "Esc",
        keywords: ["unpin", "clear", "deselect"],
        action: () => {
          onOpenChange(false);
          window.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
        },
      },
      {
        id: "act-shortcuts",
        label: "Keyboard shortcuts help",
        description: "View all workspace keyboard hotkeys",
        icon: <HelpCircle className="w-4 h-4" />,
        group: "Actions",
        shortcut: "?",
        keywords: ["shortcuts", "hotkeys", "help", "keys"],
        action: () => {
          onNavigate("detect");
          onOpenChange(false);
          setTimeout(() => {
            window.dispatchEvent(new KeyboardEvent("keydown", { key: "?" }));
          }, 100);
        },
      },
      {
        id: "act-export",
        label: "Export detections",
        description: "Download JSON, CSV, or annotated PNG",
        icon: <Download className="w-4 h-4" />,
        group: "Actions",
        keywords: ["export", "download", "csv", "json", "png"],
        action: () => {
          onNavigate("detect");
          onOpenChange(false);
        },
      },

      // Theme
      {
        id: "theme-light",
        label: "Light theme",
        icon: <Sun className="w-4 h-4" />,
        group: "Theme",
        keywords: ["light", "day", "white"],
        action: () => {
          setTheme("light" as Theme);
          onOpenChange(false);
        },
      },
      {
        id: "theme-dark",
        label: "Dark theme",
        icon: <Moon className="w-4 h-4" />,
        group: "Theme",
        keywords: ["dark", "night", "black"],
        action: () => {
          setTheme("dark" as Theme);
          onOpenChange(false);
        },
      },
      {
        id: "theme-system",
        label: "System theme",
        description: "Follow OS colour scheme",
        icon: <Laptop className="w-4 h-4" />,
        group: "Theme",
        keywords: ["system", "auto", "os"],
        action: () => {
          setTheme("system" as Theme);
          onOpenChange(false);
        },
      },
    ],
    [onNavigate, onOpenChange, setTheme]
  );

  const q = query.trim().toLowerCase();
  const filtered = useMemo(() => {
    if (!q) return allItems;
    return allItems.filter(
      (item) =>
        item.label.toLowerCase().includes(q) ||
        item.description?.toLowerCase().includes(q) ||
        item.keywords?.some((k) => k.includes(q))
    );
  }, [allItems, q]);

  // Group results
  const groups: Array<{ label: CommandItem["group"]; items: CommandItem[] }> = useMemo(() => {
    const res: Array<{ label: CommandItem["group"]; items: CommandItem[] }> = [];
    for (const item of filtered) {
      let g = res.find((group) => group.label === item.group);
      if (!g) {
        g = { label: item.group, items: [] };
        res.push(g);
      }
      g.items.push(item);
    }
    return res;
  }, [filtered]);

  const flatFiltered = useMemo(() => groups.flatMap((g) => g.items), [groups]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      setQuery("");
      setSelectedIndex(0);
    }
    onOpenChange(nextOpen);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((i) => Math.min(i + 1, flatFiltered.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      flatFiltered[selectedIndex]?.action();
    } else if (e.key === "Escape") {
      handleOpenChange(false);
    }
  };

  // Scroll selected item into view
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-idx="${selectedIndex}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [selectedIndex]);

  let flatIdx = 0;

  return (
    <RadixDialog.Root open={open} onOpenChange={handleOpenChange}>
      <RadixDialog.Portal>
        <RadixDialog.Overlay className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0" />
        <RadixDialog.Content
          className="fixed left-1/2 top-[20vh] z-50 w-full max-w-[560px] -translate-x-1/2 rounded-panel border border-border-subtle bg-card shadow-2xl data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0 data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95 data-[state=open]:slide-in-from-top-[8%] focus:outline-none"
          aria-label="Command palette"
          onKeyDown={handleKeyDown}
        >
          {/* Search input */}
          <div className="flex items-center gap-2.5 border-b border-border-subtle px-4 py-3">
            <Search className="w-4 h-4 text-text-tertiary shrink-0" />
            <input
              ref={inputRef}
              type="text"
              placeholder="Type a command or search…"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setSelectedIndex(0);
              }}
              className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary outline-none"
              autoComplete="off"
              spellCheck={false}
              autoFocus
            />
            <kbd className="hidden sm:inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1.5 py-0.5 text-[10px] font-mono text-text-tertiary">
              esc
            </kbd>
          </div>

          {/* Results */}
          <div
            ref={listRef}
            className="max-h-[320px] overflow-y-auto p-1.5"
            role="listbox"
            aria-label="Commands"
          >
            {groups.length === 0 && (
              <div className="py-8 text-center text-sm text-text-tertiary">
                No results for &ldquo;{query}&rdquo;
              </div>
            )}
            {groups.map((group) => (
              <div key={group.label}>
                <div className="px-2 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-widest text-text-tertiary select-none">
                  {group.label}
                </div>
                {group.items.map((item) => {
                  const idx = flatIdx++;
                  const isSelected = idx === selectedIndex;
                  return (
                    <button
                      key={item.id}
                      type="button"
                      role="option"
                      aria-selected={isSelected}
                      data-idx={idx}
                      onClick={item.action}
                      onMouseEnter={() => setSelectedIndex(idx)}
                      className={cn(
                        "flex w-full items-center gap-3 rounded-control px-3 py-2 text-left transition-colors",
                        isSelected
                          ? "bg-brand-subtle text-brand"
                          : "text-text-secondary hover:bg-elevated hover:text-text-primary"
                      )}
                    >
                      <span className="shrink-0 opacity-70">{item.icon}</span>
                      <span className="flex-1 min-w-0">
                        <span className="block text-sm font-medium leading-tight">
                          {item.label}
                        </span>
                        {item.description && (
                          <span className="block text-xs opacity-60 truncate mt-0.5">
                            {item.description}
                          </span>
                        )}
                      </span>
                      {item.shortcut && (
                        <kbd className="font-mono text-[10px] px-1.5 py-0.5 rounded-chip bg-elevated border border-border-subtle text-text-tertiary mr-1">
                          {item.shortcut}
                        </kbd>
                      )}
                      {isSelected && (
                        <ArrowRight className="w-3.5 h-3.5 shrink-0 opacity-50" />
                      )}
                    </button>
                  );
                })}
              </div>
            ))}
          </div>

          {/* Footer hint */}
          <div className="flex items-center gap-3 border-t border-border-subtle px-4 py-2 text-[10px] text-text-tertiary select-none">
            <span className="flex items-center gap-1">
              <kbd className="inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1 py-0.5 font-mono">↑↓</kbd>
              navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1 py-0.5 font-mono">↵</kbd>
              open
            </span>
            <span className="flex items-center gap-1">
              <kbd className="inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1 py-0.5 font-mono">esc</kbd>
              close
            </span>
            <span className="ml-auto flex items-center gap-1 opacity-60">
              <kbd className="inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1 py-0.5 font-mono">
                {isMac ? "⌘" : "ctrl"}
              </kbd>
              <kbd className="inline-flex items-center rounded-chip border border-border-muted bg-elevated px-1 py-0.5 font-mono">K</kbd>
            </span>
          </div>
        </RadixDialog.Content>
      </RadixDialog.Portal>
    </RadixDialog.Root>
  );
};
