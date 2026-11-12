import React, { useState, useMemo } from "react";
import {
  Clock,
  Trash2,
  Search,
  LayoutGrid,
  List,
  ArrowRight,
  Filter,
  ShieldCheck,
  AlertTriangle,
  Box,
} from "lucide-react";
import type { HistoryRecord } from "../types";
import { formatTimeAgo } from "../lib/historyStorage";
import { colorForClass } from "../lib/colors";
import { Button } from "../components/ui/Button";
import { Badge } from "../components/ui/Badge";
import { Input } from "../components/ui/Input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "../components/ui/Dialog";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "../components/ui/Table";
import { cn } from "../lib/utils";

interface HistoryViewProps {
  history: HistoryRecord[];
  onSelectRecord: (record: HistoryRecord) => void;
  onClearHistory: () => void;
  onDeleteRecord?: (id: string) => void;
  onNavigateToDetect?: () => void;
}

type SortOption = "newest" | "oldest" | "objects-desc" | "objects-asc" | "latency-asc";

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onSelectRecord,
  onClearHistory,
  onDeleteRecord,
  onNavigateToDetect,
}) => {
  const [viewMode, setViewMode] = useState<"grid" | "table">("grid");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [classFilter, setClassFilter] = useState<string>("all");
  const [sortOption, setSortOption] = useState<SortOption>("newest");
  const [clearDialogOpen, setClearDialogOpen] = useState<boolean>(false);

  // Extract all unique classes in history for filter dropdown
  const allClasses = useMemo(() => {
    const set = new Set<string>();
    history.forEach((item) => {
      item.result.detections.forEach((d) => set.add(d.class_name));
    });
    return Array.from(set).sort();
  }, [history]);

  // Filter & Sort pipeline
  const filteredHistory = useMemo(() => {
    let list = [...history];

    // Search query filter
    const q = searchQuery.trim().toLowerCase();
    if (q) {
      list = list.filter((item) => {
        const matchesClass = item.result.detections.some((d) =>
          d.class_name.toLowerCase().includes(q)
        );
        const matchesTime = item.timestamp.toLowerCase().includes(q);
        const matchesId = item.id.toLowerCase().includes(q);
        return matchesClass || matchesTime || matchesId;
      });
    }

    // Class category filter
    if (classFilter !== "all") {
      list = list.filter((item) =>
        item.result.detections.some((d) => d.class_name === classFilter)
      );
    }

    // Sorting
    list.sort((a, b) => {
      const timeA = a.createdAt || 0;
      const timeB = b.createdAt || 0;

      if (sortOption === "newest") return timeB - timeA;
      if (sortOption === "oldest") return timeA - timeB;
      if (sortOption === "objects-desc") {
        return b.result.total_detections - a.result.total_detections;
      }
      if (sortOption === "objects-asc") {
        return a.result.total_detections - b.result.total_detections;
      }
      if (sortOption === "latency-asc") {
        return a.result.inference_time_ms - b.result.inference_time_ms;
      }
      return 0;
    });

    return list;
  }, [history, searchQuery, classFilter, sortOption]);

  const handleClearConfirmed = () => {
    onClearHistory();
    setClearDialogOpen(false);
  };

  return (
    <div className="flex flex-col gap-6 w-full">
      {/* Privacy Notice Strip */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-control bg-elevated border border-border-subtle text-xs text-text-secondary select-none">
        <ShieldCheck className="w-4 h-4 text-brand shrink-0" />
        <span className="flex-1">
          History is stored locally in your browser storage (IndexedDB). No images, frames, or detection logs are ever uploaded or saved on a remote server.
        </span>
        <span className="font-mono text-[11px] text-text-tertiary tabular-nums">
          {history.length} / 200 stored
        </span>
      </div>

      {/* Toolbar: Search, Filters, Sorters, View Toggle, Clear All */}
      {history.length > 0 && (
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 pb-2 border-b border-border-subtle">
          {/* Left: Search input */}
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-text-tertiary" />
            <Input
              type="text"
              placeholder="Search by class, date, or ID…"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-8 text-xs h-8"
            />
          </div>

          {/* Right Controls */}
          <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">
            {/* Class filter dropdown */}
            <div className="flex items-center gap-1.5 bg-card border border-border-subtle rounded-control px-2 h-8 text-xs">
              <Filter className="w-3.5 h-3.5 text-text-tertiary shrink-0" />
              <select
                value={classFilter}
                onChange={(e) => setClassFilter(e.target.value)}
                className="bg-transparent text-xs text-text-primary outline-none cursor-pointer pr-1"
                aria-label="Filter by class"
              >
                <option value="all" className="bg-card">All classes</option>
                {allClasses.map((cls) => (
                  <option key={cls} value={cls} className="bg-card">
                    {cls}
                  </option>
                ))}
              </select>
            </div>

            {/* Sort selector */}
            <select
              value={sortOption}
              onChange={(e) => setSortOption(e.target.value as SortOption)}
              className="bg-card border border-border-subtle rounded-control px-2.5 h-8 text-xs text-text-primary outline-none cursor-pointer"
              aria-label="Sort history"
            >
              <option value="newest" className="bg-card">Newest first</option>
              <option value="oldest" className="bg-card">Oldest first</option>
              <option value="objects-desc" className="bg-card">Most objects</option>
              <option value="objects-asc" className="bg-card">Fewest objects</option>
              <option value="latency-asc" className="bg-card">Fastest latency</option>
            </select>

            {/* View Mode Toggle: Grid vs Table */}
            <div className="inline-flex p-0.5 rounded-control bg-elevated border border-border-subtle shrink-0">
              <button
                type="button"
                onClick={() => setViewMode("grid")}
                className={cn(
                  "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  viewMode === "grid"
                    ? "bg-card text-text-primary shadow-sm border border-border-subtle"
                    : "text-text-tertiary hover:text-text-primary"
                )}
                title="Grid view"
                aria-label="Grid view"
              >
                <LayoutGrid className="w-3.5 h-3.5" />
              </button>
              <button
                type="button"
                onClick={() => setViewMode("table")}
                className={cn(
                  "p-1.5 rounded-[4px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
                  viewMode === "table"
                    ? "bg-card text-text-primary shadow-sm border border-border-subtle"
                    : "text-text-tertiary hover:text-text-primary"
                )}
                title="Table view"
                aria-label="Table view"
              >
                <List className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Clear All Dialog Trigger */}
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setClearDialogOpen(true)}
              leftIcon={<Trash2 className="w-3.5 h-3.5 text-status-danger" />}
            >
              Clear all
            </Button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      {history.length === 0 ? (
        /* Designed Empty State */
        <div className="flex flex-col items-center justify-center py-20 px-4 bg-card border border-border-subtle rounded-panel text-center">
          <div className="flex h-12 w-12 items-center justify-center rounded-control bg-elevated border border-border-subtle text-text-tertiary mb-3">
            <Clock className="w-6 h-6" />
          </div>
          <h3 className="text-base font-semibold tracking-tight text-text-primary mb-1">
            No detection history yet
          </h3>
          <p className="text-xs text-text-secondary max-w-sm mb-5 leading-relaxed">
            Inferences processed from the Detect workspace, sample images, or live webcam will appear here for inspection.
          </p>
          <Button
            variant="primary"
            size="md"
            onClick={onNavigateToDetect}
            rightIcon={<ArrowRight className="w-3.5 h-3.5" />}
          >
            Open Detect workspace
          </Button>
        </div>
      ) : filteredHistory.length === 0 ? (
        /* No Search Matches */
        <div className="py-16 text-center text-xs text-text-secondary bg-card border border-border-subtle rounded-panel">
          <p className="font-semibold text-text-primary mb-1">No matching detections found</p>
          <p className="text-text-tertiary mb-3">Try adjusting your search query or class filter.</p>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              setSearchQuery("");
              setClassFilter("all");
            }}
          >
            Reset filters
          </Button>
        </div>
      ) : viewMode === "grid" ? (
        /* Responsive Grid View */
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filteredHistory.map((item) => {
            const timeAgo = item.createdAt ? formatTimeAgo(item.createdAt) : item.timestamp;
            const topDetections = item.result.detections.slice(0, 3);
            const remainingCount = item.result.detections.length - topDetections.length;

            return (
              <div
                key={item.id}
                onClick={() => onSelectRecord(item)}
                className="group relative flex flex-col overflow-hidden rounded-panel border border-border-subtle bg-card transition hover:border-border-strong hover:shadow-sm cursor-pointer select-none"
              >
                {/* Thumbnail with pre-drawn boxes */}
                <div className="relative aspect-video w-full overflow-hidden bg-subtle">
                  <img
                    src={item.thumbnail || item.previewUrl}
                    alt={`Detection ${item.id}`}
                    loading="lazy"
                    className="h-full w-full object-cover transition duration-200 group-hover:scale-105"
                  />

                  {/* Top-right latency badge */}
                  <span className="absolute top-2 right-2 px-1.5 py-0.5 rounded-chip bg-card/90 backdrop-blur border border-border-subtle text-[10px] font-mono text-text-secondary tabular-nums">
                    {item.result.inference_time_ms.toFixed(1)} ms
                  </span>

                  {/* Bottom-left objects count */}
                  <span className="absolute bottom-2 left-2 px-1.5 py-0.5 rounded-chip bg-card/90 backdrop-blur border border-border-subtle text-[11px] font-medium text-text-primary flex items-center gap-1">
                    <Box className="w-3 h-3 text-brand" />
                    <span>{item.result.total_detections} detected</span>
                  </span>
                </div>

                {/* Card Details */}
                <div className="p-3.5 flex flex-col gap-2.5 flex-1 justify-between">
                  <div>
                    {/* Timestamp & Relative time */}
                    <div className="flex items-center justify-between text-[11px] text-text-tertiary mb-1.5">
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{timeAgo}</span>
                      </span>
                      <span className="font-mono text-[10px]">{item.timestamp}</span>
                    </div>

                    {/* Class Chips */}
                    <div className="flex flex-wrap gap-1 mt-1">
                      {topDetections.map((d, i) => {
                        const classCol = colorForClass(d.class_id);
                        return (
                          <span
                            key={i}
                            className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-chip bg-elevated border border-border-subtle text-[10px] font-medium text-text-primary truncate max-w-[120px]"
                          >
                            <span
                              className="w-1.5 h-1.5 rounded-full shrink-0"
                              style={{ backgroundColor: classCol.hex }}
                            />
                            <span className="truncate">{d.class_name}</span>
                          </span>
                        );
                      })}
                      {remainingCount > 0 && (
                        <span className="px-1.5 py-0.5 rounded-chip bg-elevated border border-border-subtle text-[10px] font-mono text-text-tertiary">
                          +{remainingCount}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Bottom Actions */}
                  <div className="flex items-center justify-between pt-2 border-t border-border-subtle text-xs">
                    {onDeleteRecord ? (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteRecord(item.id);
                        }}
                        className="p-1 rounded-control text-text-tertiary hover:text-status-danger transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
                        title="Delete item"
                        aria-label="Delete history item"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    ) : (
                      <div />
                    )}

                    <span className="flex items-center gap-1 text-xs font-medium text-brand group-hover:underline">
                      <span>Inspect in Detect</span>
                      <ArrowRight className="w-3 h-3 transition-transform group-hover:translate-x-0.5" />
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        /* Dense Table View */
        <Table dense>
          <TableHeader>
            <TableRow>
              <TableHead className="w-20">Preview</TableHead>
              <TableHead>Time</TableHead>
              <TableHead>Objects</TableHead>
              <TableHead>Classes</TableHead>
              <TableHead>Latency</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredHistory.map((item) => {
              const timeAgo = item.createdAt ? formatTimeAgo(item.createdAt) : item.timestamp;
              const topClasses = Array.from(
                new Set(item.result.detections.map((d) => d.class_name))
              ).slice(0, 3);

              return (
                <TableRow
                  key={item.id}
                  onClick={() => onSelectRecord(item)}
                  className="cursor-pointer hover:bg-subtle transition-colors"
                >
                  {/* Thumbnail */}
                  <TableCell className="w-20 py-2">
                    <div className="w-16 h-10 rounded-control overflow-hidden bg-subtle border border-border-subtle">
                      <img
                        src={item.thumbnail || item.previewUrl}
                        alt="Thumbnail"
                        className="w-full h-full object-cover"
                        loading="lazy"
                      />
                    </div>
                  </TableCell>

                  {/* Time */}
                  <TableCell>
                    <div className="flex flex-col">
                      <span className="font-medium text-text-primary text-xs">{timeAgo}</span>
                      <span className="text-[10px] text-text-tertiary font-mono">{item.timestamp}</span>
                    </div>
                  </TableCell>

                  {/* Objects */}
                  <TableCell>
                    <Badge variant="neutral" mono>
                      {item.result.total_detections} objects
                    </Badge>
                  </TableCell>

                  {/* Classes */}
                  <TableCell>
                    <div className="flex items-center gap-1.5 flex-wrap">
                      {topClasses.map((name) => (
                        <span
                          key={name}
                          className="px-1.5 py-0.5 rounded-chip bg-elevated border border-border-subtle text-[10px] text-text-secondary"
                        >
                          {name}
                        </span>
                      ))}
                    </div>
                  </TableCell>

                  {/* Latency */}
                  <TableCell className="font-mono text-[11px] tabular-nums text-text-secondary">
                    {item.result.inference_time_ms.toFixed(1)} ms
                  </TableCell>

                  {/* Actions */}
                  <TableCell className="text-right">
                    <div className="flex items-center justify-end gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectRecord(item);
                        }}
                        rightIcon={<ArrowRight className="w-3 h-3" />}
                      >
                        Inspect
                      </Button>
                      {onDeleteRecord && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteRecord(item.id);
                          }}
                          className="p-1.5 rounded-control text-text-tertiary hover:text-status-danger transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
                          title="Delete record"
                          aria-label="Delete record"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      )}

      {/* Clear All Confirmation Modal */}
      <Dialog open={clearDialogOpen} onOpenChange={setClearDialogOpen}>
        <DialogContent className="max-w-sm">
          <DialogHeader>
            <div className="flex h-10 w-10 items-center justify-center rounded-control bg-status-danger-subtle text-status-danger mb-2">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <DialogTitle>Clear Detection History</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete all {history.length} saved detection records from browser storage? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="secondary" size="sm">
                Cancel
              </Button>
            </DialogClose>
            <Button variant="danger" size="sm" onClick={handleClearConfirmed}>
              Clear all records
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};
