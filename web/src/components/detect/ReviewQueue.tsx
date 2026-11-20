/**
 * web/src/components/detect/ReviewQueue.tsx
 * -----------------------------------------
 * Active learning review queue with uncertainty ranking (Day 4 Commit 9).
 * Features:
 *   - Queue sorted by compound uncertainty (threshold proximity + model disagreements)
 *   - Human-readable uncertainty reasons per item
 *   - Review status tracking (pending, reviewed, skipped, corrected)
 *   - Progress counter with visual breakdown
 *   - Keyboard shortcuts (R: Review, S: Skip, [: Prev, ]: Next)
 *   - Fully accessible with visible focus and responsive mobile/desktop layouts
 */

import React, { useState, useEffect, useMemo, useCallback } from "react";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FastForward,
  AlertTriangle,
  Sparkles,
  ListFilter,
  Check,
  Clock,
  Edit3,
} from "lucide-react";
import { Button } from "../ui/Button";
import { Badge } from "../ui/Badge";
import { Kbd } from "../ui/Kbd";
import { cn } from "../../lib/utils";
import type { ReviewQueueItem, ReviewStatus } from "../../types";

interface ReviewQueueProps {
  queue: ReviewQueueItem[];
  currentIndex: number;
  onSelectIndex: (index: number) => void;
  onUpdateStatus: (itemId: string, status: ReviewStatus) => void;
  onSaveCorrection?: () => void;
  isSavingCorrection?: boolean;
  className?: string;
}

export const ReviewQueue: React.FC<ReviewQueueProps> = ({
  queue,
  currentIndex,
  onSelectIndex,
  onUpdateStatus,
  onSaveCorrection,
  isSavingCorrection = false,
  className,
}) => {
  const [filter, setFilter] = useState<"all" | ReviewStatus>("all");
  const [showDrawer, setShowDrawer] = useState<boolean>(false);
  const [announcement, setAnnouncement] = useState<string>("");

  const currentItem = queue[currentIndex] || null;

  // Counts for progress counter
  const counts = useMemo(() => {
    let pending = 0;
    let reviewed = 0;
    let skipped = 0;
    let corrected = 0;
    for (const item of queue) {
      if (item.status === "pending") pending++;
      else if (item.status === "reviewed") reviewed++;
      else if (item.status === "skipped") skipped++;
      else if (item.status === "corrected") corrected++;
    }
    const completed = reviewed + corrected;
    const total = queue.length;
    const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
    return { pending, reviewed, skipped, corrected, completed, total, percent };
  }, [queue]);

  const handleNext = useCallback(() => {
    if (currentIndex < queue.length - 1) {
      onSelectIndex(currentIndex + 1);
    }
  }, [currentIndex, queue.length, onSelectIndex]);

  const handlePrev = useCallback(() => {
    if (currentIndex > 0) {
      onSelectIndex(currentIndex - 1);
    }
  }, [currentIndex, onSelectIndex]);

  const handleMarkReviewed = useCallback(() => {
    if (!currentItem) return;
    onUpdateStatus(currentItem.id, "reviewed");
    setAnnouncement(`Image ${currentItem.imageName} marked as reviewed.`);
    handleNext();
  }, [currentItem, onUpdateStatus, handleNext]);

  const handleSkip = useCallback(() => {
    if (!currentItem) return;
    onUpdateStatus(currentItem.id, "skipped");
    setAnnouncement(`Image ${currentItem.imageName} skipped.`);
    handleNext();
  }, [currentItem, onUpdateStatus, handleNext]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger if user is typing in an input
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      ) {
        return;
      }

      if (e.key === "r" || e.key === "R") {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          handleMarkReviewed();
        }
      } else if (e.key === "s" || e.key === "S") {
        if (!e.ctrlKey && !e.metaKey) {
          e.preventDefault();
          handleSkip();
        }
      } else if (e.key === "]") {
        e.preventDefault();
        handleNext();
      } else if (e.key === "[") {
        e.preventDefault();
        handlePrev();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleMarkReviewed, handleSkip, handleNext, handlePrev]);

  // Filtered queue for sidebar list
  const filteredQueue = useMemo(() => {
    if (filter === "all") return queue;
    return queue.filter((item) => item.status === filter);
  }, [queue, filter]);

  const getStatusBadge = (status: ReviewStatus) => {
    switch (status) {
      case "reviewed":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            <Check className="w-3 h-3" /> Reviewed
          </span>
        );
      case "corrected":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-brand/10 text-brand border border-brand/20">
            <Edit3 className="w-3 h-3" /> Corrected
          </span>
        );
      case "skipped":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-amber-500/10 text-amber-500 border border-amber-500/20">
            <FastForward className="w-3 h-3" /> Skipped
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-medium bg-text-tertiary/10 text-text-tertiary border border-border-subtle">
            <Clock className="w-3 h-3" /> Pending
          </span>
        );
    }
  };

  return (
    <div
      className={cn(
        "flex flex-col bg-card border border-border-subtle rounded-panel overflow-hidden",
        className
      )}
    >
      {/* Live region for screen readers */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {announcement}
      </div>

      {/* Progress Counter & Global Queue Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-3 bg-surface border-b border-border-subtle gap-2">
        <div className="flex flex-col gap-1 min-w-[200px]">
          <div className="flex items-center justify-between text-xs font-medium">
            <span className="flex items-center gap-1.5 text-text-primary">
              <Sparkles className="w-3.5 h-3.5 text-brand" />
              <span>Active Learning Queue</span>
            </span>
            <span className="text-text-secondary font-mono">
              {counts.completed} / {counts.total} ({counts.percent}%)
            </span>
          </div>
          {/* Progress bar */}
          <div
            className="w-full bg-elevated h-1.5 rounded-full overflow-hidden flex"
            role="progressbar"
            aria-valuenow={counts.percent}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label="Review progress"
          >
            <div
              className="bg-emerald-500 h-full transition-all duration-300"
              style={{
                width: `${counts.total > 0 ? (counts.reviewed / counts.total) * 100 : 0}%`,
              }}
              title={`Reviewed: ${counts.reviewed}`}
            />
            <div
              className="bg-brand h-full transition-all duration-300"
              style={{
                width: `${counts.total > 0 ? (counts.corrected / counts.total) * 100 : 0}%`,
              }}
              title={`Corrected: ${counts.corrected}`}
            />
            <div
              className="bg-amber-500 h-full transition-all duration-300"
              style={{
                width: `${counts.total > 0 ? (counts.skipped / counts.total) * 100 : 0}%`,
              }}
              title={`Skipped: ${counts.skipped}`}
            />
          </div>
        </div>

        {/* Status count chips & list toggle */}
        <div className="flex items-center flex-wrap gap-1.5 text-xs">
          <Badge variant="neutral" className="font-mono text-[11px]">
            Pending: {counts.pending}
          </Badge>
          <Badge variant="success" className="font-mono text-[11px]">
            Reviewed: {counts.reviewed}
          </Badge>
          <Badge variant="brand" className="font-mono text-[11px]">
            Corrected: {counts.corrected}
          </Badge>
          <Badge variant="warning" className="font-mono text-[11px]">
            Skipped: {counts.skipped}
          </Badge>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDrawer((prev) => !prev)}
            className="text-xs h-7 ml-1"
            leftIcon={<ListFilter className="w-3 h-3" />}
          >
            {showDrawer ? "Hide List" : "Show Queue"}
          </Button>
        </div>
      </div>

      {/* Main Review Body */}
      <div className="flex flex-col lg:flex-row min-h-0 flex-1">
        {/* Active Candidate Details & Controls */}
        <div className="flex-1 p-3 flex flex-col gap-3 min-w-0">
          {currentItem ? (
            <>
              {/* Top Item Metadata */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-border-subtle">
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded bg-brand/10 text-brand text-xs font-semibold font-mono border border-brand/20">
                    Rank #{currentItem.rank}
                  </span>
                  <span className="text-xs text-text-tertiary">|</span>
                  <span className="text-xs font-medium text-text-primary truncate max-w-[200px]" title={currentItem.imageName}>
                    {currentItem.imageName}
                  </span>
                  {getStatusBadge(currentItem.status)}
                </div>

                <div className="flex items-center gap-2 text-xs font-mono text-text-secondary">
                  <span>Score: {currentItem.score.toFixed(2)}</span>
                  <span className="text-text-tertiary">
                    ({currentIndex + 1} of {queue.length})
                  </span>
                </div>
              </div>

              {/* Reasons Callout */}
              <div className="flex flex-col gap-1.5 p-2.5 rounded-panel bg-elevated/60 border border-border-subtle text-xs">
                <div className="flex items-center gap-1.5 font-medium text-text-secondary">
                  <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                  <span>Uncertainty Reasons</span>
                </div>
                <ul className="list-disc list-inside space-y-1 text-text-primary pl-1">
                  {currentItem.reasons.map((reason, idx) => (
                    <li key={idx} className="leading-relaxed">
                      {reason}
                    </li>
                  ))}
                </ul>
              </div>

              {/* Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                {/* Previous / Next */}
                <div className="flex items-center gap-1">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handlePrev}
                    disabled={currentIndex <= 0}
                    title="Previous item ([)"
                    aria-label="Previous item in queue"
                  >
                    <ChevronLeft className="w-4 h-4 mr-0.5" /> Prev <Kbd className="ml-1 text-[10px]">[</Kbd>
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleNext}
                    disabled={currentIndex >= queue.length - 1}
                    title="Next item (])"
                    aria-label="Next item in queue"
                  >
                    Next <Kbd className="ml-1 text-[10px]">]</Kbd> <ChevronRight className="w-4 h-4 ml-0.5" />
                  </Button>
                </div>

                {/* Status change actions */}
                <div className="flex items-center flex-wrap gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={handleSkip}
                    title="Skip candidate (S)"
                    aria-label="Skip candidate image"
                    className="text-amber-500 hover:text-amber-400 hover:bg-amber-500/10"
                  >
                    <FastForward className="w-3.5 h-3.5 mr-1" /> Skip <Kbd className="ml-1.5 text-[10px]">S</Kbd>
                  </Button>

                  {onSaveCorrection && (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={onSaveCorrection}
                      isLoading={isSavingCorrection}
                      title="Save manual box corrections"
                      aria-label="Save corrected boxes"
                    >
                      <Edit3 className="w-3.5 h-3.5 mr-1" /> Save Correction
                    </Button>
                  )}

                  <Button
                    variant="primary"
                    size="sm"
                    onClick={handleMarkReviewed}
                    title="Mark verified without changes (R)"
                    aria-label="Mark candidate as reviewed"
                    className="bg-emerald-600 hover:bg-emerald-500 text-white"
                  >
                    <CheckCircle2 className="w-3.5 h-3.5 mr-1" /> Mark Reviewed <Kbd className="ml-1.5 text-[10px] bg-emerald-700 text-white">R</Kbd>
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center p-8 text-xs text-text-tertiary">
              No review items available in the queue.
            </div>
          )}
        </div>

        {/* Collapsible Queue List Drawer */}
        {showDrawer && (
          <div className="w-full lg:w-72 border-t lg:border-t-0 lg:border-l border-border-subtle bg-surface flex flex-col p-2 gap-2 max-h-[360px] overflow-y-auto">
            <div className="flex items-center justify-between pb-1 border-b border-border-subtle">
              <span className="text-xs font-medium text-text-secondary">Queue Pool</span>
              <div className="flex items-center gap-1">
                {(["all", "pending", "reviewed", "corrected", "skipped"] as const).map((f) => (
                  <button
                    key={f}
                    type="button"
                    onClick={() => setFilter(f)}
                    className={cn(
                      "px-1.5 py-0.5 text-[10px] rounded font-medium capitalize transition-colors",
                      filter === f
                        ? "bg-brand text-white"
                        : "text-text-tertiary hover:text-text-primary"
                    )}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col gap-1.5">
              {filteredQueue.map((item) => {
                const globalIndex = queue.findIndex((q) => q.id === item.id);
                const isSelected = globalIndex === currentIndex;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => onSelectIndex(globalIndex)}
                    className={cn(
                      "flex items-center justify-between p-2 rounded text-left text-xs transition-colors border",
                      isSelected
                        ? "bg-brand/10 border-brand/30 text-text-primary shadow-sm"
                        : "bg-card hover:bg-elevated border-border-subtle text-text-secondary"
                    )}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <span className="font-mono text-[10px] px-1 py-0.5 rounded bg-surface text-brand font-semibold shrink-0">
                        #{item.rank}
                      </span>
                      <span className="truncate max-w-[120px] font-medium" title={item.imageName}>
                        {item.imageName}
                      </span>
                    </div>

                    <div className="flex items-center gap-1.5 shrink-0">
                      <span className="font-mono text-[10px] text-text-tertiary">
                        {item.score.toFixed(1)}
                      </span>
                      {item.status === "reviewed" && <Check className="w-3 h-3 text-emerald-500" />}
                      {item.status === "corrected" && <Edit3 className="w-3 h-3 text-brand" />}
                      {item.status === "skipped" && <FastForward className="w-3 h-3 text-amber-500" />}
                      {item.status === "pending" && <Clock className="w-3 h-3 text-text-tertiary" />}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
