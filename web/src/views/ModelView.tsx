import React, { useState, useEffect, useMemo, lazy, Suspense } from "react";
import { Search, ExternalLink, HelpCircle, AlertTriangle } from "lucide-react";
import { fetchModelInfo, fetchClasses } from "../api";
import {
  BENCHMARK_METRICS,
  PER_CLASS_EVALUATION,
  MODEL_SPECS,
  type ClassEvaluation,
} from "../data/evaluationData";
import { colorForClass } from "../lib/colors";
import { cn } from "../lib/utils";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "../components/ui/Table";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import { Input } from "../components/ui/Input";
import { Skeleton } from "../components/ui/Skeleton";
import { ModelCardModal } from "../components/model/ModelCardModal";
import type { ModelInfoResponse, ClassItem } from "../types";

// Lazy-load charts to maintain lean initial bundle size
const ModelCharts = lazy(() => import("../components/model/ModelCharts"));

function ChartsSkeleton() {
  return (
    <div className="flex flex-col gap-6">
      <div className="rounded-panel border border-border-subtle bg-card p-4 space-y-3">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-44 w-full" />
      </div>
      <div className="rounded-panel border border-border-subtle bg-card p-4 space-y-3">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-52 w-full" />
      </div>
      <div className="rounded-panel border border-border-subtle bg-card p-4 space-y-3">
        <Skeleton className="h-5 w-48" />
        <Skeleton className="h-72 w-full" />
      </div>
    </div>
  );
}

export const ModelView: React.FC = () => {
  const [, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [, setApiClasses] = useState<ClassItem[]>([]);
  const [filterMode, setFilterMode] = useState<"all" | "shared" | "novel">("all");
  const [searchQuery, setSearchQuery] = useState("");
  const [modelCardOpen, setModelCardOpen] = useState(false);
  const [sortField, setSortField] = useState<"id" | "name" | "ap" | "samples">("id");
  const [sortAsc, setSortAsc] = useState(true);

  // Load live API metadata if online, with fallback to verified local evaluation data
  useEffect(() => {
    let mounted = true;
    async function loadData() {
      try {
        const [mRes, cRes] = await Promise.all([fetchModelInfo(), fetchClasses()]);
        if (mounted) {
          setModelInfo(mRes);
          setApiClasses(cRes.classes);
        }
      } catch {
        // Handled silently - component uses verified evaluationData constants
      }
    }
    void loadData();
    return () => {
      mounted = false;
    };
  }, []);

  // Filter & sort taxonomy
  const filteredTaxonomy = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    const result = PER_CLASS_EVALUATION.filter((cls) => {
      // Tab filter
      if (filterMode === "shared" && !cls.is_coco) return false;
      if (filterMode === "novel" && cls.is_coco) return false;

      // Text search filter
      if (query) {
        const matchName = cls.name.toLowerCase().includes(query);
        const matchMid = cls.mid.toLowerCase().includes(query);
        const matchId = cls.id.toString() === query || `#${cls.id}` === query;
        const matchDesc = cls.description.toLowerCase().includes(query);
        return matchName || matchMid || matchId || matchDesc;
      }
      return true;
    });

    result.sort((a, b) => {
      let cmp = 0;
      if (sortField === "id") cmp = a.id - b.id;
      else if (sortField === "name") cmp = a.name.localeCompare(b.name);
      else if (sortField === "ap") cmp = b.finetuned_ap - a.finetuned_ap;
      else if (sortField === "samples") cmp = b.sample_count - a.sample_count;
      return sortAsc ? cmp : -cmp;
    });

    return result;
  }, [filterMode, searchQuery, sortField, sortAsc]);

  const handleSort = (field: "id" | "name" | "ap" | "samples") => {
    if (sortField === field) {
      setSortAsc((prev) => !prev);
    } else {
      setSortField(field);
      setSortAsc(field === "id" || field === "name");
    }
  };

  return (
    <div className="flex flex-col gap-10">
      {/* 1. Compact Spec List (Replaces 4 colored icon tiles) */}
      <section aria-labelledby="model-specs-heading">
        <div className="rounded-panel border border-border-subtle bg-card p-4 sm:p-5">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-border-subtle">
            <div>
              <h3 id="model-specs-heading" className="text-sm font-semibold text-text-primary">
                Model Technical Specifications
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Native PyTorch Single Shot MultiBox Detector with MobileNet V2 feature extractor
              </p>
            </div>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setModelCardOpen(true)}
              className="gap-1.5 text-xs"
            >
              <ExternalLink className="w-3.5 h-3.5" />
              <span>Model Card</span>
            </Button>
          </div>

          <dl className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {MODEL_SPECS.map((spec) => (
              <div
                key={spec.label}
                className="flex flex-col justify-between p-3 rounded-control bg-subtle border border-border-subtle"
              >
                <dt className="text-xs text-text-muted font-medium">{spec.label}</dt>
                <dd className="mt-1">
                  <div className="text-sm font-semibold font-mono text-text-primary">
                    {spec.value}
                  </div>
                  {spec.detail && (
                    <p className="text-[11px] text-text-secondary mt-1 leading-snug">
                      {spec.detail}
                    </p>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      {/* 2. Benchmark Metrics Table */}
      <section aria-labelledby="benchmark-metrics-heading">
        <div className="flex flex-col gap-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 id="benchmark-metrics-heading" className="text-sm font-semibold text-text-primary">
                Validation Benchmark Evaluation
              </h3>
              <p className="text-xs text-text-muted mt-0.5">
                Quantitative detection performance evaluated across 40 validation images at IoU threshold 0.50
              </p>
            </div>
            <div className="flex items-center gap-2 text-xs text-text-muted font-mono">
              <span>Optimal Operating Point:</span>
              <span className="font-semibold text-brand">0.25 conf / 0.45 IoU</span>
            </div>
          </div>

          <div className="overflow-x-auto">
            <Table dense>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[36%]">Metric</TableHead>
                  <TableHead className="w-[16%]">COCO Baseline</TableHead>
                  <TableHead className="w-[16%]">Fine-Tuned Model</TableHead>
                  <TableHead className="w-[18%]">Target Requirement</TableHead>
                  <TableHead className="w-[14%] text-right">Delta</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {BENCHMARK_METRICS.map((row) => (
                  <TableRow key={row.id}>
                    <TableCell className="font-sans font-medium text-text-primary">
                      {row.name}
                    </TableCell>
                    <TableCell className="font-mono text-text-secondary">
                      {row.baseline.toFixed(4)}
                    </TableCell>
                    <TableCell className="font-mono font-semibold text-text-primary">
                      {row.fineTuned.toFixed(4)}
                    </TableCell>
                    <TableCell className="font-mono text-text-muted text-[11px]">
                      {row.target}
                    </TableCell>
                    <TableCell className="text-right font-mono text-xs">
                      <span className="inline-flex items-center gap-1 font-medium text-brand">
                        <span>↑</span>
                        <span>{row.deltaPts}</span>
                      </span>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </div>
      </section>

      {/* 3. Lazy-Loaded Evaluation Charts */}
      <section aria-labelledby="charts-heading">
        <h3 id="charts-heading" className="sr-only">
          Evaluation Charts
        </h3>
        <Suspense fallback={<ChartsSkeleton />}>
          <ModelCharts />
        </Suspense>
      </section>

      {/* 4. Class Taxonomy Explorer */}
      <section aria-labelledby="taxonomy-heading" className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-border-subtle pb-3">
          <div>
            <h3 id="taxonomy-heading" className="text-sm font-semibold text-text-primary">
              Class Taxonomy Explorer
            </h3>
            <p className="text-xs text-text-muted mt-0.5">
              20 target categories evaluated on Open Images with verified ground-truth sample counts
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2">
            {/* Filter Tabs */}
            <div className="flex items-center rounded-control bg-subtle p-0.5 border border-border-subtle text-xs">
              <button
                type="button"
                onClick={() => setFilterMode("all")}
                className={cn(
                  "px-3 py-1 rounded-[4px] font-medium transition-colors",
                  filterMode === "all"
                    ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                All (20)
              </button>
              <button
                type="button"
                onClick={() => setFilterMode("shared")}
                className={cn(
                  "px-3 py-1 rounded-[4px] font-medium transition-colors",
                  filterMode === "shared"
                    ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                COCO Shared (10)
              </button>
              <button
                type="button"
                onClick={() => setFilterMode("novel")}
                className={cn(
                  "px-3 py-1 rounded-[4px] font-medium transition-colors",
                  filterMode === "novel"
                    ? "bg-card text-text-primary border border-border-subtle shadow-sm"
                    : "text-text-secondary hover:text-text-primary"
                )}
              >
                Novel Concepts (10)
              </button>
            </div>

            {/* Search Input */}
            <div className="relative w-full sm:w-48">
              <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-text-tertiary" />
              <Input
                type="search"
                placeholder="Filter classes…"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="h-8 pl-8 text-xs w-full"
              />
            </div>
          </div>
        </div>

        {/* Taxonomy Table */}
        <div className="overflow-x-auto">
          <Table dense>
            <TableHeader>
              <TableRow>
                <TableHead
                  sortable
                  sortDirection={sortField === "id" ? (sortAsc ? "asc" : "desc") : false}
                  onSort={() => handleSort("id")}
                  className="w-14"
                >
                  ID
                </TableHead>
                <TableHead
                  sortable
                  sortDirection={sortField === "name" ? (sortAsc ? "asc" : "desc") : false}
                  onSort={() => handleSort("name")}
                  className="w-40"
                >
                  Class Name
                </TableHead>
                <TableHead className="w-28">Category</TableHead>
                <TableHead className="w-28">Open Images MID</TableHead>
                <TableHead
                  sortable
                  sortDirection={sortField === "samples" ? (sortAsc ? "asc" : "desc") : false}
                  onSort={() => handleSort("samples")}
                  className="w-28 text-right"
                >
                  Sample Count
                </TableHead>
                <TableHead
                  sortable
                  sortDirection={sortField === "ap" ? (sortAsc ? "asc" : "desc") : false}
                  onSort={() => handleSort("ap")}
                  className="w-36 text-right"
                >
                  Fine-Tuned AP
                </TableHead>
                <TableHead className="min-w-[200px]">Description</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredTaxonomy.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={7} className="h-24 text-center text-text-muted">
                    No classes match the filter criteria &quot;{searchQuery}&quot;.
                  </TableCell>
                </TableRow>
              ) : (
                filteredTaxonomy.map((cls: ClassEvaluation) => {
                  const classColor = colorForClass(cls.id).hex;
                  return (
                    <TableRow key={cls.id}>
                      {/* ID */}
                      <TableCell className="font-mono text-text-muted text-[11px]">
                        #{cls.id.toString().padStart(2, "0")}
                      </TableCell>

                      {/* Name with Color Dot */}
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span
                            className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                            style={{ backgroundColor: classColor }}
                            aria-hidden="true"
                          />
                          <span className="font-medium text-text-primary text-xs">
                            {cls.name}
                          </span>
                        </div>
                      </TableCell>

                      {/* Category Badge */}
                      <TableCell>
                        <Badge
                          variant={cls.is_coco ? "neutral" : "brand"}
                          className="text-[10px] py-0"
                        >
                          {cls.is_coco ? "COCO Shared" : "Novel Target"}
                        </Badge>
                      </TableCell>

                      {/* Open Images MID */}
                      <TableCell className="font-mono text-[11px] text-text-muted">
                        {cls.mid}
                      </TableCell>

                      {/* Sample Count */}
                      <TableCell className="text-right font-mono text-text-secondary">
                        {cls.sample_count}
                      </TableCell>

                      {/* Fine-Tuned AP */}
                      <TableCell className="text-right font-mono font-medium text-text-primary">
                        <div className="flex items-center justify-end gap-2">
                          <span>{cls.finetuned_ap.toFixed(4)}</span>
                          <span
                            className={cn(
                              "text-[10px] font-mono",
                              cls.delta >= 0 ? "text-brand" : "text-text-muted"
                            )}
                          >
                            {cls.deltaPts}
                          </span>
                        </div>
                      </TableCell>

                      {/* Description */}
                      <TableCell className="text-text-secondary text-xs truncate max-w-xs">
                        {cls.description}
                      </TableCell>
                    </TableRow>
                  );
                })
              )}
            </TableBody>
          </Table>
        </div>
      </section>

      {/* 5. Plain-Language Notes & Known Limits */}
      <section aria-labelledby="notes-heading" className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Metric Reading Guide */}
        <div className="rounded-panel border border-border-subtle bg-card p-4 sm:p-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-text-primary font-semibold text-sm">
            <HelpCircle className="w-4 h-4 text-brand" />
            <h4 id="notes-heading">How to Read These Metrics</h4>
          </div>
          <div className="flex flex-col gap-2.5 text-xs text-text-secondary leading-relaxed">
            <p>
              <strong className="text-text-primary">mAP@0.5:</strong> Mean Average Precision requiring at least 50% Intersection over Union (IoU) between prediction and ground-truth bounding box. This is the primary detection fidelity benchmark.
            </p>
            <p>
              <strong className="text-text-primary">COCO mAP@[.5:.95]:</strong> Stricter standard computing the mean of 10 AP scores across IoU thresholds from 0.50 to 0.95 in 0.05 increments, heavily penalizing imprecise box borders.
            </p>
            <p>
              <strong className="text-text-primary">Confidence Threshold (0.25):</strong> Optimal operating point identified through F1 sweep, balancing 65.0% precision against 63.0% recall to minimize false alarms.
            </p>
          </div>
        </div>

        {/* Known Limitations */}
        <div className="rounded-panel border border-border-subtle bg-card p-4 sm:p-5 flex flex-col gap-3">
          <div className="flex items-center gap-2 text-text-primary font-semibold text-sm">
            <AlertTriangle className="w-4 h-4 text-text-muted" />
            <h4>Model Limitations & Operational Boundaries</h4>
          </div>
          <div className="flex flex-col gap-2.5 text-xs text-text-secondary leading-relaxed">
            <p>
              <strong className="text-text-primary">Small Objects (&lt; 32 px):</strong> Due to fixed 300 × 300 input resolution, objects occupying minimal pixel areas exhibit lower recall.
            </p>
            <p>
              <strong className="text-text-primary">Dense Occlusions:</strong> More than 5 overlapping instances in close proximity can trigger mutual suppression under non-maximum suppression (NMS).
            </p>
            <p>
              <strong className="text-text-primary">Environmental Variation:</strong> Low lighting and extreme camera motion blur soften edge gradients, reducing feature activations in depthwise convolutions.
            </p>
          </div>
          <div className="pt-2 border-t border-border-subtle">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setModelCardOpen(true)}
              className="gap-1.5 text-xs px-0 hover:bg-transparent hover:text-brand"
            >
              <span>Read complete Model Card documentation</span>
              <ExternalLink className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </section>

      {/* Model Card Modal */}
      <ModelCardModal open={modelCardOpen} onOpenChange={setModelCardOpen} />
    </div>
  );
};
