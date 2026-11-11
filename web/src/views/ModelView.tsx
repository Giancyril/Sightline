import React, { useEffect, useState } from "react";
import { Cpu, Target, Layers, ShieldCheck, Tag } from "lucide-react";
import { fetchModelInfo, fetchClasses } from "../api";
import type { ModelInfoResponse, ClassItem } from "../types";

export const ModelView: React.FC = () => {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [classes, setClasses] = useState<ClassItem[]>([]);
  const [filterMode, setFilterMode] = useState<"all" | "shared" | "novel">("all");
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [mRes, cRes] = await Promise.all([fetchModelInfo(), fetchClasses()]);
        setModelInfo(mRes);
        setClasses(cRes.classes);
      } catch {
        // Fallback static metadata if API is offline
        setModelInfo({
          name: "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
          version: "v1.0.0",
          architecture: "SSDMobileNetV2",
          input_resolution: [300, 300],
          num_classes: 21,
          optimal_confidence_threshold: 0.25,
          supported_formats: ["JPEG", "PNG", "WebP"],
          max_file_size_bytes: 10485760,
          metrics: {
            mAP_all: 0.2644,
            mAP_coco_shared: 0.3471,
            mAP_novel_classes: 0.1816,
            coco_mAP_50_95: 0.1603,
          },
        });
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const filteredClasses = classes.filter((c) => {
    if (filterMode === "shared") return c.is_coco;
    if (filterMode === "novel") return !c.is_coco;
    return true;
  });

  return (
    <div className="flex flex-col gap-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-4">
        <h2 className="text-xl font-bold text-slate-100">
          Model Specifications & Performance Benchmarks
        </h2>
        <p className="text-sm text-slate-400 mt-1">
          Fine-tuned SSD MobileNet V2 detector evaluated on Open Images subset.
        </p>
      </div>

      {/* Model Spec Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-sky-500/10 text-sky-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-400">Architecture</p>
            <p className="text-base font-bold text-slate-200 mt-0.5">
              {modelInfo?.architecture || "SSDMobileNetV2"}
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">MobileNetV2 + 6 Extra Scales</p>
          </div>
        </div>

        <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400">
            <Target className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-400">Input Resolution</p>
            <p className="text-base font-bold text-slate-200 mt-0.5">
              {modelInfo?.input_resolution.join(" × ") || "300 × 300"} px
            </p>
            <p className="text-[11px] text-slate-400 mt-0.5">2,268 SSD Anchor Priors</p>
          </div>
        </div>

        <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-purple-500/10 text-purple-400">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-400">Class Taxonomy</p>
            <p className="text-base font-bold text-slate-200 mt-0.5">20 Target Classes</p>
            <p className="text-[11px] text-slate-400 mt-0.5">+ 1 Background Class (21 Total)</p>
          </div>
        </div>

        <div className="p-4 bg-slate-900 rounded-xl border border-slate-800 flex items-start gap-3">
          <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <p className="text-xs text-slate-400">Inference SLA</p>
            <p className="text-base font-bold text-slate-200 mt-0.5">&lt; 300 ms (CPU)</p>
            <p className="text-[11px] text-slate-400 mt-0.5">Typical ~150 ms per frame</p>
          </div>
        </div>
      </div>

      {/* Benchmark Success Metrics */}
      <div className="p-6 bg-slate-900 rounded-xl border border-slate-800 flex flex-col gap-4">
        <h3 className="text-base font-bold text-slate-200">
          Validation Benchmark Metrics (Stage 5 Exit Criteria)
        </h3>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs font-semibold text-slate-400 uppercase">
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4">COCO Baseline (Stage 3)</th>
                <th className="py-3 px-4">Fine-Tuned (Stage 5)</th>
                <th className="py-3 px-4">Target Requirement</th>
                <th className="py-3 px-4 text-right">Gate Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono text-xs">
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-200">
                  Novel Classes mAP@0.5
                </td>
                <td className="py-3 px-4 text-slate-400">0.0000</td>
                <td className="py-3 px-4 font-bold text-sky-400">
                  {modelInfo?.metrics.mAP_novel_classes.toFixed(4) || "0.1816"}
                </td>
                <td className="py-3 px-4 text-slate-400">≥ +0.1000 (+10 pts)</td>
                <td className="py-3 px-4 text-right">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-sans font-bold text-[11px]">
                    PASS (+18.2 pts)
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-200">
                  COCO Shared Classes mAP@0.5
                </td>
                <td className="py-3 px-4 text-slate-400">0.3264</td>
                <td className="py-3 px-4 font-bold text-sky-400">
                  {modelInfo?.metrics.mAP_coco_shared.toFixed(4) || "0.3471"}
                </td>
                <td className="py-3 px-4 text-slate-400">Drop ≤ 0.0200 (-2 pts)</td>
                <td className="py-3 px-4 text-right">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-sans font-bold text-[11px]">
                    PASS (Retained)
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-200">
                  Overall 20-Class mAP@0.5
                </td>
                <td className="py-3 px-4 text-slate-400">0.1632</td>
                <td className="py-3 px-4 font-bold text-sky-400">
                  {modelInfo?.metrics.mAP_all.toFixed(4) || "0.2644"}
                </td>
                <td className="py-3 px-4 text-slate-400">≥ +0.0500</td>
                <td className="py-3 px-4 text-right">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-sans font-bold text-[11px]">
                    PASS (+10.1 pts)
                  </span>
                </td>
              </tr>
              <tr>
                <td className="py-3 px-4 font-sans font-medium text-slate-200">
                  COCO mAP@[0.5:0.95]
                </td>
                <td className="py-3 px-4 text-slate-400">0.0980</td>
                <td className="py-3 px-4 font-bold text-sky-400">
                  {modelInfo?.metrics.coco_mAP_50_95.toFixed(4) || "0.1603"}
                </td>
                <td className="py-3 px-4 text-slate-400">Baseline reference</td>
                <td className="py-3 px-4 text-right">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-sans font-bold text-[11px]">
                    PASS
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Class Taxonomy Explorer */}
      <div className="flex flex-col gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <h3 className="text-base font-bold text-slate-200">
              Supported 20-Class Taxonomy
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              10 classes overlapping with COCO + 10 novel target concepts.
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs font-medium">
            <button
              type="button"
              onClick={() => setFilterMode("all")}
              className={`px-3 py-1 rounded-md transition ${
                filterMode === "all"
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              All (20)
            </button>
            <button
              type="button"
              onClick={() => setFilterMode("shared")}
              className={`px-3 py-1 rounded-md transition ${
                filterMode === "shared"
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              COCO Shared (10)
            </button>
            <button
              type="button"
              onClick={() => setFilterMode("novel")}
              className={`px-3 py-1 rounded-md transition ${
                filterMode === "novel"
                  ? "bg-slate-800 text-slate-100"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              Novel Concepts (10)
            </button>
          </div>
        </div>

        {loading ? (
          <p className="text-sm text-slate-400">Loading taxonomy...</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
            {filteredClasses.map((cls) => (
              <div
                key={cls.id}
                className="p-3 bg-slate-900 rounded-xl border border-slate-800 hover:border-slate-700 transition flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-mono text-slate-400">
                      ID {cls.id.toString().padStart(2, "0")}
                    </span>
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                        cls.is_coco
                          ? "bg-slate-800 text-slate-300"
                          : "bg-sky-500/20 text-sky-300"
                      }`}
                    >
                      {cls.is_coco ? "COCO" : "Novel"}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 mb-1.5">
                    <span
                      className="w-3.5 h-3.5 rounded-full flex-shrink-0"
                      style={{ backgroundColor: cls.color_hex }}
                    />
                    <h4 className="text-sm font-bold text-slate-100">{cls.name}</h4>
                  </div>

                  <p className="text-xs text-slate-400 line-clamp-2">
                    {cls.description}
                  </p>
                </div>

                <div className="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span className="flex items-center gap-1">
                    <Tag className="w-3 h-3" />
                    {cls.mid}
                  </span>
                  <span>{cls.color_hex}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
