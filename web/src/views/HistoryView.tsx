import React from "react";
import { Clock, Trash2, ArrowRight } from "lucide-react";
import type { HistoryRecord } from "../types";

interface HistoryViewProps {
  history: HistoryRecord[];
  onSelectRecord: (record: HistoryRecord) => void;
  onClearHistory: () => void;
}

export const HistoryView: React.FC<HistoryViewProps> = ({
  history,
  onSelectRecord,
  onClearHistory,
}) => {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div>
          <h2 className="text-lg font-bold text-slate-100">Session Detection History</h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Past image inferences during this browser session.
          </p>
        </div>
        {history.length > 0 && (
          <button
            type="button"
            onClick={onClearHistory}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-red-950/40 hover:bg-red-900/60 border border-red-800/80 text-red-300 text-xs font-medium transition"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear All</span>
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="text-center py-20 bg-slate-900/40 rounded-xl border border-slate-800/80">
          <Clock className="w-10 h-10 text-slate-600 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-300">No detection history yet</p>
          <p className="text-xs text-slate-400 mt-1">
            Uploaded images or webcam frames will appear here.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {history.map((item) => (
            <div
              key={item.id}
              onClick={() => onSelectRecord(item)}
              className="group bg-slate-900 rounded-xl border border-slate-800 hover:border-sky-500 overflow-hidden cursor-pointer transition flex flex-col"
            >
              <div className="relative aspect-video bg-slate-950 overflow-hidden">
                <img
                  src={item.previewUrl}
                  alt="Detection preview"
                  className="w-full h-full object-cover group-hover:scale-105 transition duration-300"
                />
                <span className="absolute bottom-2 right-2 px-2 py-0.5 bg-slate-950/80 backdrop-blur rounded text-[11px] font-mono text-sky-400 border border-slate-800">
                  {item.result.inference_time_ms} ms
                </span>
              </div>

              <div className="p-3.5 flex flex-col gap-2 flex-grow justify-between">
                <div>
                  <div className="flex items-center justify-between text-xs text-slate-400">
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {item.timestamp}
                    </span>
                    <span className="font-semibold text-emerald-400">
                      {item.result.total_detections} detected
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-1 mt-2">
                    {item.result.detections.slice(0, 3).map((d, i) => (
                      <span
                        key={i}
                        className="text-[10px] px-1.5 py-0.5 rounded font-medium text-slate-200"
                        style={{ backgroundColor: `${d.color_hex}33`, borderColor: d.color_hex }}
                      >
                        {d.class_name}
                      </span>
                    ))}
                    {item.result.detections.length > 3 && (
                      <span className="text-[10px] px-1 py-0.5 text-slate-400">
                        +{item.result.detections.length - 3}
                      </span>
                    )}
                  </div>
                </div>

                <div className="flex items-center justify-end text-xs font-semibold text-sky-400 group-hover:text-sky-300 pt-2 border-t border-slate-800/80">
                  <span>Inspect</span>
                  <ArrowRight className="w-3.5 h-3.5 ml-1 transition-transform group-hover:translate-x-0.5" />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

