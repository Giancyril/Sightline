import React, { useState, useCallback } from "react";
import { CheckCircle2, AlertCircle, Info, X } from "lucide-react";
import { ToastContext, type ToastMessage } from "./ToastContext";

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const toast = useCallback(({ title, description, type = "info" }: Omit<ToastMessage, "id">) => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, title, description, type }]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const removeToast = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-sm w-full pointer-events-none">
        {toasts.map((t) => (
          <div
            key={t.id}
            className="pointer-events-auto flex items-start gap-3 p-3.5 bg-card border border-border-subtle rounded-panel shadow-lg animate-in slide-in-from-bottom-2 fade-in duration-200"
          >
            {t.type === "success" && (
              <CheckCircle2 className="w-4 h-4 text-status-success flex-shrink-0 mt-0.5" />
            )}
            {t.type === "error" && (
              <AlertCircle className="w-4 h-4 text-status-danger flex-shrink-0 mt-0.5" />
            )}
            {t.type === "info" && (
              <Info className="w-4 h-4 text-brand flex-shrink-0 mt-0.5" />
            )}

            <div className="flex-1 min-w-0">
              <p className="text-xs font-semibold text-text-primary leading-tight">
                {t.title}
              </p>
              {t.description && (
                <p className="text-[11px] text-text-secondary mt-0.5 leading-snug">
                  {t.description}
                </p>
              )}
            </div>

            <button
              type="button"
              onClick={() => removeToast(t.id)}
              className="text-text-tertiary hover:text-text-primary p-0.5 rounded-control transition-colors"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

export type { ToastType, ToastMessage } from "./ToastContext";
