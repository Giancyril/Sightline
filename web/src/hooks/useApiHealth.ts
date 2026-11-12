import { useState, useEffect } from "react";

export type ApiHealthStatus = "online" | "slow" | "offline" | "checking";

export interface ApiHealth {
  status: ApiHealthStatus;
  latencyMs: number | null;
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const POLL_INTERVAL = 30_000; // 30s
const SLOW_THRESHOLD = 800; // ms

export function useApiHealth(): ApiHealth {
  const [health, setHealth] = useState<ApiHealth>({
    status: "checking",
    latencyMs: null,
  });

  useEffect(() => {
    let mounted = true;

    const runCheck = async () => {
      const start = performance.now();
      try {
        const res = await fetch(`${API_BASE}/health`, {
          signal: AbortSignal.timeout(5000),
        });
        const latencyMs = Math.round(performance.now() - start);
        if (mounted) {
          if (res.ok) {
            setHealth({
              status: latencyMs > SLOW_THRESHOLD ? "slow" : "online",
              latencyMs,
            });
          } else {
            setHealth({ status: "offline", latencyMs: null });
          }
        }
      } catch {
        if (mounted) {
          setHealth({ status: "offline", latencyMs: null });
        }
      }
    };

    void runCheck();
    const id = setInterval(runCheck, POLL_INTERVAL);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return health;
}
