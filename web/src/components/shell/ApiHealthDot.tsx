import React from "react";
import { Tooltip } from "../ui/Tooltip";
import { cn } from "../../lib/utils";
import type { ApiHealth } from "../../hooks/useApiHealth";

interface ApiHealthDotProps {
  health: ApiHealth;
  className?: string;
}

const STATUS_CONFIG = {
  online: {
    color: "bg-status-success",
    pulse: true,
    label: "API online",
    detail: (ms: number | null) => (ms !== null ? `${ms} ms` : ""),
  },
  slow: {
    color: "bg-status-warning",
    pulse: true,
    label: "API responding slowly",
    detail: (ms: number | null) => (ms !== null ? `${ms} ms` : ""),
  },
  offline: {
    color: "bg-status-danger",
    pulse: false,
    label: "API offline",
    detail: () => "Cannot reach backend",
  },
  checking: {
    color: "bg-border-muted",
    pulse: false,
    label: "Checking API…",
    detail: () => "",
  },
} as const;

export const ApiHealthDot: React.FC<ApiHealthDotProps> = ({
  health,
  className,
}) => {
  const cfg = STATUS_CONFIG[health.status];
  const detail = cfg.detail(health.latencyMs);
  const tooltipContent = (
    <span>
      {cfg.label}
      {detail && (
        <>
          {" "}
          <span className="font-mono opacity-70">{detail}</span>
        </>
      )}
    </span>
  );

  return (
    <Tooltip content={tooltipContent} side="bottom">
      <button
        type="button"
        aria-label={`${cfg.label}${detail ? ` — ${detail}` : ""}`}
        className={cn(
          "relative flex h-6 w-6 items-center justify-center rounded-control focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
          className
        )}
      >
        {cfg.pulse && (
          <span
            className={cn(
              "absolute inline-flex h-2.5 w-2.5 rounded-full opacity-60 animate-ping",
              cfg.color
            )}
          />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", cfg.color)} />
      </button>
    </Tooltip>
  );
};

