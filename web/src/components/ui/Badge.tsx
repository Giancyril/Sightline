import React from "react";
import { cn } from "../../lib/utils";

export type BadgeVariant =
  | "neutral"
  | "brand"
  | "success"
  | "danger"
  | "warning";

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  mono?: boolean;
  dot?: boolean;
  dotColor?: string;
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = "neutral",
  mono = false,
  dot = false,
  dotColor,
  children,
  ...props
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    neutral:
      "bg-elevated text-text-secondary border-border-subtle",
    brand:
      "bg-brand-subtle text-brand border-brand/20",
    success:
      "bg-status-success-subtle text-status-success border-status-success/20",
    danger:
      "bg-status-danger-subtle text-status-danger border-status-danger/20",
    warning:
      "bg-status-warning-subtle text-status-warning border-status-warning/20",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 px-2 py-0.5 rounded-chip text-[11px] font-medium border select-none leading-none",
        mono && "font-mono tabular-nums",
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {dot && (
        <span
          className="w-1.5 h-1.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: dotColor || "currentColor" }}
        />
      )}
      <span>{children}</span>
    </span>
  );
};

