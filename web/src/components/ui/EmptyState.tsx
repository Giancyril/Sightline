import React from "react";
import { cn } from "../../lib/utils";

export interface EmptyStateProps {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  icon,
  title,
  description,
  action,
  className,
}) => {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center p-8 rounded-panel border border-dashed border-border-subtle bg-subtle/50 select-none",
        className
      )}
    >
      {icon && (
        <div className="mb-3 text-text-tertiary flex items-center justify-center p-2 rounded-control bg-elevated border border-border-subtle">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-text-primary mb-1">
        {title}
      </h3>
      {description && (
        <p className="text-xs text-text-secondary max-w-sm mb-4 leading-relaxed">
          {description}
        </p>
      )}
      {action && <div className="mt-1">{action}</div>}
    </div>
  );
};

