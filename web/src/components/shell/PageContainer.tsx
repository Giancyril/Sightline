import React from "react";
import { cn } from "../../lib/utils";

interface PageContainerProps {
  children: React.ReactNode;
  className?: string;
}

/**
 * Uniform page wrapper: max-1280px, 24px gutters, consistent top padding.
 * All page bodies are wrapped in this to prevent header shift.
 */
export const PageContainer: React.FC<PageContainerProps> = ({
  children,
  className,
}) => (
  <div
    className={cn(
      "max-w-container mx-auto w-full px-6 py-8",
      className
    )}
  >
    {children}
  </div>
);

interface PageHeaderProps {
  title: string;
  description?: string;
  actions?: React.ReactNode;
  className?: string;
}

/**
 * Consistent page title block: title + one-line description + optional action slot.
 */
export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  description,
  actions,
  className,
}) => (
  <div
    className={cn(
      "flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 mb-7",
      className
    )}
  >
    <div className="min-w-0">
      <h1 className="text-lg font-semibold tracking-tight text-text-primary leading-snug">
        {title}
      </h1>
      {description && (
        <p className="text-xs text-text-tertiary mt-0.5 leading-relaxed">
          {description}
        </p>
      )}
    </div>
    {actions && (
      <div className="flex items-center gap-2 shrink-0">{actions}</div>
    )}
  </div>
);

