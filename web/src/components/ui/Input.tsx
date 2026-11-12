import React from "react";
import { cn } from "../../lib/utils";

export interface InputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  leftAddon?: React.ReactNode;
  rightAddon?: React.ReactNode;
  error?: boolean;
}

export const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, leftAddon, rightAddon, error, disabled, ...props }, ref) => {
    return (
      <div
        className={cn(
          "flex items-center h-8 w-full rounded-control border bg-card px-2.5 text-sm transition-colors duration-150",
          error
            ? "border-status-danger focus-within:ring-2 focus-within:ring-status-danger focus-within:ring-offset-1"
            : "border-border-muted focus-within:border-border-strong focus-within:ring-2 focus-within:ring-brand/40 focus-within:ring-offset-1 focus-within:ring-offset-canvas",
          disabled && "opacity-50 cursor-not-allowed bg-subtle",
          className
        )}
      >
        {leftAddon && (
          <span className="text-text-tertiary mr-2 select-none text-xs flex items-center">
            {leftAddon}
          </span>
        )}
        <input
          ref={ref}
          disabled={disabled}
          className="w-full bg-transparent text-text-primary placeholder:text-text-tertiary focus:outline-none disabled:cursor-not-allowed font-sans text-sm"
          {...props}
        />
        {rightAddon && (
          <span className="text-text-tertiary ml-2 select-none text-xs flex items-center font-mono">
            {rightAddon}
          </span>
        )}
      </div>
    );
  }
);

Input.displayName = "Input";

