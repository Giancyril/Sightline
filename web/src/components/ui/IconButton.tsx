import React from "react";
import { cn } from "../../lib/utils";

export interface IconButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  size?: "sm" | "md" | "lg";
  variant?: "primary" | "secondary" | "ghost" | "danger";
  "aria-label": string;
}

export const IconButton = React.forwardRef<HTMLButtonElement, IconButtonProps>(
  (
    {
      className,
      size = "md",
      variant = "ghost",
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center rounded-control transition-colors duration-150 select-none disabled:opacity-50 disabled:cursor-not-allowed focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-canvas";

    const variantStyles = {
      primary: "bg-brand text-text-inverse hover:bg-brand-hover shadow-sm",
      secondary: "bg-card text-text-primary border border-border-subtle hover:bg-elevated hover:border-border-muted",
      ghost: "bg-transparent text-text-secondary hover:text-text-primary hover:bg-elevated",
      danger: "bg-transparent text-status-danger hover:bg-status-danger-subtle",
    };

    const sizeStyles = {
      sm: "w-7 h-7 text-xs [&>svg]:w-3.5 [&>svg]:h-3.5",
      md: "w-8 h-8 text-sm [&>svg]:w-4 [&>svg]:h-4",
      lg: "w-10 h-10 text-base [&>svg]:w-5 [&>svg]:h-5",
    };

    return (
      <button
        ref={ref}
        disabled={disabled}
        className={cn(baseStyles, variantStyles[variant], sizeStyles[size], className)}
        {...props}
      >
        {children}
      </button>
    );
  }
);

IconButton.displayName = "IconButton";

