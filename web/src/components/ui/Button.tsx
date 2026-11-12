import React from "react";
import { Loader2 } from "lucide-react";
import { cn } from "../../lib/utils";

export type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";
export type ButtonSize = "sm" | "md" | "lg";

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  isLoading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = "secondary",
      size = "md",
      isLoading = false,
      leftIcon,
      rightIcon,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const baseStyles =
      "inline-flex items-center justify-center font-medium rounded-control transition-colors duration-150 select-none disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-canvas";

    const variantStyles: Record<ButtonVariant, string> = {
      primary:
        "bg-brand text-text-inverse hover:bg-brand-hover shadow-sm border border-transparent",
      secondary:
        "bg-card text-text-primary border border-border-subtle hover:bg-elevated hover:border-border-muted shadow-sm",
      ghost:
        "bg-transparent text-text-secondary hover:text-text-primary hover:bg-elevated border border-transparent",
      danger:
        "bg-status-danger text-white hover:opacity-90 shadow-sm border border-transparent",
    };

    const sizeStyles: Record<ButtonSize, string> = {
      sm: "h-7 px-2.5 text-xs gap-1.5",
      md: "h-8 px-3.5 text-sm gap-2",
      lg: "h-10 px-4 text-base gap-2.5",
    };

    return (
      <button
        ref={ref}
        disabled={disabled || isLoading}
        className={cn(baseStyles, variantStyles[variant], sizeStyles[size], className)}
        {...props}
      >
        {isLoading ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin" />
        ) : (
          leftIcon
        )}
        <span>{children}</span>
        {!isLoading && rightIcon}
      </button>
    );
  }
);

Button.displayName = "Button";

