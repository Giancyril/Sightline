import React from "react";
import { cn } from "../../lib/utils";

export interface KbdProps extends React.HTMLAttributes<HTMLElement> {}

export const Kbd: React.FC<KbdProps> = ({ className, children, ...props }) => {
  return (
    <kbd
      className={cn(
        "inline-flex items-center justify-center font-mono text-[10px] font-medium h-5 min-w-[20px] px-1.5 rounded-chip bg-elevated border border-border-muted text-text-secondary select-none shadow-[inset_0_-1px_0_rgba(0,0,0,0.1)]",
        className
      )}
      {...props}
    >
      {children}
    </kbd>
  );
};

