import React from "react";
import { cn } from "../../lib/utils";

export function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-control bg-elevated/70 border border-border-subtle/50",
        className
      )}
      {...props}
    />
  );
}

