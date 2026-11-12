import React from "react";
import * as RadixTabs from "@radix-ui/react-tabs";
import { cn } from "../../lib/utils";

export const Tabs = RadixTabs.Root;

export interface TabsListProps
  extends React.ComponentPropsWithoutRef<typeof RadixTabs.List> {
  variant?: "underline" | "pill";
}

export const TabsList = React.forwardRef<
  React.ElementRef<typeof RadixTabs.List>,
  TabsListProps
>(({ className, variant = "underline", ...props }, ref) => (
  <RadixTabs.List
    ref={ref}
    className={cn(
      "flex items-center gap-1 select-none",
      variant === "underline" && "border-b border-border-subtle w-full",
      variant === "pill" && "p-1 rounded-control bg-elevated border border-border-subtle",
      className
    )}
    {...props}
  />
));
TabsList.displayName = RadixTabs.List.displayName;

export interface TabsTriggerProps
  extends React.ComponentPropsWithoutRef<typeof RadixTabs.Trigger> {
  variant?: "underline" | "pill";
}

export const TabsTrigger = React.forwardRef<
  React.ElementRef<typeof RadixTabs.Trigger>,
  TabsTriggerProps
>(({ className, variant = "underline", children, ...props }, ref) => (
  <RadixTabs.Trigger
    ref={ref}
    className={cn(
      "inline-flex items-center justify-center font-medium transition-all duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2 focus-visible:ring-offset-canvas disabled:pointer-events-none disabled:opacity-50 text-xs sm:text-sm",
      variant === "underline" &&
        "px-3.5 py-2 -mb-px border-b-2 border-transparent text-text-secondary hover:text-text-primary data-[state=active]:border-brand data-[state=active]:text-text-primary font-medium",
      variant === "pill" &&
        "px-3 py-1.5 rounded-[4px] text-text-secondary hover:text-text-primary data-[state=active]:bg-card data-[state=active]:text-text-primary data-[state=active]:shadow-sm border border-transparent data-[state=active]:border-border-subtle",
      className
    )}
    {...props}
  >
    {children}
  </RadixTabs.Trigger>
));
TabsTrigger.displayName = RadixTabs.Trigger.displayName;

export const TabsContent = React.forwardRef<
  React.ElementRef<typeof RadixTabs.Content>,
  React.ComponentPropsWithoutRef<typeof RadixTabs.Content>
>(({ className, ...props }, ref) => (
  <RadixTabs.Content
    ref={ref}
    className={cn(
      "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand focus-visible:ring-offset-2",
      className
    )}
    {...props}
  />
));
TabsContent.displayName = RadixTabs.Content.displayName;

