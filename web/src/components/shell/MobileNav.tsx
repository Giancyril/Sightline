import React from "react";
import { Menu, Scan, History, Cpu, Terminal, Palette } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "../ui/DropdownMenu";
import type { NavTab } from "./CommandPalette";

interface MobileNavProps {
  activeTab: NavTab;
  onSelectTab: (tab: NavTab) => void;
  historyCount: number;
}

export const MobileNav: React.FC<MobileNavProps> = ({
  activeTab,
  onSelectTab,
  historyCount,
}) => {
  const items: Array<{
    id: NavTab;
    label: string;
    icon: React.ReactNode;
    count?: number;
  }> = [
    { id: "detect", label: "Detect", icon: <Scan className="w-4 h-4" /> },
    {
      id: "history",
      label: "History",
      icon: <History className="w-4 h-4" />,
      count: historyCount,
    },
    { id: "model", label: "Model", icon: <Cpu className="w-4 h-4" /> },
    { id: "api", label: "API", icon: <Terminal className="w-4 h-4" /> },
  ];

  return (
    <div className="flex min-[720px]:hidden">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            className="flex h-7 w-7 items-center justify-center rounded-control border border-border-subtle bg-elevated text-text-secondary hover:text-text-primary hover:bg-subtle focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
            aria-label="Toggle navigation menu"
          >
            <Menu className="w-4 h-4" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuLabel>Navigation</DropdownMenuLabel>
          <DropdownMenuSeparator />
          {items.map((item) => {
            const isActive = activeTab === item.id;
            return (
              <DropdownMenuItem
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={isActive ? "bg-brand-subtle text-brand font-medium" : ""}
              >
                <span className="mr-2 opacity-75">{item.icon}</span>
                <span className="flex-1">{item.label}</span>
                {item.count !== undefined && item.count > 0 && (
                  <span className="ml-auto font-mono text-[10px] px-1 rounded-chip bg-elevated border border-border-subtle">
                    {item.count}
                  </span>
                )}
              </DropdownMenuItem>
            );
          })}
          <DropdownMenuSeparator />
          <DropdownMenuItem
            onClick={() => onSelectTab("dev")}
            className={activeTab === "dev" ? "bg-brand-subtle text-brand font-medium" : "text-text-tertiary"}
          >
            <Palette className="w-4 h-4 mr-2 opacity-75" />
            <span>Components</span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};
