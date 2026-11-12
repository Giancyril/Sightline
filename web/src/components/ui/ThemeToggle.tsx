import React from "react";
import { Sun, Moon, Laptop } from "lucide-react";
import { useTheme, type Theme } from "../../lib/theme";
import { cn } from "../../lib/utils";

export const ThemeToggle: React.FC<{ className?: string }> = ({ className }) => {
  const { theme, setTheme } = useTheme();

  const options: { value: Theme; label: string; icon: React.ReactNode }[] = [
    { value: "light", label: "Light", icon: <Sun className="w-3.5 h-3.5" /> },
    { value: "dark", label: "Dark", icon: <Moon className="w-3.5 h-3.5" /> },
    { value: "system", label: "System", icon: <Laptop className="w-3.5 h-3.5" /> },
  ];

  return (
    <div
      className={cn(
        "inline-flex items-center p-0.5 rounded-control bg-elevated border border-border-subtle",
        className
      )}
    >
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => setTheme(opt.value)}
          title={`Theme: ${opt.label}`}
          aria-label={`Theme: ${opt.label}`}
          className={cn(
            "p-1.5 rounded-[4px] text-xs font-medium transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand",
            theme === opt.value
              ? "bg-card text-text-primary shadow-sm border border-border-subtle"
              : "text-text-tertiary hover:text-text-primary border border-transparent"
          )}
        >
          {opt.icon}
        </button>
      ))}
    </div>
  );
};

