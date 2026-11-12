import React, { useEffect, useState } from "react";
import { Terminal, Shield, Sparkles } from "lucide-react";
import { cn } from "../../lib/utils";

interface NavSection {
  id: string;
  label: string;
  method?: "POST" | "GET";
  badge?: string;
}

const SECTIONS: NavSection[] = [
  { id: "overview", label: "Overview" },
  { id: "authentication", label: "Base URL & Headers" },
  { id: "detect", label: "/v1/detect", method: "POST" },
  { id: "classes", label: "/v1/classes", method: "GET" },
  { id: "model", label: "/v1/model", method: "GET" },
  { id: "health", label: "/healthz", method: "GET" },
  { id: "try-it", label: "Interactive Console", badge: "Try It" },
];

export const ApiNavSidebar: React.FC = () => {
  const [activeId, setActiveId] = useState<string>("overview");

  useEffect(() => {
    const handleScroll = () => {
      const scrollY = window.scrollY;
      const headers = SECTIONS.map((sec) => {
        const el = document.getElementById(sec.id);
        return {
          id: sec.id,
          top: el ? el.offsetTop - 120 : 0,
        };
      });

      for (let i = headers.length - 1; i >= 0; i--) {
        if (scrollY >= headers[i].top) {
          setActiveId(headers[i].id);
          break;
        }
      }
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <nav
      aria-label="API Documentation Navigation"
      className="sticky top-20 flex flex-col gap-1 w-52 shrink-0 select-none text-xs"
    >
      <div className="text-[11px] font-semibold text-text-muted uppercase tracking-wider px-2 py-1 mb-1">
        API Reference
      </div>

      <div className="flex flex-col gap-0.5">
        {SECTIONS.map((sec) => {
          const isActive = activeId === sec.id;
          return (
            <button
              key={sec.id}
              type="button"
              onClick={() => scrollToSection(sec.id)}
              className={cn(
                "flex items-center justify-between w-full px-2.5 py-1.5 rounded-control text-left transition-colors font-medium",
                isActive
                  ? "bg-elevated text-brand border border-border-subtle font-semibold"
                  : "text-text-secondary hover:text-text-primary hover:bg-subtle/70"
              )}
            >
              <div className="flex items-center gap-1.5 truncate">
                {sec.method && (
                  <span
                    className={cn(
                      "font-mono text-[9px] font-bold px-1 py-0.2 rounded-[3px] uppercase border",
                      sec.method === "POST"
                        ? "bg-subtle text-text-primary border-border-default"
                        : "bg-subtle text-text-secondary border-border-subtle"
                    )}
                  >
                    {sec.method}
                  </span>
                )}
                {sec.id === "try-it" && (
                  <Sparkles className="w-3 h-3 text-brand shrink-0" />
                )}
                {sec.id === "overview" && (
                  <Terminal className="w-3 h-3 text-text-muted shrink-0" />
                )}
                {sec.id === "authentication" && (
                  <Shield className="w-3 h-3 text-text-muted shrink-0" />
                )}
                <span className="truncate">{sec.label}</span>
              </div>

              {sec.badge && (
                <span className="text-[10px] font-mono text-brand bg-brand-subtle px-1.5 py-0.2 rounded-chip shrink-0">
                  {sec.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
};
