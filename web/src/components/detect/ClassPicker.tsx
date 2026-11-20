import React, { useState, useRef, useEffect, useMemo } from "react";
import { Search, Check, ChevronDown, Tag } from "lucide-react";
import { CLASS_PALETTE, colorForClass, type ClassColor } from "../../lib/colors.ts";
import { cn } from "../../lib/utils";

export interface ClassPickerProps {
  selectedClassId: number;
  onSelectClass: (classId: number, className: string) => void;
  className?: string;
  size?: "sm" | "md";
}

export const ClassPicker: React.FC<ClassPickerProps> = ({
  selectedClassId,
  onSelectClass,
  className,
  size = "md",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const dropdownRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const selectedClass = useMemo(
    () => colorForClass(selectedClassId),
    [selectedClassId]
  );

  const filteredClasses = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return CLASS_PALETTE;
    return CLASS_PALETTE.filter((c: ClassColor) =>
      c.name.toLowerCase().includes(q) || String(c.id).includes(q)
    );
  }, [query]);

  const closeDropdown = () => {
    setIsOpen(false);
    setQuery("");
  };

  // Focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  // Click outside to close
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(e.target as Node)
      ) {
        closeDropdown();
      }
    };
    if (isOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  // Keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      closeDropdown();
    }
  };

  return (
    <div
      ref={dropdownRef}
      className={cn("relative inline-block text-left", className)}
      onKeyDown={handleKeyDown}
    >
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label={`Select class: currently ${selectedClass.name}`}
        className={cn(
          "flex items-center gap-1.5 rounded-control border border-border-subtle bg-card hover:bg-elevated transition-colors text-text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand select-none",
          size === "sm" ? "px-2 py-1 text-xs" : "px-2.5 py-1.5 text-sm"
        )}
      >
        <span
          className="w-2.5 h-2.5 rounded-full shrink-0"
          style={{ backgroundColor: selectedClass.hex }}
          aria-hidden="true"
        />
        <span className="font-medium truncate max-w-[120px]">
          {selectedClass.name}
        </span>
        <ChevronDown
          className={cn(
            "w-3.5 h-3.5 text-text-tertiary transition-transform shrink-0",
            isOpen && "rotate-180"
          )}
        />
      </button>

      {isOpen && (
        <div
          role="listbox"
          aria-label="Available detection classes"
          className="absolute left-0 mt-1.5 w-60 max-h-72 z-50 rounded-control border border-border-subtle bg-card/95 backdrop-blur-md shadow-lg flex flex-col overflow-hidden animate-in fade-in-0 zoom-in-95 duration-100"
        >
          {/* Search Header */}
          <div className="p-2 border-b border-border-subtle flex items-center gap-1.5 bg-canvas/50">
            <Search className="w-3.5 h-3.5 text-text-tertiary shrink-0" />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search classes..."
              aria-label="Search classes"
              className="w-full bg-transparent text-xs text-text-primary placeholder:text-text-tertiary focus:outline-none"
            />
          </div>

          {/* Class Options */}
          <div className="overflow-y-auto max-h-56 p-1 space-y-0.5">
            {filteredClasses.length === 0 ? (
              <div className="p-3 text-center text-xs text-text-tertiary">
                No classes matching &quot;{query}&quot;
              </div>
            ) : (
              filteredClasses.map((item: ClassColor) => {
                const isSelected = item.id === selectedClassId;
                return (
                  <button
                    key={item.id}
                    type="button"
                    role="option"
                    aria-selected={isSelected}
                    onClick={() => {
                      onSelectClass(item.id, item.name);
                      closeDropdown();
                    }}
                    className={cn(
                      "w-full flex items-center justify-between px-2 py-1.5 rounded-[4px] text-xs text-left transition-colors",
                      isSelected
                        ? "bg-brand/10 text-brand font-medium"
                        : "text-text-primary hover:bg-elevated"
                    )}
                  >
                    <div className="flex items-center gap-2 truncate">
                      <span
                        className="w-2.5 h-2.5 rounded-full shrink-0"
                        style={{ backgroundColor: item.hex }}
                        aria-hidden="true"
                      />
                      <span className="truncate">{item.name}</span>
                    </div>
                    <div className="flex items-center gap-1.5 shrink-0 ml-2">
                      <span className="text-[10px] font-mono text-text-tertiary">
                        #{item.id}
                      </span>
                      {isSelected ? (
                        <Check className="w-3.5 h-3.5 text-brand" />
                      ) : (
                        <span className="w-3.5" />
                      )}
                    </div>
                  </button>
                );
              })
            )}
          </div>

          {/* Quick Footer */}
          <div className="px-2 py-1 border-t border-border-subtle bg-canvas/30 flex items-center justify-between text-[10px] text-text-tertiary">
            <span>20 classes</span>
            <span className="flex items-center gap-1">
              <Tag className="w-2.5 h-2.5" /> Okabe-Ito
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
