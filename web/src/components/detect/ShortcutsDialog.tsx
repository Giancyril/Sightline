import React from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "../ui/Dialog";
import { Kbd } from "../ui/Kbd";

interface ShortcutsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const SHORTCUT_GROUPS = [
  {
    title: "Workspace & Navigation",
    shortcuts: [
      { key: "U", description: "Open file picker to upload image" },
      { key: "Space", description: "Toggle label chips on bounding boxes" },
      { key: "B", description: "Toggle bounding box outlines" },
      { key: "F", description: "Toggle fullscreen canvas view" },
      { key: "Esc", description: "Clear active selection or pinned detection" },
      { key: "?", description: "Show keyboard shortcuts dialog" },
    ],
  },
  {
    title: "Detection Inspection",
    shortcuts: [
      { key: "↑ / ↓", description: "Step through detections in inspector" },
      { key: "↵", description: "Pin or inspect selected detection" },
      { key: "⌘V / Ctrl+V", description: "Paste image directly from clipboard" },
      { key: "⌘K / Ctrl+K", description: "Open global command palette" },
    ],
  },
];

export const ShortcutsDialog: React.FC<ShortcutsDialogProps> = ({
  open,
  onOpenChange,
}) => {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Keyboard Shortcuts</DialogTitle>
          <DialogDescription>
            Quick hotkeys to navigate, inspect, and operate the detection workspace.
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-5 py-2">
          {SHORTCUT_GROUPS.map((group) => (
            <div key={group.title} className="flex flex-col gap-2">
              <h4 className="text-[11px] font-semibold uppercase tracking-wider text-text-tertiary">
                {group.title}
              </h4>
              <div className="flex flex-col rounded-control border border-border-subtle bg-elevated/40 divide-y divide-border-subtle overflow-hidden">
                {group.shortcuts.map((sc) => (
                  <div
                    key={sc.key}
                    className="flex items-center justify-between px-3 py-2 text-xs"
                  >
                    <span className="text-text-secondary">{sc.description}</span>
                    <Kbd>{sc.key}</Kbd>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </DialogContent>
    </Dialog>
  );
};

