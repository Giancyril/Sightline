import React, { useState } from "react";
import {
  Scan,
  Download,
  Copy,
  Trash2,
  HelpCircle,
  MoreVertical,
  Plus,
  Terminal,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { IconButton } from "../components/ui/IconButton";
import { Input } from "../components/ui/Input";
import { Slider } from "../components/ui/Slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "../components/ui/Tabs";
import { Badge } from "../components/ui/Badge";
import { Tooltip } from "../components/ui/Tooltip";
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "../components/ui/Dialog";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "../components/ui/DropdownMenu";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "../components/ui/Table";
import { useToast } from "../components/ui/useToast";
import { Skeleton } from "../components/ui/Skeleton";
import { Kbd } from "../components/ui/Kbd";
import { EmptyState } from "../components/ui/EmptyState";
import { ThemeToggle } from "../components/ui/ThemeToggle";
import { CLASS_PALETTE, colorForClass } from "../lib/colors";
import { useTheme } from "../lib/theme";

export const DevComponentsView: React.FC = () => {
  const { resolvedTheme } = useTheme();
  const { toast } = useToast();

  const [sliderVal, setSliderVal] = useState<number>(0.25);
  const [sortField, setSortField] = useState<string>("name");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");

  const tableData = [
    { id: 1, name: "Person", count: 42, score: 0.942, latency: 142.3 },
    { id: 2, name: "Laptop", count: 18, score: 0.887, latency: 138.6 },
    { id: 9, name: "Car", count: 24, score: 0.912, latency: 154.1 },
    { id: 11, name: "Guitar", count: 7, score: 0.745, latency: 148.0 },
  ];

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("asc");
    }
  };

  return (
    <div className="max-w-[1280px] mx-auto px-4 sm:px-6 py-8 flex flex-col gap-10">
      {/* Dev Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border-subtle pb-6">
        <div>
          <div className="flex items-center gap-2.5">
            <h1 className="text-xl font-semibold tracking-tight text-text-primary">
              Design System & Component Showcase
            </h1>
            <Badge variant="brand" mono>
              /dev/components
            </Badge>
          </div>
          <p className="text-xs text-text-secondary mt-1">
            Standardized UI tokens, interactive states, accessibility focus rings, and Okabe-Ito color taxonomy.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Badge variant="neutral" mono>
            Active Theme: {resolvedTheme}
          </Badge>
          <ThemeToggle />
        </div>
      </div>

      {/* 1. Categorical Color Palette */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            1. Categorical Class Palette (Okabe-Ito & Carbon 20 Colors)
          </h2>
          <p className="text-xs text-text-secondary mt-0.5">
            Color-blind accessible sequence assigned deterministically by Class ID (1–20).
          </p>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-10 gap-2">
          {CLASS_PALETTE.map((c) => (
            <div
              key={c.id}
              className="p-2.5 rounded-panel border border-border-subtle bg-card flex flex-col items-center text-center gap-1.5 shadow-sm"
            >
              <span
                className="w-5 h-5 rounded-full border border-border-subtle shadow-sm flex-shrink-0"
                style={{ backgroundColor: c.hex }}
              />
              <span className="text-[11px] font-medium text-text-primary truncate w-full">
                {c.name}
              </span>
              <span className="font-mono text-[10px] text-text-tertiary tabular-nums">
                #{c.id} · {c.hex}
              </span>
            </div>
          ))}
        </div>
      </section>

      {/* 2. Buttons & IconButtons */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            2. Buttons & IconButtons (Variants, Sizes & States)
          </h2>
          <p className="text-xs text-text-secondary mt-0.5">
            1px hairline borders, 6px radius controls, accessible focus rings.
          </p>
        </div>

        <div className="flex flex-col gap-4 p-5 rounded-panel border border-border-subtle bg-card">
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" size="md">
              Primary Action
            </Button>
            <Button variant="secondary" size="md" leftIcon={<Scan className="w-3.5 h-3.5" />}>
              Secondary Action
            </Button>
            <Button variant="ghost" size="md">
              Ghost Action
            </Button>
            <Button variant="danger" size="md" leftIcon={<Trash2 className="w-3.5 h-3.5" />}>
              Destructive
            </Button>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-border-subtle">
            <span className="text-xs text-text-tertiary w-24">Sizes:</span>
            <Button variant="secondary" size="sm">
              Small (28px)
            </Button>
            <Button variant="secondary" size="md">
              Medium (32px)
            </Button>
            <Button variant="secondary" size="lg">
              Large (40px)
            </Button>
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-border-subtle">
            <span className="text-xs text-text-tertiary w-24">States:</span>
            <Button variant="primary" size="md" isLoading>
              Loading
            </Button>
            <Button variant="secondary" size="md" disabled>
              Disabled State
            </Button>
            <div className="flex items-center gap-2">
              <IconButton aria-label="Download" variant="secondary" size="md">
                <Download className="w-4 h-4" />
              </IconButton>
              <IconButton aria-label="Copy code" variant="ghost" size="md">
                <Copy className="w-4 h-4" />
              </IconButton>
              <IconButton aria-label="Settings" variant="ghost" size="md" disabled>
                <MoreVertical className="w-4 h-4" />
              </IconButton>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Inputs & Sliders */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            3. Form Inputs & Precision Sliders
          </h2>
          <p className="text-xs text-text-secondary mt-0.5">
            Live numeric values, tick marks, prefix/suffix addons.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-col gap-3">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Text Inputs
            </h3>
            <Input placeholder="Standard placeholder..." />
            <Input
              defaultValue="0.25"
              leftAddon={<Terminal className="w-3.5 h-3.5" />}
              rightAddon="IoU"
            />
            <Input defaultValue="Invalid payload" error />
            <Input defaultValue="Disabled input" disabled />
          </div>

          <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-col gap-4">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Radix Range Sliders
            </h3>
            <Slider
              label="Confidence Threshold"
              value={sliderVal}
              min={0.05}
              max={0.95}
              step={0.05}
              onValueChange={setSliderVal}
            />
            <Slider
              label="Disabled Slider"
              value={0.45}
              disabled
              onValueChange={() => {}}
            />
          </div>
        </div>
      </section>

      {/* 4. Tabs & Navigation */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            4. Underline Tabs & Pill Switchers
          </h2>
        </div>

        <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-col gap-4">
          <Tabs defaultValue="tab1">
            <TabsList variant="underline">
              <TabsTrigger value="tab1" variant="underline">
                Overview
              </TabsTrigger>
              <TabsTrigger value="tab2" variant="underline">
                Benchmarks
              </TabsTrigger>
              <TabsTrigger value="tab3" variant="underline">
                Taxonomy
              </TabsTrigger>
              <TabsTrigger value="tab4" variant="underline" disabled>
                Disabled
              </TabsTrigger>
            </TabsList>
            <TabsContent value="tab1" className="pt-3 text-xs text-text-secondary">
              Overview tab content rendering cleanly without layout shift.
            </TabsContent>
            <TabsContent value="tab2" className="pt-3 text-xs text-text-secondary">
              Benchmark metrics comparison pane.
            </TabsContent>
            <TabsContent value="tab3" className="pt-3 text-xs text-text-secondary">
              20-class taxonomy reference pane.
            </TabsContent>
          </Tabs>
        </div>
      </section>

      {/* 5. Badges, Tooltips & Keyboard Shortcuts */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            5. Badges, Tooltips & Keyboard Shortcuts
          </h2>
        </div>

        <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-wrap items-center gap-4">
          <Badge variant="neutral">Neutral</Badge>
          <Badge variant="brand">Brand</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="danger">Danger</Badge>
          <Badge variant="neutral" mono dot dotColor={colorForClass(1).hex}>
            Person (42)
          </Badge>

          <div className="h-4 w-px bg-border-subtle" />

          <Tooltip content="Tooltip showing helpful contextual guidance">
            <Button variant="ghost" size="sm" leftIcon={<HelpCircle className="w-3.5 h-3.5" />}>
              Hover for Tooltip
            </Button>
          </Tooltip>

          <div className="h-4 w-px bg-border-subtle" />

          <div className="flex items-center gap-1.5 text-xs text-text-secondary">
            <span>Shortcut:</span>
            <Kbd>⌘</Kbd>
            <Kbd>K</Kbd>
          </div>
        </div>
      </section>

      {/* 6. Dense Technical Table */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            6. Dense Technical Data Table
          </h2>
        </div>

        <Table dense>
          <TableHeader>
            <TableRow>
              <TableHead
                sortable
                sortDirection={sortField === "id" ? sortDir : false}
                onSort={() => handleSort("id")}
              >
                ID
              </TableHead>
              <TableHead
                sortable
                sortDirection={sortField === "name" ? sortDir : false}
                onSort={() => handleSort("name")}
              >
                Class Name
              </TableHead>
              <TableHead>Detections</TableHead>
              <TableHead>Precision Score</TableHead>
              <TableHead>Latency</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {tableData.map((row) => (
              <TableRow key={row.id}>
                <TableCell mono>{row.id}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: colorForClass(row.id).hex }}
                    />
                    <span className="font-medium">{row.name}</span>
                  </div>
                </TableCell>
                <TableCell mono>{row.count}</TableCell>
                <TableCell mono>{(row.score * 100).toFixed(1)}%</TableCell>
                <TableCell mono>{row.latency.toFixed(1)} ms</TableCell>
                <TableCell className="text-right">
                  <Button variant="ghost" size="sm">
                    Inspect
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </section>

      {/* 7. Dialog, Dropdown & Toasts */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            7. Modals, Dropdowns & Toasts
          </h2>
        </div>

        <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-wrap items-center gap-4">
          {/* Dialog */}
          <Dialog>
            <DialogTrigger asChild>
              <Button variant="secondary" size="md">
                Open Dialog Modal
              </Button>
            </DialogTrigger>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Reset Detection Session</DialogTitle>
                <DialogDescription>
                  Are you sure you want to clear current detection buffers and history?
                </DialogDescription>
              </DialogHeader>
              <DialogFooter>
                <DialogClose asChild>
                  <Button variant="ghost" size="sm">
                    Cancel
                  </Button>
                </DialogClose>
                <Button variant="danger" size="sm">
                  Confirm Reset
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Dropdown Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="secondary" size="md" rightIcon={<MoreVertical className="w-3.5 h-3.5" />}>
                Dropdown Options
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuLabel>Export Actions</DropdownMenuLabel>
              <DropdownMenuItem>
                <Download className="w-3.5 h-3.5 mr-2" /> Export JSON
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Copy className="w-3.5 h-3.5 mr-2" /> Copy Results
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="text-status-danger">
                <Trash2 className="w-3.5 h-3.5 mr-2" /> Clear History
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Toasts */}
          <Button
            variant="secondary"
            size="md"
            onClick={() =>
              toast({
                title: "Inference completed",
                description: "4 objects detected in 142.3 ms",
                type: "success",
              })
            }
          >
            Trigger Success Toast
          </Button>

          <Button
            variant="ghost"
            size="md"
            onClick={() =>
              toast({
                title: "Validation error",
                description: "Image size exceeds 10MB ceiling",
                type: "error",
              })
            }
          >
            Trigger Error Toast
          </Button>
        </div>
      </section>

      {/* 8. Skeletons & Empty States */}
      <section className="flex flex-col gap-3">
        <div className="border-b border-border-subtle pb-2">
          <h2 className="text-sm font-semibold text-text-primary">
            8. Skeletons & Empty States
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-5 rounded-panel border border-border-subtle bg-card flex flex-col gap-3">
            <h3 className="text-xs font-semibold text-text-secondary uppercase tracking-wider">
              Loading Skeletons
            </h3>
            <Skeleton className="h-6 w-3/4" />
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-5/6" />
            <div className="flex gap-2 pt-2">
              <Skeleton className="h-8 w-24" />
              <Skeleton className="h-8 w-20" />
            </div>
          </div>

          <EmptyState
            icon={<Scan className="w-5 h-5 text-text-tertiary" />}
            title="No Detections Yet"
            description="Drag and drop a test image or enable the live webcam to stream predictions."
            action={
              <Button variant="primary" size="sm" leftIcon={<Plus className="w-3.5 h-3.5" />}>
                Upload Sample Image
              </Button>
            }
          />
        </div>
      </section>
    </div>
  );
};
