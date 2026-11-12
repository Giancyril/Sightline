import React from "react";
import { Download, FileJson, FileSpreadsheet, Image as ImageIcon, Terminal, Check } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuLabel,
} from "../ui/DropdownMenu";
import { useToast } from "../ui/useToast";
import type { DetectionItem, DetectionResponse } from "../../types";

interface ExportMenuProps {
  detections: DetectionItem[];
  rawResult: DetectionResponse | null;
  confThreshold: number;
  nmsThreshold: number;
  onExportAnnotatedImage: () => void;
  disabled?: boolean;
}

export const ExportMenu: React.FC<ExportMenuProps> = ({
  detections,
  rawResult,
  confThreshold,
  nmsThreshold,
  onExportAnnotatedImage,
  disabled = false,
}) => {
  const { toast } = useToast();
  const [copiedCurl, setCopiedCurl] = React.useState(false);

  // Export JSON
  const handleExportJson = () => {
    if (!rawResult) return;
    const exportData = {
      model_name: rawResult.model_name,
      model_version: rawResult.model_version,
      thresholds: {
        confidence: confThreshold,
        nms_iou: nmsThreshold,
      },
      image_dimensions: rawResult.image_dimensions,
      inference_time_ms: rawResult.inference_time_ms,
      total_detections: detections.length,
      detections,
      exported_at: new Date().toISOString(),
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vision-ssd-detections-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Export complete",
      description: `Saved ${detections.length} detections as JSON`,
      type: "success",
    });
  };

  // Export CSV
  const handleExportCsv = () => {
    if (!detections.length) return;
    const headers = [
      "class_id",
      "class_name",
      "confidence",
      "box_xmin_px",
      "box_ymin_px",
      "box_width_px",
      "box_height_px",
      "area_px",
      "norm_xmin",
      "norm_ymin",
      "norm_xmax",
      "norm_ymax",
    ];

    const rows = detections.map((d) => [
      d.class_id,
      `"${d.class_name.replace(/"/g, '""')}"`,
      d.confidence.toFixed(4),
      d.box_pixel.xmin,
      d.box_pixel.ymin,
      d.box_pixel.width,
      d.box_pixel.height,
      d.box_pixel.width * d.box_pixel.height,
      d.box_normalized.xmin.toFixed(4),
      d.box_normalized.ymin.toFixed(4),
      d.box_normalized.xmax.toFixed(4),
      d.box_normalized.ymax.toFixed(4),
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vision-ssd-detections-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Export complete",
      description: `Saved ${detections.length} detections as CSV`,
      type: "success",
    });
  };

  // Copy cURL command
  const handleCopyCurl = () => {
    const curlCmd = `curl -X POST "http://localhost:8000/v1/detect?confidence_threshold=${confThreshold}&nms_iou_threshold=${nmsThreshold}" \\
  -H "Accept: application/json" \\
  -F "file=@image.jpg"`;

    void navigator.clipboard.writeText(curlCmd);
    setCopiedCurl(true);
    setTimeout(() => setCopiedCurl(false), 2000);

    toast({
      title: "cURL command copied",
      description: "Copied API request with active thresholds to clipboard",
      type: "info",
    });
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-control border border-border-subtle bg-card text-text-primary hover:bg-elevated disabled:opacity-50 disabled:cursor-not-allowed transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand"
          aria-label="Export menu"
        >
          <Download className="w-3.5 h-3.5 text-text-tertiary" />
          <span>Export</span>
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuLabel>Export results</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleExportJson} disabled={!detections.length}>
          <FileJson className="w-3.5 h-3.5 mr-2 text-brand" />
          <span>JSON payload</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={handleExportCsv} disabled={!detections.length}>
          <FileSpreadsheet className="w-3.5 h-3.5 mr-2 text-status-success" />
          <span>CSV spreadsheet</span>
        </DropdownMenuItem>
        <DropdownMenuItem onClick={onExportAnnotatedImage} disabled={!detections.length}>
          <ImageIcon className="w-3.5 h-3.5 mr-2 text-status-warning" />
          <span>Annotated PNG</span>
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem onClick={handleCopyCurl}>
          {copiedCurl ? (
            <Check className="w-3.5 h-3.5 mr-2 text-status-success" />
          ) : (
            <Terminal className="w-3.5 h-3.5 mr-2 text-text-tertiary" />
          )}
          <span>Copy cURL</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
};
