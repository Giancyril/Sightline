export interface NormalizedBBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
}

export interface PixelBBox {
  ymin: number;
  xmin: number;
  ymax: number;
  xmax: number;
  width: number;
  height: number;
}

export interface DetectionItem {
  class_id: number;
  class_name: string;
  confidence: number;
  box_normalized: NormalizedBBox;
  box_pixel: PixelBBox;
  color_hex: string;
  color_rgb: [number, number, number];
}

export interface ImageDimensions {
  width: number;
  height: number;
  channels: number;
}

export interface ThresholdsApplied {
  confidence_threshold: number;
  nms_iou_threshold: number;
  threshold_mode?: string;
  calibration_applied?: boolean;
}

export interface DetectionResponse {
  model_id?: string;
  model_name: string;
  model_version: string;
  total_detections: number;
  inference_time_ms: number;
  total_time_ms: number;
  image_dimensions: ImageDimensions;
  thresholds_applied: ThresholdsApplied;
  detections: DetectionItem[];
  threshold_mode?: string;
  calibration_applied?: boolean;
}

export interface ClassThresholdResponseItem {
  class_id: number;
  class_name: string;
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  precision_raw: number;
  recall_raw: number;
  f1_raw: number;
  num_detections: number;
  fell_back: boolean;
  fallback_reason?: string | null;
}

export interface ModelThresholdsResponse {
  model_id: string;
  rule: string;
  date: string;
  data_hash: string;
  global_threshold: number;
  fallbacks: string[];
  thresholds: Record<string, number>;
  per_class: ClassThresholdResponseItem[];
}

export interface ModelRegistryItem {
  id: string;
  display_name: string;
  version: string;
  path: string | null;
  input_size: [number, number];
  num_classes: number;
  class_list: string[];
  metrics: Record<string, number>;
  loaded: boolean;
  status: string;
}

export interface ModelsResponse {
  total_models: number;
  models: ModelRegistryItem[];
}

export interface CompareResponse {
  model_a_result: DetectionResponse;
  model_b_result: DetectionResponse;
  total_time_ms: number;
}

export interface ClassItem {
  id: number;
  name: string;
  mid: string;
  is_coco: boolean;
  color_hex: string;
  color_rgb: [number, number, number];
  description: string;
}

export interface ClassesResponse {
  total_classes: number;
  classes: ClassItem[];
}

export interface ModelMetrics {
  mAP_all: number;
  mAP_coco_shared: number;
  mAP_novel_classes: number;
  coco_mAP_50_95: number;
}

export interface ModelInfoResponse {
  name: string;
  version: string;
  architecture: string;
  input_resolution: [number, number];
  num_classes: number;
  optimal_confidence_threshold: number;
  supported_formats: string[];
  max_file_size_bytes: number;
  metrics: ModelMetrics;
}

export interface HistoryRecord {
  id: string;
  timestamp: string;
  previewUrl: string;
  result: DetectionResponse;
  model_id?: string;
  createdAt?: number;
  thumbnail?: string;
}

export interface VideoJobStatus {
  job_id: string;
  state: "queued" | "running" | "done" | "failed";
  progress: number;
  frames_processed: number;
  total_frames_to_process: number;
  fps: number;
  duration_s: number;
  width: number;
  height: number;
  error?: string | null;
  created_at?: number;
  updated_at?: number;
  has_results: boolean;
  results?: {
    annotated_video: string;
    tracks_csv: string;
    summary_json: string;
  };
}

export interface VideoUploadResponse {
  job_id: string;
  state: string;
  fps: number;
  duration_s: number;
  frame_count: number;
  total_frames_to_process: number;
  width: number;
  height: number;
  mime_type: string;
  file_size_bytes: number;
}

export interface VideoSummary {
  job_id: string;
  unique_counts: number;
  counts_per_class: Record<string, number>;
  line_counts: {
    total_in: number;
    total_out: number;
    total_crossings: number;
    by_class: Record<string, { in: number; out: number }>;
    line: { p1: { x: number; y: number }; p2: { x: number; y: number } };
  };
  fps_processed: number;
  total_frames_processed: number;
  duration_s: number;
}

