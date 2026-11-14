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

