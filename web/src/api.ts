import type {
  ClassesResponse,
  CompareResponse,
  CorrectionCreateRequest,
  CorrectionResponse,
  DetectionResponse,
  ModelInfoResponse,
  ModelsResponse,
  ModelThresholdsResponse,
  OperationsTelemetry,
  VideoJobStatus,
  VideoSummary,
  VideoUploadResponse,
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
export const API_DOCS_DISPLAY_BASE_URL =
  import.meta.env.VITE_API_PUBLIC_URL || "https://api.vision-ssd.ai";

export async function detectImage(
  imageBlob: Blob | File,
  confidenceThreshold: number = 0.25,
  nmsIouThreshold: number = 0.45,
  modelId?: string,
  thresholdMode: "global" | "recommended" | "custom" = "global",
  customThresholds?: Record<string, number>
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", imageBlob, "image.jpg");

  const url = new URL(`${API_BASE_URL}/v1/detect`);
  url.searchParams.set("confidence_threshold", confidenceThreshold.toString());
  url.searchParams.set("nms_iou_threshold", nmsIouThreshold.toString());
  if (modelId) {
    url.searchParams.set("model_id", modelId);
  }
  if (thresholdMode) {
    url.searchParams.set("threshold_mode", thresholdMode);
  }
  if (thresholdMode === "custom" && customThresholds) {
    url.searchParams.set("custom_thresholds", JSON.stringify(customThresholds));
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Inference failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function fetchModelThresholds(
  modelId: string
): Promise<ModelThresholdsResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/models/${modelId}/thresholds`);
  if (!response.ok) {
    if (response.status === 404) {
      throw new Error(`No threshold file for model "${modelId}"`);
    }
    throw new Error(`Failed to fetch thresholds: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchClasses(): Promise<ClassesResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/classes`);
  if (!response.ok) {
    throw new Error(`Failed to fetch taxonomy: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchModelInfo(): Promise<ModelInfoResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/model`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchModels(): Promise<ModelsResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/models`);
  if (!response.ok) {
    throw new Error(`Failed to fetch models: ${response.statusText}`);
  }
  return response.json();
}

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean; version: string }> {
  const response = await fetch(`${API_BASE_URL}/healthz`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}

export async function fetchOpenApiSpec(): Promise<Record<string, unknown>> {
  const response = await fetch(`${API_BASE_URL}/openapi.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch OpenAPI spec: ${response.statusText}`);
  }
  return response.json();
}

export async function compareModels(
  imageBlob: Blob | File,
  modelA: string = "fine-tuned",
  modelB: string = "coco-baseline",
  confidenceThreshold: number = 0.25,
  nmsIouThreshold: number = 0.45
): Promise<CompareResponse> {
  const formData = new FormData();
  formData.append("file", imageBlob, "image.jpg");

  const url = new URL(`${API_BASE_URL}/v1/compare`);
  url.searchParams.set("model_a", modelA);
  url.searchParams.set("model_b", modelB);
  url.searchParams.set("confidence_threshold", confidenceThreshold.toString());
  url.searchParams.set("nms_iou_threshold", nmsIouThreshold.toString());

  const response = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Comparison failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function uploadVideo(
  file: File,
  options?: {
    model?: string;
    frameStride?: number;
    confidenceThreshold?: number;
    nmsIouThreshold?: number;
    countingLine?: {
      p1: { x: number; y: number };
      p2: { x: number; y: number };
    };
  }
): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append("file", file, file.name);

  const url = new URL(`${API_BASE_URL}/v1/video`);
  if (options?.model) url.searchParams.set("model", options.model);
  if (options?.frameStride) {
    url.searchParams.set("frame_stride", options.frameStride.toString());
  }
  if (options?.confidenceThreshold !== undefined) {
    url.searchParams.set(
      "confidence_threshold",
      options.confidenceThreshold.toString()
    );
  }
  if (options?.nmsIouThreshold !== undefined) {
    url.searchParams.set(
      "nms_iou_threshold",
      options.nmsIouThreshold.toString()
    );
  }
  if (options?.countingLine) {
    const cl = options.countingLine;
    url.searchParams.set("line_p1_x", cl.p1.x.toString());
    url.searchParams.set("line_p1_y", cl.p1.y.toString());
    url.searchParams.set("line_p2_x", cl.p2.x.toString());
    url.searchParams.set("line_p2_y", cl.p2.y.toString());
  }

  const response = await fetch(url.toString(), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Upload failed with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export async function fetchVideoJobStatus(
  jobId: string
): Promise<VideoJobStatus> {
  const response = await fetch(`${API_BASE_URL}/v1/video/${jobId}`);
  if (!response.ok) {
    let errorDetail = `Failed to get job status: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }
  return response.json();
}

export async function cancelVideoJob(jobId: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/v1/video/${jobId}`, {
    method: "DELETE",
  });
  if (!response.ok) {
    throw new Error(`Failed to cancel job: ${response.statusText}`);
  }
}

export async function fetchVideoSummary(
  jobId: string
): Promise<VideoSummary> {
  const response = await fetch(`${API_BASE_URL}/v1/video/${jobId}/summary`);
  if (!response.ok) {
    throw new Error(`Failed to fetch summary: ${response.statusText}`);
  }
  return response.json();
}

export async function saveCorrection(
  payload: CorrectionCreateRequest
): Promise<CorrectionResponse> {
  const response = await fetch(`${API_BASE_URL}/v1/corrections`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let errorDetail = `Failed to save correction with status ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // ignore
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export function getCorrectionsExportUrl(): string {
  return `${API_BASE_URL}/v1/corrections/export`;
}

export async function fetchOperationsTelemetry(): Promise<OperationsTelemetry> {
  const response = await fetch(`${API_BASE_URL}/v1/operations`);
  if (!response.ok) {
    throw new Error(`Failed to fetch operations telemetry: ${response.statusText}`);
  }
  return response.json();
}


