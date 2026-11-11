import type { ClassesResponse, DetectionResponse, ModelInfoResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function detectImage(
  imageBlob: Blob | File,
  confidenceThreshold: number = 0.25,
  nmsIouThreshold: number = 0.45
): Promise<DetectionResponse> {
  const formData = new FormData();
  formData.append("file", imageBlob, "image.jpg");

  const url = new URL(`${API_BASE_URL}/v1/detect`);
  url.searchParams.set("confidence_threshold", confidenceThreshold.toString());
  url.searchParams.set("nms_iou_threshold", nmsIouThreshold.toString());

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

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean; version: string }> {
  const response = await fetch(`${API_BASE_URL}/healthz`);
  if (!response.ok) {
    throw new Error(`Health check failed: ${response.statusText}`);
  }
  return response.json();
}
