import { API_DOCS_DISPLAY_BASE_URL } from "../api";

export interface ParameterSpec {
  name: string;
  in: "query" | "formData" | "header" | "path";
  type: string;
  required: boolean;
  default?: string;
  description: string;
}

export interface SchemaFieldSpec {
  field: string;
  type: string;
  description: string;
}

export interface ErrorCodeSpec {
  status: number;
  name: string;
  description: string;
}

export interface EndpointSpec {
  id: string;
  method: "POST" | "GET";
  path: string;
  summary: string;
  description: string;
  parameters: ParameterSpec[];
  responseFields: SchemaFieldSpec[];
  errorCodes: ErrorCodeSpec[];
  snippets: {
    curl: string;
    js: string;
    python: string;
  };
  exampleResponse: string;
}

export const ENDPOINTS_DATA: EndpointSpec[] = [
  {
    id: "detect",
    method: "POST",
    path: "/v1/detect",
    summary: "Detect objects in image or frame",
    description:
      "Performs real-time object detection across 20 target categories using the fine-tuned SSD MobileNet V2 model. Accepts multipart form data with image binary and optional threshold tuners.",
    parameters: [
      {
        name: "file",
        in: "formData",
        type: "binary",
        required: true,
        description: "Image payload. Supported formats: JPEG, PNG, WebP. Maximum size: 10 MB.",
      },
      {
        name: "confidence_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.25",
        description:
          "Confidence cutoff for reported detections (0.01 to 1.0). Detections below this score are filtered.",
      },
      {
        name: "nms_iou_threshold",
        in: "query",
        type: "float",
        required: false,
        default: "0.45",
        description:
          "Non-Maximum Suppression (NMS) intersection-over-union threshold (0.01 to 1.0) for deduplicating overlapping boxes.",
      },
    ],
    responseFields: [
      { field: "model_name", type: "string", description: "Name of the served detector architecture" },
      { field: "model_version", type: "string", description: "Semantic version of the fine-tuned checkpoint" },
      { field: "total_detections", type: "integer", description: "Count of predicted bounding boxes exceeding confidence threshold" },
      { field: "inference_time_ms", type: "float", description: "Raw neural network forward pass time in milliseconds" },
      { field: "total_time_ms", type: "float", description: "End-to-end pipeline latency including decode and post-processing" },
      { field: "image_dimensions", type: "object", description: "Input width, height, and channel dimensions in pixels" },
      { field: "detections[]", type: "array<object>", description: "List of detected objects with class labels, coordinates, and color tokens" },
    ],
    errorCodes: [
      { status: 400, name: "Bad Request", description: "Corrupt image payload or unreadable image stream" },
      { status: 413, name: "Payload Too Large", description: "Image file exceeds maximum limit of 10,485,760 bytes (10 MB)" },
      { status: 415, name: "Unsupported Media Type", description: "File format is not JPEG, PNG, or WebP" },
      { status: 422, name: "Validation Error", description: "Missing required file parameter or threshold query parameters out of range" },
    ],
    snippets: {
      curl: `curl -X POST "${API_DOCS_DISPLAY_BASE_URL}/v1/detect?confidence_threshold=0.25&nms_iou_threshold=0.45" \\
  -H "Accept: application/json" \\
  -F "file=@image.jpg"`,
      js: `const formData = new FormData();
formData.append("file", imageFile, "photo.jpg");

const url = new URL("${API_DOCS_DISPLAY_BASE_URL}/v1/detect");
url.searchParams.set("confidence_threshold", "0.25");
url.searchParams.set("nms_iou_threshold", "0.45");

const response = await fetch(url, {
  method: "POST",
  body: formData,
});

const result = await response.json();
console.log(\`Detected \${result.total_detections} objects in \${result.inference_time_ms}ms\`);`,
      python: `import requests

url = "${API_DOCS_DISPLAY_BASE_URL}/v1/detect"
params = {
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45
}

with open("image.jpg", "rb") as f:
    files = {"file": ("image.jpg", f, "image/jpeg")}
    response = requests.post(url, params=params, files=files)

data = response.json()
print(f"Total: {data['total_detections']} (latency: {data['inference_time_ms']}ms)")
for det in data["detections"]:
    print(f"- {det['class_name']}: {det['confidence']*100:.1f}%")`,
    },
    exampleResponse: `{
  "model_name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
  "model_version": "v1.0.0",
  "total_detections": 2,
  "inference_time_ms": 14.28,
  "total_time_ms": 18.52,
  "image_dimensions": {
    "width": 1280,
    "height": 720,
    "channels": 3
  },
  "thresholds_applied": {
    "confidence_threshold": 0.25,
    "nms_iou_threshold": 0.45
  },
  "detections": [
    {
      "class_id": 2,
      "class_name": "Laptop",
      "confidence": 0.9412,
      "box_normalized": {
        "ymin": 0.342,
        "xmin": 0.215,
        "ymax": 0.781,
        "xmax": 0.658
      },
      "box_pixel": {
        "ymin": 246,
        "xmin": 275,
        "ymax": 562,
        "xmax": 842,
        "width": 567,
        "height": 316
      },
      "color_hex": "#E69F00",
      "color_rgb": [230, 159, 0]
    }
  ]
}`,
  },
  {
    id: "classes",
    method: "GET",
    path: "/v1/classes",
    summary: "Get 20-class target taxonomy",
    description:
      "Returns the full list of supported object classes along with their COCO overlap flags, Open Images MID identifiers, accessible color palettes, and semantic definitions.",
    parameters: [],
    responseFields: [
      { field: "total_classes", type: "integer", description: "Total count of active target classes (20)" },
      { field: "classes[]", type: "array<object>", description: "Array of class taxonomy objects" },
      { field: "classes[].id", type: "integer", description: "Deterministic class ID (1 to 20)" },
      { field: "classes[].name", type: "string", description: "Human-readable concept name (e.g., Person, Laptop)" },
      { field: "classes[].mid", type: "string", description: "Open Images ontology Machine Identifier (/m/...)" },
      { field: "classes[].is_coco", type: "boolean", description: "True if concept overlaps with COCO 80-class pretraining" },
      { field: "classes[].color_hex", type: "string", description: "WCAG-compliant hex color code assigned to class" },
      { field: "classes[].description", type: "string", description: "Official taxonomy scope and semantic boundary definition" },
    ],
    errorCodes: [
      { status: 500, name: "Internal Server Error", description: "Failed to read taxonomy registry file" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/classes" \\
  -H "Accept: application/json"`,
      js: `const response = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/classes");
const data = await response.json();
console.log(\`Loaded \${data.total_classes} classes\`);`,
      python: `import requests

response = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/classes")
data = response.json()
for c in data["classes"]:
    tag = "COCO" if c["is_coco"] else "Novel"
    print(f"#{c['id']} {c['name']} [{tag}]")`,
    },
    exampleResponse: `{
  "total_classes": 20,
  "classes": [
    {
      "id": 1,
      "name": "Person",
      "mid": "/m/01g317",
      "is_coco": true,
      "color_hex": "#0072B2",
      "color_rgb": [0, 114, 178],
      "description": "Individual human instance (adult, child, infant)."
    },
    {
      "id": 11,
      "name": "Guitar",
      "mid": "/m/0342h",
      "is_coco": false,
      "color_hex": "#332288",
      "color_rgb": [51, 34, 136],
      "description": "Acoustic or electric stringed musical instrument."
    }
  ]
}`,
  },
  {
    id: "model",
    method: "GET",
    path: "/v1/model",
    summary: "Get model metadata & benchmarks",
    description:
      "Retrieves technical architecture details, input resolution constraints (300×300), anchor box counts, and validation evaluation metrics (mAP@0.5 and COCO mAP@[.5:.95]).",
    parameters: [],
    responseFields: [
      { field: "name", type: "string", description: "Full descriptive model name" },
      { field: "version", type: "string", description: "Model checkpoint version (v1.0.0)" },
      { field: "architecture", type: "string", description: "Backbone and detector type (SSDMobileNetV2)" },
      { field: "input_resolution", type: "array<int>", description: "Required network input dimensions [width, height]" },
      { field: "num_classes", type: "integer", description: "Output logit dimension including background class (21)" },
      { field: "optimal_confidence_threshold", type: "float", description: "F1-tuned threshold operating point (0.25)" },
      { field: "metrics", type: "object", description: "Quantitative validation benchmarks (mAP_all, mAP_novel_classes)" },
    ],
    errorCodes: [
      { status: 503, name: "Service Unavailable", description: "Model weights not yet loaded into memory" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/v1/model" \\
  -H "Accept: application/json"`,
      js: `const response = await fetch("${API_DOCS_DISPLAY_BASE_URL}/v1/model");
const metadata = await response.json();
console.log(\`Model: \${metadata.name} (input: \${metadata.input_resolution.join("x")})\`);`,
      python: `import requests

response = requests.get("${API_DOCS_DISPLAY_BASE_URL}/v1/model")
meta = response.json()
print(f"Architecture: {meta['architecture']}")
print(f"Novel mAP@0.5: {meta['metrics']['mAP_novel_classes']:.4f}")`,
    },
    exampleResponse: `{
  "name": "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
  "version": "v1.0.0",
  "architecture": "SSDMobileNetV2",
  "input_resolution": [300, 300],
  "num_classes": 21,
  "optimal_confidence_threshold": 0.25,
  "supported_formats": ["JPEG", "PNG", "WebP"],
  "max_file_size_bytes": 10485760,
  "metrics": {
    "mAP_all": 0.2644,
    "mAP_coco_shared": 0.3471,
    "mAP_novel_classes": 0.1816,
    "coco_mAP_50_95": 0.1603
  }
}`,
  },
  {
    id: "health",
    method: "GET",
    path: "/healthz",
    summary: "Service liveness & readiness probe",
    description:
      "Health check endpoint designed for orchestration systems (Docker, Kubernetes) and monitoring probes. Returns model load status and runtime service version.",
    parameters: [],
    responseFields: [
      { field: "status", type: "string", description: "Service health state ('ok')" },
      { field: "model_loaded", type: "boolean", description: "True if detector PyTorch weights are initialized in memory" },
      { field: "version", type: "string", description: "Service API version" },
    ],
    errorCodes: [
      { status: 503, name: "Service Unavailable", description: "Detector failed to initialize or model weights missing" },
    ],
    snippets: {
      curl: `curl -X GET "${API_DOCS_DISPLAY_BASE_URL}/healthz" \\
  -H "Accept: application/json"`,
      js: `const res = await fetch("${API_DOCS_DISPLAY_BASE_URL}/healthz");
const health = await res.json();
console.log(\`Service status: \${health.status}, Model loaded: \${health.model_loaded}\`);`,
      python: `import requests

res = requests.get("${API_DOCS_DISPLAY_BASE_URL}/healthz")
assert res.status_code == 200
print("Service healthy:", res.json())`,
    },
    exampleResponse: `{
  "status": "ok",
  "model_loaded": true,
  "version": "1.0.0"
}`,
  },
];

