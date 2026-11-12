import type { DetectionResponse } from "../types";

export interface StoredHistoryItem {
  id: string;
  createdAt: number; // epoch ms
  timestamp: string; // human readable string
  thumbnail: string; // lightweight compressed data URL with boxes pre-drawn
  previewUrl: string; // original blob URL if active in session
  result: DetectionResponse;
}

const DB_NAME = "vision_ssd_history_db";
const STORE_NAME = "history_records";
const DB_VERSION = 1;
const MAX_RECORDS = 200;

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined" || !window.indexedDB) {
      reject(new Error("IndexedDB not available"));
      return;
    }

    const request = window.indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: "id" });
        store.createIndex("createdAt", "createdAt", { unique: false });
      }
    };

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

/**
 * Load all stored history records sorted by newest first.
 */
export async function getStoredHistory(): Promise<StoredHistoryItem[]> {
  try {
    const db = await openDB();
    return new Promise((resolve) => {
      const tx = db.transaction(STORE_NAME, "readonly");
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();

      req.onsuccess = () => {
        const items: StoredHistoryItem[] = req.result || [];
        items.sort((a, b) => b.createdAt - a.createdAt);
        resolve(items.slice(0, MAX_RECORDS));
      };

      req.onerror = () => resolve([]);
    });
  } catch {
    return [];
  }
}

/**
 * Save a new history item with size cap of MAX_RECORDS.
 */
export async function saveHistoryItem(item: StoredHistoryItem): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);

    store.put(item);

    // Enforce max cap
    const countReq = store.count();
    countReq.onsuccess = () => {
      if (countReq.result > MAX_RECORDS) {
        const index = store.index("createdAt");
        const cursorReq = index.openCursor(); // oldest first
        cursorReq.onsuccess = () => {
          const cursor = cursorReq.result;
          if (cursor && countReq.result > MAX_RECORDS) {
            cursor.delete();
          }
        };
      }
    };
  } catch (err) {
    console.warn("Could not save detection to IndexedDB:", err);
  }
}

/**
 * Delete a single history item by ID.
 */
export async function deleteHistoryItem(id: string): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).delete(id);
  } catch (err) {
    console.warn("Could not delete history item:", err);
  }
}

/**
 * Clear all history items.
 */
export async function clearAllHistory(): Promise<void> {
  try {
    const db = await openDB();
    const tx = db.transaction(STORE_NAME, "readwrite");
    tx.objectStore(STORE_NAME).clear();
  } catch (err) {
    console.warn("Could not clear history store:", err);
  }
}

/**
 * Helper to generate a lightweight 320x240 thumbnail with bounding boxes drawn.
 */
export function createLightweightThumbnail(
  imageSrc: string,
  result: DetectionResponse
): Promise<string> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const maxDim = 320;
      let w = img.naturalWidth || 320;
      let h = img.naturalHeight || 240;

      if (w > maxDim || h > maxDim) {
        if (w >= h) {
          h = Math.round((h * maxDim) / w);
          w = maxDim;
        } else {
          w = Math.round((w * maxDim) / h);
          h = maxDim;
        }
      }

      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        resolve(imageSrc);
        return;
      }

      ctx.drawImage(img, 0, 0, w, h);

      // Draw boxes
      result.detections.forEach((det) => {
        const xmin = det.box_normalized.xmin * w;
        const ymin = det.box_normalized.ymin * h;
        const bw = (det.box_normalized.xmax - det.box_normalized.xmin) * w;
        const bh = (det.box_normalized.ymax - det.box_normalized.ymin) * h;

        ctx.lineWidth = 1.5;
        ctx.strokeStyle = det.color_hex;
        ctx.strokeRect(xmin, ymin, bw, bh);

        // Minimal dot
        ctx.fillStyle = det.color_hex;
        ctx.fillRect(xmin, ymin, 3, 3);
      });

      resolve(canvas.toDataURL("image/jpeg", 0.75));
    };

    img.onerror = () => resolve(imageSrc);
    img.src = imageSrc;
  });
}

/**
 * Format relative time ago (e.g. "Just now", "5m ago", "2h ago", "yesterday").
 */
export function formatTimeAgo(createdAt: number): string {
  const diffSec = Math.floor((Date.now() - createdAt) / 1000);
  if (diffSec < 45) return "Just now";
  if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
  if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
  if (diffSec < 172800) return "Yesterday";
  return `${Math.floor(diffSec / 86400)}d ago`;
}
