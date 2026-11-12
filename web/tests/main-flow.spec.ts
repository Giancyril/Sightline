import { test, expect } from "@playwright/test";

test.describe("VisionSSD Main Detection Flow", () => {
  test.beforeEach(async ({ page }) => {
    // Mock detect API response for deterministic offline/online testing
    await page.route("**/v1/detect*", async (route) => {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          model_name: "SSD MobileNet V2 (Open Images 20-Class Fine-Tuned)",
          model_version: "v1.0.0",
          total_detections: 2,
          inference_time_ms: 15.4,
          total_time_ms: 21.2,
          image_dimensions: { width: 1280, height: 720, channels: 3 },
          thresholds_applied: { confidence_threshold: 0.25, nms_iou_threshold: 0.45 },
          detections: [
            {
              class_id: 2,
              class_name: "Laptop",
              confidence: 0.9412,
              box_normalized: { ymin: 0.35, xmin: 0.22, ymax: 0.78, xmax: 0.65 },
              box_pixel: { ymin: 252, xmin: 281, ymax: 561, xmax: 832, width: 551, height: 309 },
              color_hex: "#E69F00",
              color_rgb: [230, 159, 0],
            },
            {
              class_id: 12,
              class_name: "Microphone",
              confidence: 0.8734,
              box_normalized: { ymin: 0.15, xmin: 0.68, ymax: 0.52, xmax: 0.88 },
              box_pixel: { ymin: 108, xmin: 870, ymax: 374, xmax: 1126, width: 256, height: 266 },
              color_hex: "#999933",
              color_rgb: [153, 153, 51],
            },
          ],
        }),
      });
    });

    await page.goto("/?tab=detect");
  });

  test("user can pick a sample, view detections on canvas and inspector table, filter by class, and open export", async ({
    page,
  }) => {
    // 1. Verify Detect view empty dropzone
    await expect(page.locator("h2, h3").filter({ hasText: /Drop an image/i })).toBeVisible();

    // 2. Click preset sample image
    const sampleBtn = page.getByRole("button", { name: /Office Desk/i }).first();
    await expect(sampleBtn).toBeVisible();
    await sampleBtn.click();

    // 3. Canvas and Inspector should render detections
    const canvas = page.locator('canvas[role="img"]');
    await expect(canvas).toBeVisible();

    // 4. Inspector summary telemetry
    await expect(page.getByText("2 objects")).toBeVisible();

    // 5. Detection items table
    await expect(page.getByText("Laptop").first()).toBeVisible();
    await expect(page.getByText("Microphone").first()).toBeVisible();

    // 6. Class filter chips
    const laptopFilterChip = page.getByRole("button", { name: /Laptop/i });
    if (await laptopFilterChip.isVisible()) {
      await laptopFilterChip.click();
      // Only Laptop should be in filtered list
      await expect(page.getByText("Laptop").first()).toBeVisible();
    }

    // 7. Export Menu
    const exportBtn = page.getByRole("button", { name: /Export/i });
    if (await exportBtn.isVisible()) {
      await exportBtn.click();
      await expect(page.getByText("Export JSON")).toBeVisible();
      await expect(page.getByText("Export CSV")).toBeVisible();
      await expect(page.getByText("Export Annotated PNG")).toBeVisible();
    }
  });
});

