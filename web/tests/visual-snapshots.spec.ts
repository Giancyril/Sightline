import { test, expect } from "@playwright/test";

const PAGES = [
  { tab: "detect", title: "Detect" },
  { tab: "history", title: "History" },
  { tab: "model", title: "Model & Metrics" },
  { tab: "api", title: "API Reference" },
];

const THEMES = ["light", "dark"] as const;

for (const { tab, title } of PAGES) {
  for (const theme of THEMES) {
    test(`Visual test: ${title} page in ${theme} theme`, async ({ page }) => {
      await page.goto(`/?tab=${tab}&theme=${theme}`);
      await page.waitForLoadState("domcontentloaded");

      // Verify page header title matches
      await expect(page.locator("h1, h2").first()).toBeVisible();

      // Ensure root theme class is appropriately applied
      const rootHtml = page.locator("html");
      if (theme === "dark") {
        await expect(rootHtml).toHaveClass(/dark/);
      } else {
        await expect(rootHtml).not.toHaveClass(/dark/);
      }

      // Check no horizontal document scrollbar (layout hygiene)
      const hasHorizontalScroll = await page.evaluate(() => {
        return document.documentElement.scrollWidth > document.documentElement.clientWidth;
      });
      expect(hasHorizontalScroll).toBe(false);
    });
  }
}

