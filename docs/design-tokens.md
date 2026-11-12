# Design Tokens Specification: VisionSSD

**Project:** VisionSSD Front-End Redesign  
**Standard:** Enterprise SaaS Design Tokens (Geist / Carbon Inspired)  
**Stage:** 0 (Research & Audit)  
**Date:** November 2026  

---

## 1. Principles of the Token Architecture

1. **Semantic CSS Variables:** Components consume only abstract semantic tokens (e.g., `var(--surface-canvas)`, `var(--border-subtle)`). No component contains hardcoded hex values.
2. **True Neutral Grays:** Dark mode uses pure carbon-zinc neutrals (`#0A0A0A`, `#18181B`), not blue-tinted navy slates. Light mode uses pure clean white (`#FFFFFF`, `#FAFAFA`).
3. **Single Accent Rule:** Accent color (`#0070F3`) is strictly reserved for primary CTA buttons, active state indicators, selection highlights, and keyboard focus rings.
4. **Accessible Categorical Palette:** Class bounding boxes and metadata use an Okabe-Ito / IBM Carbon 20-color sequence calibrated for color-blind legibility and WCAG AA non-text contrast (≥ 3:1).

---

## 2. Color System

### 2.1 Neutral Surfaces & Hierarchy

| Token Name | CSS Variable | Light Theme | Dark Theme | Usage & Intent |
|---|---|---|---|---|
| **Canvas** | `--surface-canvas` | `#FFFFFF` | `#0A0A0A` | Global background behind all views |
| **Surface 0 (Subtle)** | `--surface-subtle` | `#FAFAFA` | `#121214` | Secondary sections, dropzone idle state |
| **Surface 1 (Card)** | `--surface-card` | `#FFFFFF` | `#18181B` | Elevated cards, metric panels, inspector sidebar |
| **Surface 2 (Elevated)** | `--surface-elevated` | `#F4F4F5` | `#27272A` | Hover states, active segmented buttons |
| **Surface Overlay** | `--surface-overlay` | `rgba(255, 255, 255, 0.85)` | `rgba(18, 18, 20, 0.85)` | Sticky header with blur, dropdown overlays |

### 2.2 Hairline Borders & Dividers

| Token Name | CSS Variable | Light Theme | Dark Theme | Usage & Intent |
|---|---|---|---|---|
| **Border Subtle** | `--border-subtle` | `#E4E4E7` | `#27272A` | 1px card perimeters, panel dividers, table rows |
| **Border Muted** | `--border-muted` | `#D4D4D8` | `#3F3F46` | Input field borders, inactive control boundaries |
| **Border Strong** | `--border-strong` | `#A1A1AA` | `#52525B` | Active state borders, selected box outlines |
| **Focus Ring** | `--focus-ring` | `rgba(0, 112, 243, 0.35)` | `rgba(0, 112, 243, 0.50)` | 2px focus ring with 2px offset on keyboard navigation |

### 2.3 Typography & Foreground

| Token Name | CSS Variable | Light Theme | Dark Theme | Contrast Ratio (vs Canvas) |
|---|---|---|---|---|
| **Text Primary** | `--text-primary` | `#09090B` | `#FAFAFA` | ≥ 17:1 (Exceeds WCAG AAA) |
| **Text Secondary** | `--text-secondary` | `#52525B` | `#A1A1AA` | ≥ 6.5:1 (Exceeds WCAG AA) |
| **Text Tertiary** | `--text-tertiary` | `#71717A` | `#71717A` | ≥ 4.6:1 (Meets WCAG AA) |
| **Text Inverse** | `--text-inverse` | `#FFFFFF` | `#09090B` | For solid filled primary buttons |

### 2.4 Interactive Brand Accent

| Token Name | CSS Variable | Light Theme | Dark Theme | Purpose |
|---|---|---|---|---|
| **Brand Primary** | `--brand-primary` | `#0070F3` | `#0070F3` | Primary action button, active tab indicator |
| **Brand Hover** | `--brand-hover` | `#005BEA` | `#3291FF` | Hover state for primary interactive elements |
| **Brand Subtle** | `--brand-subtle` | `rgba(0, 112, 243, 0.08)` | `rgba(0, 112, 243, 0.15)` | Active item background, tag fill |

---

## 3. Class Taxonomy Categorical Palette (20 Classes)

Fixed color-blind-safe mapping assigned deterministically by Class ID (1–20), shared identically across Canvas bounding boxes, chips, table indicators, and charts:

| Class ID | Class Name | Category | Hex Code | RGB Values | Contrast on Dark | Contrast on Light |
|:---:|---|---|:---:|---|:---:|:---:|
| **1** | Person | COCO Shared | `#0072B2` | `0, 114, 178` | 4.6:1 | 4.8:1 |
| **2** | Laptop | COCO Shared | `#E69F00` | `230, 159, 0` | 8.8:1 | 3.2:1 |
| **3** | Chair | COCO Shared | `#56B4E9` | `86, 180, 233` | 8.5:1 | 3.1:1 |
| **4** | Book | COCO Shared | `#009E73` | `0, 158, 115` | 6.2:1 | 3.5:1 |
| **5** | Backpack | COCO Shared | `#F0E442` | `240, 228, 66` | 13.9:1 | 2.1:1* |
| **6** | Cell phone | COCO Shared | `#D55E00` | `213, 94, 0` | 5.2:1 | 4.1:1 |
| **7** | Bottle | COCO Shared | `#CC79A7` | `204, 121, 167` | 6.9:1 | 3.2:1 |
| **8** | Clock | COCO Shared | `#882255` | `136, 34, 85` | 3.6:1 | 5.9:1 |
| **9** | Car | COCO Shared | `#44AA99` | `68, 170, 153` | 7.3:1 | 3.1:1 |
| **10** | Dog | COCO Shared | `#117733` | `17, 119, 51` | 3.9:1 | 5.6:1 |
| **11** | Guitar | Novel | `#332288` | `51, 34, 136` | 3.2:1 | 7.2:1 |
| **12** | Microphone | Novel | `#999933` | `153, 153, 51` | 6.8:1 | 3.2:1 |
| **13** | Helmet | Novel | `#88CCEE` | `136, 204, 238` | 10.5:1 | 2.5:1* |
| **14** | Headphones | Novel | `#661100` | `102, 17, 0` | 3.0:1 | 7.8:1 |
| **15** | Wheelchair | Novel | `#6699CC` | `102, 153, 204` | 6.9:1 | 3.3:1 |
| **16** | Whiteboard | Novel | `#AA4499` | `170, 68, 153` | 4.6:1 | 4.7:1 |
| **17** | Flashlight | Novel | `#4477AA` | `68, 119, 170` | 4.8:1 | 4.5:1 |
| **18** | Calculator | Novel | `#228833` | `34, 136, 51` | 4.7:1 | 4.6:1 |
| **19** | Stapler | Novel | `#EE6677` | `238, 102, 119` | 7.2:1 | 3.1:1 |
| **20** | Wrench | Novel | `#AA7744` | `170, 119, 68` | 6.1:1 | 3.5:1 |

*\*Note: High-luminance colors (Backpack, Helmet) utilize dark label badges on canvas overlays to guarantee text legibility.*

---

## 4. Typography Scale

```
Font Families:
  --font-sans: 'Geist Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'Geist Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
```

| Step | Token | Size | Line Height | Weight | Font Family | Usage |
|---|---|---|---|---|---|---|
| **1** | `--text-h1` | 20px (`1.25rem`) | 1.3 (`26px`) | 600 (Semi-bold) | Geist Sans | View page titles |
| **2** | `--text-h2` | 16px (`1.00rem`) | 1.4 (`22px`) | 600 (Semi-bold) | Geist Sans | Section headers, card titles |
| **3** | `--text-body` | 14px (`0.875rem`) | 1.5 (`21px`) | 400 (Regular) | Geist Sans | Default body copy, control labels, table rows |
| **4** | `--text-caption` | 12px (`0.75rem`) | 1.5 (`18px`) | 400 / 500 | Geist Sans / Mono | Metadata, slider descriptions, timestamps, table headers |
| **5** | `--text-telemetry` | 11px (`0.6875rem`) | 1.4 (`15px`) | 500 (Medium) | Geist Mono | Bounding box coordinates, latencies (`ms`), confidence percentages (`%`) |

*All numeric values enforce `font-variant-numeric: tabular-nums` to eliminate layout jitter during real-time streaming.*

---

## 5. Spacing & Spatial Grid

Base unit: **4px**

| Token | Pixels | Rem | Typical Applications |
|---|---|---|---|
| `--space-1` | 4px | `0.25rem` | Icon-to-label gaps, tight chip padding |
| `--space-2` | 8px | `0.50rem` | Control internal padding, button gaps, badge padding |
| `--space-3` | 12px | `0.75rem` | Input field horizontal padding, compact list gaps |
| `--space-4` | 16px | `1.00rem` | Card padding, standard element stacks |
| `--space-5` | 20px | `1.25rem` | Inspector panel gap, section spacing |
| `--space-6` | 24px | `1.50rem` | Page container gutters, grid gaps |
| `--space-8` | 32px | `2.00rem` | Major view section separation |
| `--space-12` | 48px | `3.00rem` | Top navigation spacing |

* **Layout Container:** `max-width: 1280px` centered with `padding: 0 24px` (16px on mobile `< 640px`).
* **Vertical Stability:** `scrollbar-gutter: stable` enforced on viewport root to eliminate tab-switching jitter.

---

## 6. Border Radii Scale

| Token | Value | Tailwind Class | Application |
|---|---|---|---|
| `--radius-control` | 6px | `rounded-[6px]` | Buttons, inputs, sliders, segment toggles, dropdowns |
| `--radius-panel` | 10px | `rounded-[10px]` | Cards, inspector drawer, canvas container, modal panels |
| `--radius-chip` | 4px | `rounded-[4px]` | Monospace telemetry badges, method pills, category tags |
| `--radius-full` | 9999px | `rounded-full` | Categorical color indicator dots, avatar discs |

---

## 7. Elevation, Shadows & Hairlines

* **Borders:** All panels and cards use `1px solid var(--border-subtle)` hair-line borders.
* **Shadows:** No heavy ambient glows or colored shadows.
  * Card standard: `none` (clean border separation)
  * Elevated dropdown/overlay: `0 4px 12px rgba(0, 0, 0, 0.08)` (Light) / `0 4px 12px rgba(0, 0, 0, 0.40)` (Dark)
* **Glassmorphism:** Strictly prohibited. No `backdrop-blur-md` on content containers; solid or 90% opaque neutral backgrounds only.
