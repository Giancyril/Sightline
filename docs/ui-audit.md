# Comprehensive UI/UX Audit: VisionSSD

**Project:** VisionSSD Front-End Redesign  
**Auditor:** Senior Product Designer & Front-End Engineer  
**Stage:** 0 (Research & Audit)  
**Date:** November 2026  

---

## 1. Audit Overview

This audit evaluates the current VisionSSD front-end against modern enterprise-grade design standards (Linear, Vercel Geist, Stripe). Every issue is documented with its user-experience impact, proposed technical resolution, and the implementation stage scheduled to address it.

---

## 2. Issues Audit Matrix

| Page / Area | Element | Issue Description | Why It Hurts (UX / Design Impact) | Proposed Fix | Fixing Stage |
|---|---|---|---|---|---|
| **Global Shell** | Color Theme | Hardcoded cyan-on-navy palette (`bg-slate-950`, `bg-slate-900`, `text-sky-400`). No light theme support. | Fails enterprise standards; unreadable in bright environments; looks like an unstyled developer prototype. | Introduce complete dual-theme CSS variable design system (Canvas, Surface, Border, Text). Default to system with manual toggle. | **Stage 1** |
| **Global Shell** | Navigation Logo | Glowing gradient icon (`bg-gradient-to-tr from-sky-600 to-indigo-600 shadow-sky-950`). | Outdated "crypto/Web3" aesthetic. Distracts from content and looks unpolished. | Replace with crisp monochrome 1px hairline optical glyph icon in 16px neutral frame. | **Stage 1** |
| **Global Shell** | Navigation Tabs | Filled pill buttons (`bg-sky-600 text-white shadow-sm`). | Heavy visual weight at the top of the viewport competes with primary canvas actions. | Replace with Geist-style understated segmented control with 1px border and neutral active state. | **Stage 1** |
| **Global Shell** | Layout Shift | Header and page content shift horizontally when switching between short and tall tabs. | Visible layout jitter whenever the browser vertical scrollbar toggles on and off. | Apply `scrollbar-gutter: stable;` to `html`/`body` and standardize container widths to `max-w-[1280px]`. | **Stage 1** |
| **Global Shell** | Footer Copy | Mentions internal network ports (`FastAPI Backend :8000 · React + Vite Frontend :5173`). | Internal development port numbers look amateurish and leak internal infrastructure details. | Clean up footer to show production-ready metadata: version, ML engine (PyTorch MobileNet V2), and clean documentation links. | **Stage 1** |
| **Global Shell** | Typography | Uses fallback system sans-serif without tabular figures for numeric values. | Numbers jitter during real-time streaming; typography lacks technical precision. | Integrate Geist Sans for UI prose and Geist Mono with tabular figures (`tabular-nums`) for telemetry. | **Stage 1** |
| **Detect** | Dropzone Layout | Massive empty vertical void below the upload dropzone when no image is loaded. | Poor information density; looks broken or incomplete before an image is selected. | Expand dropzone into a balanced hero drop container with sample preview chips and keyboard triggers. | **Stage 2** |
| **Detect** | Canvas Stacking | When an image is loaded, the dropzone remains visible above the canvas, pushing the image down. | User has to scroll to see their image and the detection boxes; waste of prime screen real estate. | Collapse dropzone into a compact top toolbar (swap image, clear, camera toggle) when canvas is active. | **Stage 2** |
| **Detect** | Sliders | Range inputs lack direct numeric inputs, tick intervals, and fine adjustment controls. | Imprecise threshold adjustments; users cannot type exact numbers like `0.25` or `0.45`. | Pair each range slider with a tabular numeric input box (`0.00`–`1.00`), tick indicators, and default resets. | **Stage 2** |
| **Detect** | Detections List | Static container with hardcoded cyan accents and no interactive hover linkage. | Hovering an item does not visually highlight the box on the canvas, hindering rapid verification. | Add bidirectional hover: hovering item highlights box on canvas; clicking canvas box scrolls & focuses item in list. | **Stage 2** |
| **Detect** | Mode Toggle | Upload vs. Webcam mode toggle uses heavy saturated buttons (`bg-sky-600`). | Visual noise; draws attention away from the primary canvas inspection area. | Convert to a clean 2-segment hairline control with 16px Lucide icons (`Upload`, `Camera`). | **Stage 2** |
| **Detect** | Bounding Boxes | Bounding boxes use thick saturated borders with arbitrary opacity fills. | Occludes image details underneath detection boxes; colors are not unified across views. | Implement 1.5px hairline box borders with calibrated Okabe-Ito categorical colors and subtle 10% fills. | **Stage 2** |
| **Model & Metrics** | Spec Cards | Four differently colored icon tiles (sky, emerald, purple, amber). | "Carnival dashboard" look. Colored tiles distract from semantic information. | Unify into quiet monochromatic cards with 1px hairline borders and uniform neutral icons. | **Stage 3** |
| **Model & Metrics** | Benchmark Table | Header and labels expose internal milestone jargon: *"Stage 5 Exit Criteria"*, *"Stage 3 Baseline"*. | Confuses external users; looks like an internal project ticket checklist rather than a product specification. | Rename to *"Validation Benchmarks & Accuracy"*, *"COCO Baseline Model"*, and *"Fine-Tuned Open Images Model"*. | **Stage 3** |
| **Model & Metrics** | SLA Card | Stat tile explicitly uses jargon phrase: *"Inference SLA < 300 ms (CPU)"*. | "SLA" is internal contract jargon; does not describe real-world frame throughput clearly. | Rephrase to *"CPU Inference Latency"* with typical benchmark time (~150 ms) and throughput. | **Stage 3** |
| **Model & Metrics** | Gate Badges | Success badges use harsh neon green fills (`bg-emerald-500/20 text-emerald-400`). | Distracting high-saturation blocks that break visual hierarchy. | Replace with subtle neutral/emerald badges with 1px border and clean typography. | **Stage 3** |
| **Model & Metrics** | Class Explorer | Class taxonomy cards have inconsistent color pill indicators and lack searchable filtering. | Difficult to inspect 20 classes quickly; categorical color mapping is not readily scannable. | Implement clean searchable taxonomy table with Okabe-Ito color swatches, COCO tag chips, and sample counts. | **Stage 3** |
| **History** | Session Gallery | History cards use heavy dark backgrounds with hardcoded cyan latency badges. | Does not adapt to light theme; metadata hierarchy is cluttered. | Redesign cards with 1px hairline border, aspect-ratio thumbnails, and tabular metadata chips. | **Stage 4** |
| **History** | Item Selection | Clicking a history card switches tab to Detect but only prints `console.log` (does not load image). | Broken user flow: user expects clicking a past detection to re-open it in the inspector canvas. | Connect history item click to load image preview and detection response directly into the Detect canvas. | **Stage 4** |
| **History** | Clear Action | "Clear All" uses aggressive full-red container styling (`bg-red-950/40 text-red-300`). | Calls undue attention to a destructive secondary action. | Replace with ghost button with subtle hover state and confirmation safety dialog. | **Stage 4** |
| **API Docs** | Width Constraint | Page wrapper is constrained to `max-w-4xl` (896px) inside `max-w-7xl` container. | Looks cramped; leaves one-third of the screen completely empty on desktop displays. | Expand to full `max-w-[1280px]` container using a clean two-column split (Console/Code on right, Reference on left). | **Stage 4** |
| **API Docs** | Code Snippets | Hardcoded local port numbers (`http://localhost:8000/v1/detect`) in code snippets. | Hardcoded `:8000` port looks like local scratch code; does not provide base URL customization. | Use configurable base URL with `/v1/detect` endpoint and origin detection (`window.location.origin`). | **Stage 4** |
| **API Docs** | Endpoint Table | Endpoint method badges use thick colored borders and generic styling. | Hard to parse at a glance; method badges should follow HTTP spec conventions quietly. | Clean HTTP method pills (`POST`, `GET`) with standard monospace font and subtle borders. | **Stage 4** |

---

## 3. WCAG AA Contrast Audit

| Element | Current Color Pairing | Current Contrast | WCAG AA Requirement | Status | Fix |
|---|---|---|---|---|---|
| Helper Text | `#94a3b8` on `#0f172a` | 4.8:1 (11px text) | 4.5:1 (requires ≥14px or 3:1 bold) | ⚠️ Borderline for 11px | Bump size to 12px with calibrated `--text-secondary` (≥5.2:1) |
| Latency Badges | `#38bdf8` on `#0284c7` (pill) | 2.9:1 | 4.5:1 | ❌ FAILS | Use high-contrast tabular mono text with neutral background |
| Parameter Labels | `#cbd5e1` on `#0f172a` | 8.2:1 | 4.5:1 | ✅ PASS | Maintain contrast in both light and dark themes |
| Light Theme (Missing) | N/A | N/A | 4.5:1 normal, 3:1 UI | ❌ MISSING | Design complete light theme with ≥7:1 body contrast |

---

## 4. Stage Progression Plan

* **Stage 0:** Research, UI audit, and design tokens documentation *(Current)*
* **Stage 1:** Design system tokens, Geist typography, light/dark theme toggle, and unified shell
* **Stage 2:** Redesign Detect view (split canvas/inspector, zero-dead-space dropzone, controls)
* **Stage 3:** Redesign Model & Metrics view (monochromatic spec cards, benchmark table, taxonomy)
* **Stage 4:** Redesign API Docs & History views (full-width layout, history re-inspection)
* **Stage 5:** Cross-platform responsiveness, accessibility verification, and release polish
