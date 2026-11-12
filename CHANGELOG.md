# Changelog

All notable changes to the Object Detection System (Sightline) web interface are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [ui-v2] - 2026-11-12

Complete redesign and modernization of the Sightline Object Detection web interface, raising it to a professional-grade computer vision suite across 8 structured stages.

### Patterns Borrowed & Design System Foundations

- **Linear / Vercel Geist Aesthetic**:
  - Neutral monochrome-first color system using CSS variables with automatic dark and light theme switching.
  - Functional semantic tokens for status (`emerald` for success, `amber` for warnings, `rose` for danger/errors, `indigo` for brand accents).
  - High-density typography using Geist Sans for technical clarity and Geist Mono for coordinates, latencies, probabilities, and code.
  - Border-dominant depth hierarchy (`border-border-subtle`, `border-border-default`, `border-border-strong`) without heavy drop shadows.

- **Okabe-Ito Colorblind-Safe Palette**:
  - Universal color palette for the 20 object classes and visualization charts (Orange, Sky Blue, Bluish Green, Yellow, Blue, Vermilion, Reddish Purple, Amber, Slate, etc.), ensuring high contrast and perceptual distinguishability across deuteranopia, protanopia, and tritanopia.

- **Roboflow & Label Studio Workspace Patterns**:
  - Two-pane workspace filling 100% viewport height with independent pane scrolling.
  - Responsive layout transforming into a bottom sheet on mobile screens (<900px).
  - SVG vector overlay canvas with pan/zoom (10% to 500%), pixel-perfect bounding box alignment at all zoom levels, 2px stroke borders, and small monospace tag chips.
  - Two-way hover and selection linking: hovering/clicking a table row highlights the box on canvas; hovering/clicking a box scrolls to and selects the table row.
  - Pinned inspector card displaying class, score, bounding coordinates (`[ymin, xmin, ymax, xmax]`), and area in square pixels.

- **IBM Carbon Design System Chart Principles**:
  - Minimalist, highly informative evaluation charts without decorative junk or heavy gridlines.
  - Direct data labels on bars and grouped comparisons.
  - In-SVG vector tooltips and delta annotations (`↑ +18.2 pts`) respecting viewport bounds and theme changes.
  - Precision-Recall curves with dynamic per-class selector and optimal threshold mark.

- **Stripe & Vercel API Documentation Layout**:
  - Two-column documentation layout with sticky left scroll-spy sidebar and tri-language code panels (cURL, JavaScript Fetch, Python Requests).
  - Live OpenAPI synchronization preventing doc drift.
  - Embedded "Try It" console with real-time API invocation, latency readout, and HTTP status inspection.

- **Privacy-First Local Persistence**:
  - Client-side history storage via IndexedDB with automatic fallback to localStorage and a 200-item FIFO cap.
  - Zero-telemetry guarantee: no user images, video streams, or inference logs are ever uploaded to remote servers.

---

### Stages & Detailed Changes

#### Stage 0: Research, Audit & Design Tokens
- Conducted deep audit of previous UI limitations, color contrast issues, and responsiveness gaps.
- Defined OKLCH-based design tokens, dark/light theme palettes, and class color mappings.

#### Stage 1: Design Tokens, Theme & Base Components
- Built foundational atomic components: `Button`, `Badge`, `Card`, `Dialog`, `Dropdown`, `Input`, `Slider`, `Skeleton`, `Table`, `Tabs`, `Toast`.
- Implemented `ThemeProvider` with system preference detection, manual toggle, and no-flash persistence.

#### Stage 2: App Shell, Navigation & Command Palette
- Created responsive App Shell with desktop top navigation and mobile drawer.
- Built global Command Palette (`Ctrl+K` / `Cmd+K`) supporting navigation, theme toggle, sample runs, and keyboard shortcut previews.
- Added live API health badge in header with automatic retry polling.

#### Stage 3: Detect Workspace, Canvas & Inspector
- Replaced single-column layout with two-pane 68% / 32% responsive workspace.
- Built drop zone with clipboard paste support, drag-and-drop, and 3 bundled one-click sample images.
- Implemented SVG vector detection canvas with zoom, pan, reset, fullscreen, and label toggle.
- Added Inspector summary strip (objects, unique classes, inference latency), confidence slider with instant client-side filtering, and NMS slider.

#### Stage 4: Detect Interactions, Export & Webcam
- Added two-way linking between canvas bounding boxes and inspector table rows.
- Built multi-format export dropdown: JSON, CSV, annotated PNG, and copy cURL request.
- Implemented lazy-loaded real-time Webcam detection mode with live FPS, inference latency, pause/resume, and snapshot to History.
- Added keyboard shortcuts: `U` upload, `Space` toggle labels, arrow keys step through detections, `Esc` clear selection, `?` shortcuts modal.

#### Stage 5: History Grid & Table Views
- Built responsive history dashboard with thumbnail cards, class badges, latency chips, and relative timestamps.
- Added toggle between responsive grid and dense table views.
- Implemented search by class name, filter by class dropdown, sort by newest/oldest/objects/latency, and clear-all confirmation dialog.

#### Stage 6: Model Metrics Charts & Taxonomy
- Designed evaluation report layout with architecture specs (MobileNetV2 + SSDLite, 300x300 input, 20 classes, ~22ms CPU latency).
- Implemented lazy-loaded benchmark charts: Baseline vs Fine-Tuned grouped bars, sorted per-class AP horizontal bars, and interactive Precision-Recall curve.
- Added full 20-class taxonomy table with COCO-shared vs Novel filter tabs, class color swatches, and test sample counts.

#### Stage 7: API Docs with Try-It Panel
- Built full API reference documentation covering `POST /v1/detect`, `GET /v1/classes`, `GET /v1/model`, and `GET /health`.
- Added interactive Try-It console allowing immediate execution against live API with sample images or custom uploads.
- Code snippet generator supporting cURL, JavaScript, and Python with one-click copy confirmation.

#### Stage 8: Polish, Accessibility, Performance & Release
- Unified motion curves (150ms-200ms ease-out) and added `@media (prefers-reduced-motion)` overrides.
- Enhanced accessibility with `role="region"`, `role="img"`, ARIA labels on canvas, keyboard navigation (`tabIndex={0}`, Enter/Space) for history cards, and an `aria-live="polite"` region for dynamic detection count announcements.
- Verified production build bundle size: main bundle 160.43 kB gzipped (target: < 250 kB gzipped).
- Added comprehensive Playwright end-to-end tests for main detection flow and 4-page visual snapshot matrix across light and dark themes.
