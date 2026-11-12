/**
 * web/src/lib/colors.ts
 * ---------------------
 * Categorical color system for VisionSSD based on Okabe-Ito and IBM Carbon principles.
 * Fixed, accessible colors mapped deterministically by Class ID (1-20).
 */

export interface ClassColor {
  id: number;
  name: string;
  hex: string;
  rgb: [number, number, number];
  isCoco: boolean;
}

export const CLASS_PALETTE: readonly ClassColor[] = [
  { id: 1, name: "Person", hex: "#0072B2", rgb: [0, 114, 178], isCoco: true },
  { id: 2, name: "Laptop", hex: "#E69F00", rgb: [230, 159, 0], isCoco: true },
  { id: 3, name: "Chair", hex: "#56B4E9", rgb: [86, 180, 233], isCoco: true },
  { id: 4, name: "Book", hex: "#009E73", rgb: [0, 158, 115], isCoco: true },
  { id: 5, name: "Backpack", hex: "#F0E442", rgb: [240, 228, 66], isCoco: true },
  { id: 6, name: "Cell phone", hex: "#D55E00", rgb: [213, 94, 0], isCoco: true },
  { id: 7, name: "Bottle", hex: "#CC79A7", rgb: [204, 121, 167], isCoco: true },
  { id: 8, name: "Clock", hex: "#882255", rgb: [136, 34, 85], isCoco: true },
  { id: 9, name: "Car", hex: "#44AA99", rgb: [68, 170, 153], isCoco: true },
  { id: 10, name: "Dog", hex: "#117733", rgb: [17, 119, 51], isCoco: true },
  { id: 11, name: "Guitar", hex: "#332288", rgb: [51, 34, 136], isCoco: false },
  { id: 12, name: "Microphone", hex: "#999933", rgb: [153, 153, 51], isCoco: false },
  { id: 13, name: "Helmet", hex: "#88CCEE", rgb: [136, 204, 238], isCoco: false },
  { id: 14, name: "Headphones", hex: "#661100", rgb: [102, 17, 0], isCoco: false },
  { id: 15, name: "Wheelchair", hex: "#6699CC", rgb: [102, 153, 204], isCoco: false },
  { id: 16, name: "Whiteboard", hex: "#AA4499", rgb: [170, 68, 153], isCoco: false },
  { id: 17, name: "Flashlight", hex: "#4477AA", rgb: [68, 119, 170], isCoco: false },
  { id: 18, name: "Calculator", hex: "#228833", rgb: [34, 136, 51], isCoco: false },
  { id: 19, name: "Stapler", hex: "#EE6677", rgb: [238, 102, 119], isCoco: false },
  { id: 20, name: "Wrench", hex: "#AA7744", rgb: [170, 119, 68], isCoco: false },
] as const;

const PALETTE_MAP = new Map<number, ClassColor>(
  CLASS_PALETTE.map((c) => [c.id, c])
);

const FALLBACK_COLOR: ClassColor = {
  id: 0,
  name: "Unknown",
  hex: "#71717A",
  rgb: [113, 113, 122],
  isCoco: false,
};

/**
 * Returns the categorical color specification for a given class ID.
 */
export function colorForClass(id: number): ClassColor {
  return PALETTE_MAP.get(id) ?? FALLBACK_COLOR;
}

/**
 * Helper to generate RGBA stroke and fill styles for canvas or CSS overlays.
 */
export function classColorRgba(id: number, alpha: number = 1): string {
  const c = colorForClass(id);
  return `rgba(${c.rgb[0]}, ${c.rgb[1]}, ${c.rgb[2]}, ${alpha})`;
}

