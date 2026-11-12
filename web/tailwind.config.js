/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        canvas: "var(--surface-canvas)",
        subtle: "var(--surface-subtle)",
        card: "var(--surface-card)",
        elevated: "var(--surface-elevated)",
        overlay: "var(--surface-overlay)",

        "border-subtle": "var(--border-subtle)",
        "border-muted": "var(--border-muted)",
        "border-strong": "var(--border-strong)",

        "text-primary": "var(--text-primary)",
        "text-secondary": "var(--text-secondary)",
        "text-tertiary": "var(--text-tertiary)",
        "text-inverse": "var(--text-inverse)",

        brand: {
          DEFAULT: "var(--brand-primary)",
          hover: "var(--brand-hover)",
          subtle: "var(--brand-subtle)",
        },

        status: {
          success: "var(--status-success)",
          "success-subtle": "var(--status-success-subtle)",
          danger: "var(--status-danger)",
          "danger-subtle": "var(--status-danger-subtle)",
          warning: "var(--status-warning)",
          "warning-subtle": "var(--status-warning-subtle)",
        },
      },
      borderRadius: {
        control: "var(--radius-control)",
        panel: "var(--radius-panel)",
        chip: "var(--radius-chip)",
      },
      fontFamily: {
        sans: [
          "Geist Sans",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "Roboto",
          "sans-serif",
        ],
        mono: [
          "Geist Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Monaco",
          "Consolas",
          "monospace",
        ],
      },
      maxWidth: {
        container: "1280px",
      },
    },
  },
  plugins: [],
}
