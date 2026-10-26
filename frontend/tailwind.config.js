import forms from "@tailwindcss/forms";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#F7F6F2",
        surface: "#FFFFFF",
        subtle: "#EFECE6",
        ink: "#18181B",
        muted: "#52525B",
        accent: "#1E293B",
        indigoAccent: "#312E81",
        // Legacy aliases mapped to restrained editorial palette for backward compatibility
        night: "#F7F6F2",
        panel: "#FFFFFF",
        cyber: "#1E293B",
        alert: "#B91C1C",
        glow: "#312E81",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
        display: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        subtle: "0 1px 2px 0 rgba(24, 24, 27, 0.04)",
        card: "0 1px 3px 0 rgba(24, 24, 27, 0.06), 0 1px 2px -1px rgba(24, 24, 27, 0.04)",
        glass: "0 1px 3px 0 rgba(24, 24, 27, 0.06)",
      },
    },
  },
  plugins: [forms],
};
