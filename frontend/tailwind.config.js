import forms from "@tailwindcss/forms";

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        night: "#07111f",
        panel: "#0f1d31",
        cyber: "#64f1d6",
        alert: "#ff5e7d",
        glow: "#60a5fa"
      },
      fontFamily: {
        sans: ["Space Grotesk", "ui-sans-serif", "system-ui"],
        display: ["Sora", "ui-sans-serif", "system-ui"]
      },
      boxShadow: {
        glass: "0 24px 80px rgba(5, 15, 35, 0.35)"
      },
      backgroundImage: {
        mesh: "radial-gradient(circle at top left, rgba(100,241,214,0.18), transparent 30%), radial-gradient(circle at 80% 20%, rgba(96,165,250,0.18), transparent 28%), linear-gradient(135deg, #040b15 0%, #07111f 35%, #0d1830 100%)"
      }
    },
  },
  plugins: [forms],
};
