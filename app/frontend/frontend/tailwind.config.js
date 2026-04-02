/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          50: "#fff1f0",
          100: "#ffe0dd",
          200: "#ffc7c1",
          300: "#ffa296",
          400: "#ff6e5c",
          500: "#FF3621",
          600: "#ed2410",
          700: "#c81a09",
          800: "#a5190c",
          900: "#881b12",
        },
        accent: {
          50: "#e6f0f7",
          100: "#b3d1e6",
          200: "#80b2d5",
          300: "#4d93c4",
          400: "#2679b5",
          500: "#003159",
          600: "#002c50",
          700: "#002442",
          800: "#001c33",
          900: "#001225",
        },
      },
    },
  },
  plugins: [],
};
