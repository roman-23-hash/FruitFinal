/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Playfair Display"', 'Georgia', 'serif'],
        body: ['"DM Sans"', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        grove: {
          50:  '#f6fdf0',
          100: '#ebfadf',
          200: '#d4f5be',
          300: '#aeec8e',
          400: '#84dd58',
          500: '#60c933',
          600: '#47a522',
          700: '#38811c',
          800: '#2f661b',
          900: '#285619',
        },
        amber: {
          50:  '#fffbeb',
          400: '#fbbf24',
          500: '#f59e0b',
          600: '#d97706',
        },
        rust: {
          400: '#f87171',
          500: '#ef4444',
        }
      },
      animation: {
        'spin-slow': 'spin 1.2s linear infinite',
        'pulse-soft': 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-up': 'fadeUp 0.5s ease forwards',
        'shimmer': 'shimmer 1.5s infinite',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(12px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}
