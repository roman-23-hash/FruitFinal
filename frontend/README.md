# Frontend — Fruit Ripeness UI

Vite + React + Tailwind CSS

## Setup

```bash
cd frontend
npm install
```

## Run (development)

```bash
npm run dev
# → http://localhost:5173
```

## Build (production)

```bash
npm run build
npm run preview  # preview the built output
```

## Configuration

Copy `.env.example` to `.env` and edit `VITE_API_BASE_URL` if your backend
runs on a different host or port:

```ini
VITE_API_BASE_URL=http://localhost:8000
```

## Structure

```
src/
├── main.jsx                  # React entry point
├── App.jsx                   # Layout shell
├── index.css                 # Tailwind + global styles
└── components/
    └── ImageUploader.jsx     # Core upload + prediction UI
```
