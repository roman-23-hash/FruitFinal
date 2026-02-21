import ImageUploader from './components/ImageUploader.jsx'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* ── Header ── */}
      <header className="py-8 px-6 text-center">
        <p className="text-grove-400 font-mono text-xs tracking-widest uppercase mb-2">
          AI · Freshness Analysis
        </p>
        <h1 className="font-display text-4xl md:text-5xl font-bold text-stone-50 leading-tight">
          Fruit Ripeness
          <span className="italic text-grove-400"> Classifier</span>
        </h1>
        <p className="mt-3 text-stone-400 text-sm max-w-sm mx-auto">
          Upload a photo of any fruit and get an instant ripeness prediction powered by deep learning.
        </p>
      </header>

      {/* ── Main content ── */}
      <main className="flex-1 flex items-start justify-center px-4 pb-16 pt-4">
        <div className="w-full max-w-xl">
          <ImageUploader />
        </div>
      </main>

      {/* ── Footer ── */}
      <footer className="py-4 text-center text-stone-600 text-xs font-mono">
        fruit-ripeness-classifier · local inference only · no data leaves your machine
      </footer>
    </div>
  )
}
