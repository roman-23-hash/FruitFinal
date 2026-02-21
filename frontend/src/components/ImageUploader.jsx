import { useState, useRef, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// ─── Sub-components ──────────────────────────────────────────────────────────

function Spinner() {
  return (
    <svg
      className="animate-spin-slow w-5 h-5 text-grove-400"
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
      />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg className="w-10 h-10 text-grove-500/60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )
}

function RipenessBar({ label, confidence, isTop }) {
  const pct = Math.round(confidence * 100)

  // Color coding
  const lowerLabel = label.toLowerCase()
  let barColor = 'from-grove-400 to-grove-600'
  if (lowerLabel.includes('over') || lowerLabel.includes('rotten') || lowerLabel.includes('bad')) {
    barColor = 'from-rust-400 to-rust-500'
  } else if (lowerLabel.includes('un') || lowerLabel.includes('green') || lowerLabel.includes('raw')) {
    barColor = 'from-amber-400 to-amber-600'
  }

  return (
    <div className={`py-3 ${isTop ? '' : 'opacity-70'}`}>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className={`text-sm font-medium capitalize ${isTop ? 'text-stone-100' : 'text-stone-400'}`}>
          {label}
          {isTop && (
            <span className="ml-2 text-[10px] font-mono bg-grove-500/20 text-grove-400 px-1.5 py-0.5 rounded-full">
              TOP
            </span>
          )}
        </span>
        <span className={`font-mono text-sm ${isTop ? 'text-grove-400' : 'text-stone-500'}`}>
          {pct}%
        </span>
      </div>
      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={`${label}: ${pct}%`}
        />
      </div>
    </div>
  )
}

function ResultsPanel({ data, processingMs }) {
  const { predictions } = data

  return (
    <div className="animate-fade-up mt-6 glass rounded-2xl p-6">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4">
        <span className="w-2 h-2 rounded-full bg-grove-400 animate-pulse-soft" />
        <h2 className="text-xs font-mono uppercase tracking-widest text-stone-400">
          Analysis Complete
        </h2>
        <span className="ml-auto text-[10px] font-mono text-stone-600">
          {processingMs?.toFixed(0)} ms
        </span>
      </div>

      {/* Top prediction hero */}
      {predictions[0] && (
        <div className="mb-5 text-center py-4 border border-grove-500/20 rounded-xl bg-grove-500/5">
          <p className="text-grove-400 text-xs font-mono uppercase tracking-wider mb-1">Prediction</p>
          <p className="font-display text-3xl font-bold capitalize text-stone-50">
            {predictions[0].label}
          </p>
          <p className="text-grove-400 font-mono text-lg mt-1">
            {Math.round(predictions[0].confidence * 100)}% confident
          </p>
        </div>
      )}

      {/* All predictions */}
      {predictions.length > 1 && (
        <div className="divide-y divide-white/5">
          {predictions.map((p, i) => (
            <RipenessBar key={p.label} label={p.label} confidence={p.confidence} isTop={i === 0} />
          ))}
        </div>
      )}

      {/* Meta */}
      <div className="mt-4 pt-3 border-t border-white/5">
        <p className="text-[10px] font-mono text-stone-600">
          model input: {data.meta?.model_input_shape?.join('×') ?? '—'}
        </p>
      </div>
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ImageUploader() {
  const [preview, setPreview]   = useState(null)   // data URL
  const [file, setFile]         = useState(null)   // File object
  const [isDragging, setDragging] = useState(false)
  const [status, setStatus]     = useState('idle') // idle | loading | success | error
  const [result, setResult]     = useState(null)
  const [errorMsg, setErrorMsg] = useState('')
  const [progress, setProgress] = useState(0)      // 0–100 upload progress

  const inputRef = useRef(null)

  // ── File handling ────────────────────────────────────────────────────────
  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setErrorMsg('Please select an image file (JPEG, PNG, WEBP, BMP).')
      setStatus('error')
      return
    }
    setFile(f)
    setResult(null)
    setErrorMsg('')
    setStatus('idle')
    setProgress(0)

    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const onInputChange = (e) => handleFile(e.target.files?.[0])

  const onDragOver = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)
  const onDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files?.[0])
  }

  // ── Predict ──────────────────────────────────────────────────────────────
  const predict = async () => {
    if (!file) return

    setStatus('loading')
    setProgress(0)
    setResult(null)
    setErrorMsg('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Use XMLHttpRequest so we get upload progress events
      const response = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            setProgress(Math.round((e.loaded / e.total) * 90)) // cap at 90% until response
          }
        })

        xhr.addEventListener('load', () => {
          setProgress(100)
          resolve({ status: xhr.status, body: xhr.responseText })
        })
        xhr.addEventListener('error', () => reject(new Error('Network error — is the backend running?')))
        xhr.addEventListener('timeout', () => reject(new Error('Request timed out')))

        xhr.timeout = 60_000
        xhr.open('POST', `${API_BASE}/predict`)
        xhr.send(formData)
      })

      const data = JSON.parse(response.body)

      if (response.status === 200 && data.success) {
        setResult(data)
        setStatus('success')
      } else {
        const msg = data?.detail?.error ?? data?.error ?? data?.detail ?? 'Unknown error from server'
        setErrorMsg(msg)
        setStatus('error')
      }
    } catch (err) {
      setErrorMsg(err.message ?? 'Failed to contact the server.')
      setStatus('error')
    }
  }

  // ── Reset ────────────────────────────────────────────────────────────────
  const reset = () => {
    setPreview(null)
    setFile(null)
    setResult(null)
    setErrorMsg('')
    setStatus('idle')
    setProgress(0)
    if (inputRef.current) inputRef.current.value = ''
  }

  const isLoading = status === 'loading'

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-4">

      {/* ── Drop Zone ── */}
      <div
        role="button"
        tabIndex={0}
        aria-label="Image upload area — click or drag and drop"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onClick={() => !preview && inputRef.current?.click()}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative glass rounded-2xl overflow-hidden cursor-pointer
          transition-all duration-300
          ${isDragging ? 'drop-zone-active' : 'hover:border-grove-500/30'}
          ${preview ? 'cursor-default' : ''}
        `}
      >
        {preview ? (
          /* ── Image preview ── */
          <div className="relative group">
            <img
              src={preview}
              alt="Selected fruit"
              className="w-full max-h-72 object-contain bg-black/20"
            />
            {/* Overlay on hover to allow re-selection */}
            <div
              className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer"
              onClick={() => inputRef.current?.click()}
              role="button"
              aria-label="Change image"
            >
              <span className="text-sm text-stone-200 font-medium">Click to change image</span>
            </div>
            {/* Filename tag */}
            <div className="absolute bottom-2 left-2 bg-black/60 rounded-lg px-2 py-1">
              <span className="font-mono text-[11px] text-stone-300 truncate max-w-[200px] block">
                {file?.name}
              </span>
            </div>
          </div>
        ) : (
          /* ── Empty state ── */
          <div className="flex flex-col items-center justify-center py-14 px-6 text-center select-none">
            <UploadIcon />
            <p className="mt-4 text-stone-300 font-medium">
              {isDragging ? 'Drop your image here' : 'Drag & drop or click to upload'}
            </p>
            <p className="mt-1 text-stone-600 text-xs">JPEG, PNG, WEBP, BMP · max 20 MB</p>
          </div>
        )}

        {/* Hidden file input */}
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          onChange={onInputChange}
          className="hidden"
          aria-hidden="true"
        />
      </div>

      {/* ── Upload progress bar ── */}
      {isLoading && (
        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-grove-500 to-grove-300 transition-all duration-300 ease-out"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* ── Actions ── */}
      <div className="flex gap-3">
        <button
          onClick={predict}
          disabled={!file || isLoading}
          className="btn-primary flex-1 flex items-center justify-center gap-2"
          aria-label="Run ripeness prediction"
        >
          {isLoading ? (
            <>
              <Spinner />
              <span>Analysing…</span>
            </>
          ) : (
            <span>Predict Ripeness</span>
          )}
        </button>

        {(preview || result) && (
          <button
            onClick={reset}
            className="px-4 py-3 rounded-xl border border-white/10 text-stone-400 hover:text-stone-200 hover:border-white/20 transition-colors text-sm"
            aria-label="Clear and start over"
          >
            Clear
          </button>
        )}
      </div>

      {/* ── Error message ── */}
      {status === 'error' && errorMsg && (
        <div
          role="alert"
          className="animate-fade-up glass border-rust-500/30 rounded-xl p-4 text-rust-400 text-sm flex gap-3 items-start"
        >
          <svg className="w-4 h-4 mt-0.5 shrink-0" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <p className="font-semibold">Error</p>
            <p className="text-rust-400/80">{errorMsg}</p>
            {errorMsg.includes('backend') && (
              <p className="mt-1 text-stone-500 text-xs">
                Make sure the backend is running: <code className="font-mono bg-white/5 px-1 rounded">uvicorn app.main:app --port 8000</code>
              </p>
            )}
          </div>
        </div>
      )}

      {/* ── Results ── */}
      {status === 'success' && result && (
        <ResultsPanel data={result} processingMs={result.meta?.processing_time_ms} />
      )}
    </div>
  )
}
