import { useState, useRef, useCallback } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

// ─── Icons ────────────────────────────────────────────────────────────────────
function Spinner() {
  return (
    <svg className="animate-spin w-5 h-5 text-grove-400" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}

function UploadIcon() {
  return (
    <svg className="w-10 h-10 text-grove-500/60" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )
}

// ─── Confidence bar ───────────────────────────────────────────────────────────
function RipenessBar({ label, confidence, isTop }) {
  const pct = Math.round(confidence * 100)
  const lower = label.toLowerCase()
  let barColor = 'from-grove-400 to-grove-600'
  if (lower.includes('over') || lower.includes('rotten')) barColor = 'from-rust-400 to-rust-500'
  else if (lower.includes('un') || lower.includes('raw'))  barColor = 'from-amber-400 to-amber-600'

  return (
    <div className={`py-3 ${isTop ? '' : 'opacity-70'}`}>
      <div className="flex justify-between items-baseline mb-1.5">
        <span className={`text-sm font-medium capitalize ${isTop ? 'text-stone-100' : 'text-stone-400'}`}>
          {label}
          {isTop && <span className="ml-2 text-[10px] font-mono bg-grove-500/20 text-grove-400 px-1.5 py-0.5 rounded-full">TOP</span>}
        </span>
        <span className={`font-mono text-sm ${isTop ? 'text-grove-400' : 'text-stone-500'}`}>{pct}%</span>
      </div>
      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-700 ease-out`}
          style={{ width: `${pct}%` }}
          role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}
        />
      </div>
    </div>
  )
}

// ─── Gate rejection panel ─────────────────────────────────────────────────────
function GateRejectionPanel({ data }) {
  const pct = Math.round((data.gate_confidence ?? 0) * 100)
  return (
    <div className="animate-fade-up mt-6 rounded-2xl border border-amber-500/30 bg-amber-500/5 p-6 text-center">
      <div className="text-4xl mb-3">🚫</div>
      <h2 className="font-display text-xl font-bold text-amber-400 mb-1">Not a Guava</h2>
      <p className="text-stone-400 text-sm mb-4">
        {data.message ?? "The image doesn't appear to contain a guava fruit."}
      </p>
      <div className="inline-block glass rounded-xl px-4 py-2">
        <span className="font-mono text-xs text-stone-500">guava confidence: </span>
        <span className="font-mono text-sm text-amber-400">{pct}%</span>
      </div>
      <p className="mt-4 text-stone-600 text-xs">Please upload a photo of a guava fruit.</p>
    </div>
  )
}

// ─── Thermal image panel ──────────────────────────────────────────────────────
function ThermalPanel({ src }) {
  const [expanded, setExpanded] = useState(false)
  return (
    <div className="mt-4">
      <button
        onClick={() => setExpanded(e => !e)}
        className="w-full flex items-center justify-between px-4 py-3 glass rounded-xl hover:border-grove-500/30 transition-colors text-sm text-stone-400 hover:text-stone-200"
      >
        <span className="flex items-center gap-2">
          <span>🌡️</span>
          <span>Thermal Analysis</span>
        </span>
        <span className="font-mono text-xs text-grove-400">{expanded ? '▲ hide' : '▼ show'}</span>
      </button>

      {expanded && (
        <div className="animate-fade-up mt-2 glass rounded-xl overflow-hidden">
          <div className="px-4 pt-3 pb-1">
            <p className="text-xs text-stone-500 font-mono">
              Heatmap output from <code>thermal_out</code> layer · dark=cool · bright=warm
            </p>
          </div>
          <img
            src={src}
            alt="Thermal heatmap of the fruit"
            className="w-full object-contain max-h-64 bg-black/30"
          />
          <div className="flex items-center justify-between px-4 py-2">
            <div className="flex items-center gap-2 text-xs text-stone-600 font-mono">
              <span className="inline-block w-3 h-3 rounded-sm bg-purple-900" /> cool
              <span className="inline-block w-3 h-3 rounded-sm bg-orange-500 ml-2" /> warm
              <span className="inline-block w-3 h-3 rounded-sm bg-yellow-200 ml-2" /> hot
            </div>
            <a
              href={src}
              download="thermal.png"
              className="text-xs text-grove-400 hover:text-grove-300 font-mono"
            >
              ↓ download
            </a>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Results panel ────────────────────────────────────────────────────────────
function ResultsPanel({ data }) {
  const { predictions, thermal_image, meta } = data
  return (
    <div className="animate-fade-up mt-6 space-y-3">
      {/* Ripeness card */}
      <div className="glass rounded-2xl p-6">
        <div className="flex items-center gap-2 mb-4">
          <span className="w-2 h-2 rounded-full bg-grove-400 animate-pulse" />
          <h2 className="text-xs font-mono uppercase tracking-widest text-stone-400">Ripeness Analysis</h2>
          <span className="ml-auto text-[10px] font-mono text-stone-600">{meta?.processing_time_ms?.toFixed(0)} ms</span>
        </div>

        {/* Gate badge */}
        {meta?.gate_message && (
          <div className="mb-4 inline-flex items-center gap-1.5 bg-grove-500/10 border border-grove-500/20 rounded-full px-3 py-1">
            <span className="text-grove-400 text-xs">✓</span>
            <span className="text-grove-400 text-xs font-mono">{meta.gate_message}</span>
          </div>
        )}

        {/* Top prediction hero */}
        {predictions[0] && (
          <div className="mb-5 text-center py-4 border border-grove-500/20 rounded-xl bg-grove-500/5">
            <p className="text-grove-400 text-xs font-mono uppercase tracking-wider mb-1">Prediction</p>
            <p className="font-display text-3xl font-bold capitalize text-stone-50">{predictions[0].label}</p>
            <p className="text-grove-400 font-mono text-lg mt-1">
              {Math.round(predictions[0].confidence * 100)}% confident
            </p>
          </div>
        )}

        {/* All bars */}
        {predictions.length > 1 && (
          <div className="divide-y divide-white/5">
            {predictions.map((p, i) => (
              <RipenessBar key={p.label} label={p.label} confidence={p.confidence} isTop={i === 0} />
            ))}
          </div>
        )}

        <div className="mt-4 pt-3 border-t border-white/5">
          <p className="text-[10px] font-mono text-stone-600">
            model input: {meta?.model_input_shape?.join('×') ?? '—'}
          </p>
        </div>
      </div>

      {/* Thermal image */}
      {thermal_image && <ThermalPanel src={thermal_image} />}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function ImageUploader() {
  const [preview,    setPreview]   = useState(null)
  const [file,       setFile]      = useState(null)
  const [isDragging, setDragging]  = useState(false)
  const [status,     setStatus]    = useState('idle')
  const [result,     setResult]    = useState(null)
  const [errorMsg,   setErrorMsg]  = useState('')
  const [progress,   setProgress]  = useState(0)

  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!f.type.startsWith('image/')) {
      setErrorMsg('Please select an image file (JPEG, PNG, WEBP, BMP).')
      setStatus('error')
      return
    }
    setFile(f); setResult(null); setErrorMsg(''); setStatus('idle'); setProgress(0)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const onInputChange = (e) => handleFile(e.target.files?.[0])
  const onDragOver   = (e) => { e.preventDefault(); setDragging(true) }
  const onDragLeave  = ()  => setDragging(false)
  const onDrop       = (e) => { e.preventDefault(); setDragging(false); handleFile(e.dataTransfer.files?.[0]) }

  const predict = async () => {
    if (!file) return
    setStatus('loading'); setProgress(0); setResult(null); setErrorMsg('')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest()
        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) setProgress(Math.round((e.loaded / e.total) * 90))
        })
        xhr.addEventListener('load', () => { setProgress(100); resolve({ status: xhr.status, body: xhr.responseText }) })
        xhr.addEventListener('error',   () => reject(new Error('Network error — is the backend running?')))
        xhr.addEventListener('timeout', () => reject(new Error('Request timed out')))
        xhr.timeout = 120_000   // thermal inference can take a moment
        xhr.open('POST', `${API_BASE}/predict`)
        xhr.send(formData)
      })

      const data = JSON.parse(response.body)

      if (response.status === 200 && data.success) {
        setResult(data)
        setStatus('success')
      } else {
        setErrorMsg(data?.detail?.error ?? data?.error ?? data?.detail ?? 'Unknown error')
        setStatus('error')
      }
    } catch (err) {
      setErrorMsg(err.message ?? 'Failed to contact the server.')
      setStatus('error')
    }
  }

  const reset = () => {
    setPreview(null); setFile(null); setResult(null)
    setErrorMsg(''); setStatus('idle'); setProgress(0)
    if (inputRef.current) inputRef.current.value = ''
  }

  const isLoading = status === 'loading'

  return (
    <div className="space-y-4">
      {/* Drop zone */}
      <div
        role="button" tabIndex={0}
        aria-label="Image upload area"
        onKeyDown={(e) => e.key === 'Enter' && inputRef.current?.click()}
        onClick={() => !preview && inputRef.current?.click()}
        onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
        className={`relative glass rounded-2xl overflow-hidden cursor-pointer transition-all duration-300
          ${isDragging ? 'drop-zone-active' : 'hover:border-grove-500/30'}
          ${preview ? 'cursor-default' : ''}`}
      >
        {preview ? (
          <div className="relative group">
            <img src={preview} alt="Selected fruit" className="w-full max-h-72 object-contain bg-black/20" />
            <div
              className="absolute inset-0 bg-black/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer"
              onClick={() => inputRef.current?.click()} role="button" aria-label="Change image"
            >
              <span className="text-sm text-stone-200 font-medium">Click to change image</span>
            </div>
            <div className="absolute bottom-2 left-2 bg-black/60 rounded-lg px-2 py-1">
              <span className="font-mono text-[11px] text-stone-300 truncate max-w-[200px] block">{file?.name}</span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-14 px-6 text-center select-none">
            <UploadIcon />
            <p className="mt-4 text-stone-300 font-medium">
              {isDragging ? 'Drop your guava image here' : 'Drag & drop or click to upload'}
            </p>
            <p className="mt-1 text-stone-600 text-xs">Guava fruits only · JPEG, PNG, WEBP · max 20 MB</p>
          </div>
        )}
        <input ref={inputRef} type="file" accept="image/*" onChange={onInputChange} className="hidden" aria-hidden="true" />
      </div>

      {/* Progress bar */}
      {isLoading && (
        <div className="h-1 bg-white/5 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-grove-500 to-grove-300 transition-all duration-300" style={{ width: `${progress}%` }} />
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-3">
        <button
          onClick={predict} disabled={!file || isLoading}
          className="btn-primary flex-1 flex items-center justify-center gap-2"
        >
          {isLoading ? <><Spinner /><span>Analysing…</span></> : <span>Predict Ripeness</span>}
        </button>
        {(preview || result) && (
          <button onClick={reset} className="px-4 py-3 rounded-xl border border-white/10 text-stone-400 hover:text-stone-200 hover:border-white/20 transition-colors text-sm">
            Clear
          </button>
        )}
      </div>

      {/* Error */}
      {status === 'error' && errorMsg && (
        <div role="alert" className="animate-fade-up glass border-rust-500/30 rounded-xl p-4 text-rust-400 text-sm flex gap-3 items-start">
          <svg className="w-4 h-4 mt-0.5 shrink-0" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
          </svg>
          <div>
            <p className="font-semibold">Error</p>
            <p className="text-rust-400/80">{errorMsg}</p>
            {errorMsg.includes('backend') && (
              <p className="mt-1 text-stone-500 text-xs">
                Make sure backend is running: <code className="font-mono bg-white/5 px-1 rounded">uvicorn app.main:app --port 8000</code>
              </p>
            )}
          </div>
        </div>
      )}

      {/* Results — gate rejection or full results */}
      {status === 'success' && result && (
        result.is_guava === false
          ? <GateRejectionPanel data={result} />
          : <ResultsPanel data={result} />
      )}
    </div>
  )
}