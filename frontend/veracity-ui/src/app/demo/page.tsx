'use client'

import { useEffect, useRef, useState, useCallback, DragEvent } from 'react'

// ── Types ────────────────────────────────────────────────────────────────
interface TokenSpan {
  id: number
  text: string
  kind: 'normal' | 'high-entropy' | 'corrected'
  original?: string
  corrected?: string
}

interface CorrectionEntry {
  id: number
  wrong: string
  right: string
  src: string
  confidence: number
  diff: number
}

const BACKEND_URL = '/api'

const PROVIDERS = ['Ollama (llama3)', 'Gemini', 'OpenAI', 'Claude']

let uid = 1

// ── File type helpers ─────────────────────────────────────────────────────
function getFileIcon(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'pdf') return '📄'
  if (['doc', 'docx'].includes(ext)) return '📝'
  if (['xls', 'xlsx', 'csv'].includes(ext)) return '📊'
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext)) return '🖼️'
  if (['mp4', 'mov', 'avi', 'mkv'].includes(ext)) return '🎬'
  if (['mp3', 'wav', 'ogg'].includes(ext)) return '🎵'
  if (['zip', 'tar', 'gz', 'rar'].includes(ext)) return '🗜️'
  if (['js', 'ts', 'jsx', 'tsx', 'py', 'go', 'rs', 'json'].includes(ext)) return '💻'
  if (['txt', 'md'].includes(ext)) return '📃'
  return '📎'
}

function getFileColor(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'pdf') return '#f87171'
  if (['doc', 'docx'].includes(ext)) return '#60a5fa'
  if (['xls', 'xlsx', 'csv'].includes(ext)) return '#34d399'
  if (['png', 'jpg', 'jpeg', 'gif', 'webp', 'svg'].includes(ext)) return '#a78bfa'
  if (['js', 'ts', 'jsx', 'tsx', 'py', 'go', 'rs', 'json'].includes(ext)) return '#fbbf24'
  return '#94a3b8'
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function truncateName(name: string, max = 22): string {
  if (name.length <= max) return name
  const ext = name.includes('.') ? '.' + name.split('.').pop() : ''
  return name.slice(0, max - ext.length - 1) + '…' + ext
}

// ── Perfect File Button ───────────────────────────────────────────────────
function FileButton({
  attachedFile,
  onAttach,
  onDelete,
  isDragging,
}: {
  attachedFile: File | null
  onAttach: (f: File) => void
  onDelete: () => void
  isDragging: boolean
}) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [hoverDelete, setHoverDelete] = useState(false)
  const [hoverAttach, setHoverAttach] = useState(false)
  const [justAttached, setJustAttached] = useState(false)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) {
      onAttach(file)
      setJustAttached(true)
      setTimeout(() => setJustAttached(false), 500)
    }
    e.target.value = ''
  }

  const color = attachedFile ? getFileColor(attachedFile.name) : '#475569'
  const icon  = attachedFile ? getFileIcon(attachedFile.name)  : null

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />

      {/* ── No file: paperclip icon button ── */}
      {!attachedFile && (
        <button
          onClick={() => fileInputRef.current?.click()}
          onMouseEnter={() => setHoverAttach(true)}
          onMouseLeave={() => setHoverAttach(false)}
          title="Attach a file  (or drag & drop)"
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 28, height: 28, borderRadius: 6, flexShrink: 0,
            border: isDragging
              ? '1.5px dashed rgba(99,102,241,0.7)'
              : hoverAttach
                ? '1px solid rgba(99,102,241,0.5)'
                : '1px solid rgba(255,255,255,0.09)',
            background: isDragging
              ? 'rgba(99,102,241,0.12)'
              : hoverAttach
                ? 'rgba(99,102,241,0.07)'
                : 'rgba(255,255,255,0.03)',
            cursor: 'pointer', outline: 'none',
            transition: 'border 0.16s, background 0.16s',
          }}
        >
          {/* SVG paperclip */}
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none"
            stroke={isDragging || hoverAttach ? '#6366f1' : '#64748b'}
            strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
            style={{ transition: 'stroke 0.16s' }}
          >
            <path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/>
          </svg>
        </button>
      )}

      {/* ── File attached: compact pill ── */}
      {attachedFile && (
        <div
          style={{
            display: 'flex', alignItems: 'center', gap: 5,
            height: 28, padding: '0 4px 0 7px',
            borderRadius: 7,
            border: `1px solid ${color}40`,
            background: `${color}12`,
            animation: justAttached ? 'pill-pop 0.38s cubic-bezier(0.34,1.56,0.64,1)' : 'none',
            flexShrink: 0, maxWidth: 230,
            transition: 'border-color 0.2s',
          }}
        >
          {/* Coloured file-type icon */}
          <span style={{ fontSize: 12, lineHeight: 1, flexShrink: 0, userSelect: 'none' }}>{icon}</span>

          {/* Name + size stack */}
          <div style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
            <span style={{
              fontSize: '0.64rem', fontWeight: 600, color,
              whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
              maxWidth: 140, lineHeight: 1.2, letterSpacing: '0.01em',
            }}>
              {truncateName(attachedFile.name)}
            </span>
            <span style={{ fontSize: '0.55rem', color: '#475569', lineHeight: 1.2 }}>
              {formatSize(attachedFile.size)}
            </span>
          </div>

          {/* Thin divider */}
          <div style={{ width: 1, height: 14, background: `${color}35`, flexShrink: 0, marginLeft: 2 }} />

          {/* Delete ✕ */}
          <button
            onClick={e => { e.stopPropagation(); onDelete() }}
            onMouseEnter={() => setHoverDelete(true)}
            onMouseLeave={() => setHoverDelete(false)}
            title="Remove file"
            style={{
              width: 18, height: 18, borderRadius: 4, flexShrink: 0,
              border: 'none',
              background: hoverDelete ? 'rgba(239,68,68,0.2)' : 'transparent',
              cursor: 'pointer', display: 'flex', alignItems: 'center',
              justifyContent: 'center', transition: 'background 0.14s', padding: 0,
            }}
          >
            <svg width="8" height="8" viewBox="0 0 10 10" fill="none"
              stroke={hoverDelete ? '#ef4444' : '#64748b'}
              strokeWidth="2" strokeLinecap="round"
              style={{ transition: 'stroke 0.14s' }}
            >
              <line x1="1" y1="1" x2="9" y2="9"/>
              <line x1="9" y1="1" x2="1" y2="9"/>
            </svg>
          </button>
        </div>
      )}
    </>
  )
}

// ── Main ─────────────────────────────────────────────────────────────────
export default function DemoPage() {
  const [tab, setTab]                       = useState<'firewall' | 'evaluate'>('firewall')
  const [provider, setProvider]             = useState('Ollama (llama3)')
  const [running, setRunning]               = useState(false)
  const [query, setQuery]                   = useState('')
  const [tokens, setTokens]                 = useState<TokenSpan[]>([])
  const [corrections, setCorrections]       = useState<CorrectionEntry[]>([])
  const [claimsFound, setClaimsFound]       = useState(0)
  const [verified, setVerified]             = useState(0)
  const [hallucinations, setHallucinations] = useState(0)
  const [avgVerify, setAvgVerify]           = useState(0)
  const [pipeline, setPipeline]             = useState(0)
  const [attachedFile, setAttachedFile]     = useState<File | null>(null)
  const [isDragging, setIsDragging]         = useState(false)
  const [errorMsg, setErrorMsg]             = useState<string | null>(null)
  const dragCounter                         = useRef(0)
  const streamRef                           = useRef<HTMLDivElement>(null)
  const abortRef                            = useRef<AbortController | null>(null)

  useEffect(() => {
    if (streamRef.current) streamRef.current.scrollTop = streamRef.current.scrollHeight
  }, [tokens])

  // ── Drag & drop handlers ──
  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); dragCounter.current++
    if (e.dataTransfer.items?.length) setIsDragging(true)
  }, [])
  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); dragCounter.current--
    if (dragCounter.current === 0) setIsDragging(false)
  }, [])
  const handleDragOver  = useCallback((e: DragEvent<HTMLDivElement>) => { e.preventDefault() }, [])
  const handleDrop      = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault(); dragCounter.current = 0; setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) setAttachedFile(file)
  }, [])

  async function handleAnalyse() {
    if (!query.trim()) return
    // Cancel any ongoing stream
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setTokens([]); setCorrections([])
    setClaimsFound(0); setVerified(0); setHallucinations(0)
    setAvgVerify(0); setPipeline(0); setErrorMsg(null)
    setRunning(true)

    try {
      // 1. Upload file to vault first (if attached)
      if (attachedFile) {
        const form = new FormData()
        form.append('file', attachedFile)
        form.append('source_name', attachedFile.name)
        const upRes = await fetch(`${BACKEND_URL}/vault/upload`, { method: 'POST', body: form })
        if (!upRes.ok) {
          const err = await upRes.json().catch(() => ({ detail: 'Upload failed' }))
          setErrorMsg(`File upload error: ${err.detail ?? 'Unknown'}`)
          setRunning(false); return
        }
      }

      // 2. Stream from /chat
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
        signal: controller.signal,
      })

      if (!res.ok || !res.body) {
        const err = await res.json().catch(() => ({ detail: 'Backend error' }))
        setErrorMsg(`Backend error: ${err.detail ?? res.statusText}`)
        setRunning(false); return
      }

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const { event_type, data } = JSON.parse(line.slice(6))

            if (event_type === 'token') {
              setTokens(prev => [...prev.slice(-200), { id: uid++, text: data.text, kind: 'normal' }])
            }

            if (event_type === 'correction') {
              setTokens(prev => {
                // Mark the last few tokens that contain the corrected span
                const updated = [...prev]
                return [
                  ...updated,
                  { id: uid++, text: '', kind: 'corrected', original: data.original_claim, corrected: data.corrected_sentence },
                ]
              })
              setCorrections(prev => [{
                id: uid++,
                wrong: data.original_claim,
                right: data.corrected_sentence,
                src: data.source ?? 'vault',
                confidence: data.nli_confidence ?? 0,
                diff: data.diff_ratio ?? 0,
              }, ...prev.slice(0, 9)])
              setHallucinations(p => p + 1)
            }

            if (event_type === 'stats') {
              setClaimsFound(data.total_claims_detected ?? 0)
              setVerified(data.claims_verified ?? 0)
              setAvgVerify(Math.round(data.avg_verification_latency_ms ?? 0))
              setPipeline(Math.round(data.total_pipeline_latency_ms ?? 0))
            }

            if (event_type === 'error') {
              setErrorMsg(data.message ?? 'Unknown error')
            }

            if (event_type === 'done') {
              setClaimsFound(data.total_claims_detected ?? 0)
              setVerified(data.claims_verified ?? 0)
              setAvgVerify(Math.round(data.avg_verification_latency_ms ?? 0))
              setPipeline(Math.round(data.total_pipeline_latency_ms ?? 0))
            }
          } catch { /* malformed chunk, skip */ }
        }
      }
    } catch (e: unknown) {
      if (e instanceof Error && e.name !== 'AbortError') {
        setErrorMsg(`Connection error: ${e.message}. Is the backend running?`)
      }
    } finally {
      setRunning(false)
    }
  }

  function handleReset() {
    abortRef.current?.abort()
    setRunning(false); setTokens([]); setCorrections([])
    setClaimsFound(0); setVerified(0); setHallucinations(0)
    setAvgVerify(0); setPipeline(0); setErrorMsg(null)
  }

  const totalFixes = corrections.length

  return (
    <div style={{
      background: '#0c0e14', color: '#f1f5f9',
      fontFamily: "'Inter', system-ui, sans-serif",
      fontSize: 13, minHeight: '100vh', display: 'flex', flexDirection: 'column',
    }}>

      {/* ── TOPBAR ── */}
      <div style={{
        background: 'rgba(12,14,20,0.98)', borderBottom: '1px solid rgba(255,255,255,0.07)',
        padding: '0 20px', height: 48, display: 'flex', alignItems: 'center', gap: 10,
        flexShrink: 0, position: 'sticky', top: 0, zIndex: 50,
      }}>
        <a href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{
            width: 26, height: 26, background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
            borderRadius: 7, display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 12, flexShrink: 0,
          }}>⚡</div>
          <span style={{ fontWeight: 700, fontSize: '0.85rem', color: '#f1f5f9' }}>Veracity AI</span>
        </a>
        <span style={{
          fontSize: '0.6rem', fontWeight: 600, padding: '1px 6px',
          background: 'rgba(99,102,241,0.2)', color: '#6366f1',
          border: '1px solid rgba(99,102,241,0.3)', borderRadius: 4,
        }}>v2.1</span>
        <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.07)' }} />
        <span style={{ fontSize: '0.7rem', color: '#475569', flex: 1 }}>
          Adaptive Evaluation &amp; Self-Healing Firewall
        </span>
        <select value={provider} onChange={e => setProvider(e.target.value)} style={{
          padding: '3px 8px', borderRadius: 5, background: '#12151e',
          border: '1px solid rgba(255,255,255,0.08)', color: '#94a3b8',
          fontSize: '0.72rem', cursor: 'pointer', outline: 'none',
        }}>
          {PROVIDERS.map(p => <option key={p}>{p}</option>)}
        </select>
        {running && (
          <button onClick={() => setRunning(r => !r)} style={{
            padding: '4px 12px', borderRadius: 5, background: 'rgba(239,68,68,0.1)',
            border: '1px solid rgba(239,68,68,0.25)', color: '#ef4444',
            fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
          }}>⏸ Pause</button>
        )}
        {!running && tokens.length > 0 && (
          <button onClick={() => setRunning(true)} style={{
            padding: '4px 12px', borderRadius: 5, background: 'rgba(16,185,129,0.1)',
            border: '1px solid rgba(16,185,129,0.25)', color: '#10b981',
            fontSize: '0.7rem', fontWeight: 600, cursor: 'pointer',
          }}>▶ Resume</button>
        )}
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: '0.68rem', color: running ? '#10b981' : '#475569', fontWeight: 500 }}>
          <span style={{
            display: 'inline-block', width: 6, height: 6, borderRadius: '50%',
            background: running ? '#10b981' : '#475569',
            boxShadow: running ? '0 0 5px #10b981' : 'none',
            animation: running ? 'pulse 2s ease infinite' : 'none',
          }} />
          {running ? 'Firewall Online' : 'Standby'}
        </div>
      </div>

      {/* ── PAGE BODY ── */}
      <div style={{
        flex: 1, display: 'grid', gridTemplateColumns: '1fr 200px',
        gridTemplateRows: 'auto 1fr', gap: 12, padding: 16,
        maxWidth: 1100, margin: '0 auto', width: '100%',
      }}>

        {/* HERO */}
        <div style={{ gridColumn: '1 / -1', paddingBottom: 12, borderBottom: '1px solid rgba(255,255,255,0.07)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
            <span style={{
              fontSize: '0.6rem', fontWeight: 700, letterSpacing: '0.05em', textTransform: 'uppercase',
              color: '#6366f1', background: 'rgba(99,102,241,0.15)', border: '1px solid rgba(99,102,241,0.2)',
              padding: '2px 7px', borderRadius: 4,
            }}>Live Demo</span>
            <span style={{ color: '#475569', fontSize: '0.7rem' }}>·</span>
            <span style={{ fontSize: '0.68rem', color: '#475569' }}>Real-time Hallucination Firewall · {provider}</span>
          </div>
          <div style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.03em', lineHeight: 1.2 }}>
            Self-Healing <span style={{ color: '#6366f1' }}>AI Firewall</span>
          </div>
          <div style={{ fontSize: '0.75rem', color: '#475569', marginTop: 4 }}>
            Per-token entropy detection · NLI verification · Auto-correction before it reaches your screen
          </div>
        </div>

        {/* MAIN LEFT */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minWidth: 0 }}>

          {/* Mode tabs */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ display: 'flex', gap: 2, background: '#12151e', borderRadius: 8, padding: 3 }}>
              {(['firewall', 'evaluate'] as const).map(t => (
                <button key={t} onClick={() => setTab(t)} style={{
                  padding: '4px 12px', borderRadius: 6, fontSize: '0.72rem', fontWeight: 500,
                  cursor: 'pointer', border: 'none',
                  background: tab === t ? '#1a1f2e' : 'none',
                  color: tab === t ? '#f1f5f9' : '#94a3b8',
                  boxShadow: tab === t ? '0 1px 3px rgba(0,0,0,0.3)' : 'none',
                  transition: 'all 0.15s',
                }}>
                  {t === 'firewall' ? '🛡 Firewall' : '⚖️ Evaluate'}
                </button>
              ))}
            </div>
            <span style={{ fontSize: '0.66rem', color: '#475569' }}>
              {tab === 'firewall' ? 'Real-time hallucination correction' : 'Compare two models head-to-head'}
            </span>
          </div>

          {/* ── Query input box with drag-and-drop ── */}
          <div
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            style={{
              background: isDragging ? 'rgba(99,102,241,0.04)' : '#1a1f2e',
              border: isDragging ? '1.5px dashed rgba(99,102,241,0.55)' : '1px solid rgba(99,102,241,0.3)',
              borderRadius: 10, transition: 'all 0.17s ease', position: 'relative',
            }}
          >
            {/* Drag overlay */}
            {isDragging && (
              <div style={{
                position: 'absolute', inset: 0, borderRadius: 9, zIndex: 10, pointerEvents: 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                background: 'rgba(10,12,18,0.75)',
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                  </svg>
                  <span style={{ fontSize: '0.72rem', color: '#6366f1', fontWeight: 600, letterSpacing: '0.04em' }}>
                    Drop to attach
                  </span>
                </div>
              </div>
            )}

            {/* Input row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '9px 10px 9px 12px' }}>
              <span style={{ color: '#475569', fontSize: '0.8rem', fontFamily: 'monospace', flexShrink: 0 }}>›</span>
              <input
                value={query}
                onChange={e => setQuery(e.target.value)}
                placeholder="Ask about financial data, company metrics, market rates…"
                style={{
                  flex: 1, fontSize: '0.8rem', color: '#94a3b8',
                  background: 'none', border: 'none', outline: 'none', minWidth: 0,
                }}
              />
              <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>

                {/* ★ THE PERFECT FILE BUTTON ★ */}
                <FileButton
                  attachedFile={attachedFile}
                  onAttach={setAttachedFile}
                  onDelete={() => setAttachedFile(null)}
                  isDragging={isDragging}
                />

                <div style={{ width: 1, height: 16, background: 'rgba(255,255,255,0.07)' }} />

                <button
                  onClick={handleAnalyse}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '5px 13px', borderRadius: 6,
                    background: 'linear-gradient(135deg,#6366f1,#8b5cf6)',
                    color: 'white', border: 'none', fontSize: '0.72rem', fontWeight: 600,
                    cursor: 'pointer', whiteSpace: 'nowrap',
                    boxShadow: '0 2px 10px rgba(99,102,241,0.35)',
                    transition: 'box-shadow 0.15s, opacity 0.15s',
                  }}
                  onMouseEnter={e => { e.currentTarget.style.boxShadow = '0 3px 16px rgba(99,102,241,0.55)' }}
                  onMouseLeave={e => { e.currentTarget.style.boxShadow = '0 2px 10px rgba(99,102,241,0.35)' }}
                >
                  ▶ Analyse
                </button>
              </div>
            </div>

            {/* Subtle drop hint */}
            {!attachedFile && !isDragging && (
              <div style={{ padding: '0 12px 7px', display: 'flex', alignItems: 'center', gap: 4 }}>
                <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#2d3a4a" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                  <polyline points="17 8 12 3 7 8"/>
                  <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <span style={{ fontSize: '0.58rem', color: '#2d3a4a' }}>
                  drag &amp; drop a file, or click the paperclip to browse
                </span>
              </div>
            )}
          </div>

          {/* Token stream */}
          <div style={{
            background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.07)',
            borderRadius: 10, overflow: 'hidden', flex: 1, display: 'flex', flexDirection: 'column',
          }}>
            <div style={{
              padding: '8px 14px', borderBottom: '1px solid rgba(255,255,255,0.07)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            }}>
              <span style={{ fontSize: '0.65rem', fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                ⚡ Token Stream
              </span>
              <div style={{ display: 'flex', gap: 1 }}>
                {[
                  [10,'rgba(148,163,184,0.08)'],[8,'rgba(99,102,241,0.25)'],
                  [12,'rgba(148,163,184,0.08)'],[14,'rgba(239,68,68,0.7)'],
                  [10,'rgba(245,158,11,0.6)'],[8,'rgba(148,163,184,0.08)'],
                  [6,'rgba(99,102,241,0.25)'],[10,'rgba(148,163,184,0.08)'],
                  [16,'rgba(239,68,68,0.7)'],[8,'rgba(251,191,36,0.5)'],
                ].map(([w,bg], i) => (
                  <div key={i} style={{ height: 4, width: w as number, borderRadius: 1, background: bg as string }} />
                ))}
              </div>
            </div>
            <div ref={streamRef} style={{
              padding: 14, fontSize: '0.82rem', lineHeight: 1.9, flex: 1,
              color: '#94a3b8', fontFamily: 'monospace', overflowY: 'auto',
              minHeight: 160, maxHeight: 200,
            }}>
              {errorMsg && (
                <div style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)', borderRadius: 6, padding: '8px 12px', marginBottom: 8, fontSize: '0.75rem' }}>
                  ⚠️ {errorMsg}
                </div>
              )}
              {tokens.length === 0 && !errorMsg ? (
                <span style={{ color: '#334155' }}>Click ▶ Analyse to start — responses stream directly from your local Llama 3</span>
              ) : (
                tokens.map(tok => {
                  if (tok.kind === 'normal') return <span key={tok.id} style={{ color: '#94a3b8' }}>{tok.text}</span>
                  if (tok.kind === 'high-entropy') return (
                    <span key={tok.id} style={{ background: 'rgba(245,158,11,0.12)', borderBottom: '2px solid rgba(245,158,11,0.4)', color: '#fbbf24' }}>
                      {tok.text}
                    </span>
                  )
                  if (tok.kind === 'corrected') return (
                    <span key={tok.id}>
                      <span style={{ textDecoration: 'line-through', color: 'rgba(239,68,68,0.7)', fontSize: '0.75rem', marginRight: 2 }}>{tok.original}</span>
                      <span style={{ background: 'rgba(16,185,129,0.15)', border: '1px solid rgba(16,185,129,0.3)', color: '#10b981', borderRadius: 3, padding: '0 3px', fontWeight: 600 }}>{tok.corrected}</span>
                    </span>
                  )
                  return null
                })
              )}
              {running && tokens.length > 0 && (
                <span style={{ display: 'inline-block', width: 2, height: '0.85em', background: '#6366f1', marginLeft: 1, verticalAlign: 'text-bottom', animation: 'cursor-blink 1s step-end infinite' }} />
              )}
            </div>
          </div>

          {/* Corrections log */}
          <div style={{ background: '#1a1f2e', border: '1px solid rgba(16,185,129,0.2)', borderRadius: 8, overflow: 'hidden' }}>
            <div style={{
              padding: '6px 12px', borderBottom: '1px solid rgba(16,185,129,0.1)',
              fontSize: '0.62rem', fontWeight: 600, color: '#10b981',
              textTransform: 'uppercase', letterSpacing: '0.05em',
              display: 'flex', justifyContent: 'space-between',
            }}>
              <span>✓ Auto-Corrections</span>
              <span style={{ color: '#475569', fontWeight: 400 }}>{totalFixes} fix{totalFixes !== 1 ? 'es' : ''} applied</span>
            </div>
            {corrections.length === 0 ? (
              <div style={{ padding: '9px 12px', fontSize: '0.68rem', color: '#334155', fontFamily: 'monospace' }}>
                No corrections yet — all claims verified.
              </div>
            ) : corrections.slice(0, 4).map(c => (
              <div key={c.id} style={{ padding: '7px 12px', borderBottom: '1px solid rgba(255,255,255,0.05)', fontSize: '0.68rem', fontFamily: 'monospace' }}>
                <span style={{ color: '#ef4444', textDecoration: 'line-through' }}>{c.wrong}</span>
                <span style={{ color: '#475569', margin: '0 5px' }}>→</span>
                <span style={{ color: '#10b981', fontWeight: 600 }}>{c.right}</span>
                <span style={{ color: '#475569', fontSize: '0.63rem', display: 'block', marginTop: 2 }}>
                  via {c.src} · NLI confidence {c.confidence.toFixed(2)} · diff {c.diff.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* ── STATS SIDEBAR ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div style={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: '0.6rem', fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8, display: 'flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 5, height: 5, borderRadius: '50%', background: running ? '#10b981' : '#475569', boxShadow: running ? '0 0 5px #10b981' : 'none', display: 'inline-block' }} />
              Firewall Status
            </div>
            {[
              { label: 'State',     value: running ? 'ACTIVE' : 'STANDBY', color: running ? '#10b981' : '#475569' },
              { label: 'Provider',  value: provider,                        color: '#6366f1' },
              { label: 'Detection', value: 'Entropy',                       color: '#60a5fa' },
            ].map(r => (
              <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontSize: '0.68rem', color: '#475569' }}>{r.label}</span>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, fontFamily: 'monospace', color: r.color }}>{r.value}</span>
              </div>
            ))}
          </div>
          <div style={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: '0.6rem', fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>📊 Pipeline Metrics</div>
            {[
              { label: 'Claims found',   value: claimsFound,    color: '#f59e0b' },
              { label: 'Verified',       value: verified,       color: '#10b981' },
              { label: 'Hallucinations', value: hallucinations, color: '#ef4444' },
              { label: 'Corrections',    value: totalFixes,     color: '#ef4444' },
            ].map(r => (
              <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontSize: '0.68rem', color: '#475569' }}>{r.label}</span>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, fontFamily: 'monospace', color: r.color }}>{r.value}</span>
              </div>
            ))}
          </div>
          <div style={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: '0.6rem', fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>⚡ Latency</div>
            {[
              { label: 'Avg verify', value: `${avgVerify}ms`, color: '#10b981' },
              { label: 'Pipeline',   value: `${pipeline}ms`,  color: '#60a5fa' },
              { label: 'Target',     value: '<200ms',          color: '#6366f1' },
            ].map(r => (
              <div key={r.label} style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
                <span style={{ fontSize: '0.68rem', color: '#475569' }}>{r.label}</span>
                <span style={{ fontSize: '0.72rem', fontWeight: 700, fontFamily: 'monospace', color: r.color }}>{r.value}</span>
              </div>
            ))}
          </div>
          <div style={{ background: '#1a1f2e', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 8, padding: 12 }}>
            <div style={{ fontSize: '0.6rem', fontWeight: 600, color: '#475569', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>ℹ️ How it works</div>
            <p style={{ fontSize: '0.65rem', color: '#475569', lineHeight: 1.6, marginBottom: 5 }}>
              Per-token Shannon entropy from {provider} logprobs flags uncertain spans.
            </p>
            <p style={{ fontSize: '0.65rem', color: '#475569', lineHeight: 1.6 }}>
              NLI classifier verifies claims. Contradictions trigger REVERSE rewriter.
            </p>
          </div>
          <button onClick={handleReset} style={{
            padding: '7px', borderRadius: 6, background: 'none',
            border: '1px solid rgba(255,255,255,0.07)', color: '#475569',
            fontSize: '0.68rem', fontWeight: 600, cursor: 'pointer',
            fontFamily: 'monospace', letterSpacing: '0.04em', transition: 'all 0.15s',
          }}>
            ↺ Reset session
          </button>
        </div>
      </div>

      {/* ── FOOTER ── */}
      <div style={{ borderTop: '1px solid rgba(255,255,255,0.07)', padding: '8px 20px', display: 'flex', justifyContent: 'space-between', flexShrink: 0 }}>
        <span style={{ fontSize: '0.62rem', color: '#475569' }}>Veracity AI · Self-Healing LLM Firewall</span>
        <span style={{ fontSize: '0.62rem', color: '#475569' }}>Powered by {provider} · NLI + Entropy Detection</span>
      </div>

      <style>{`
        @keyframes pulse       { 0%,100%{opacity:1}  50%{opacity:0.4} }
        @keyframes cursor-blink{ 0%,100%{opacity:1}  50%{opacity:0}   }
        @keyframes pill-pop    {
          0%  { transform:scale(0.7);  opacity:0; }
          65% { transform:scale(1.07); opacity:1; }
          100%{ transform:scale(1);   opacity:1; }
        }
        *{ box-sizing:border-box; }
        button{ font-family:inherit; }
        input::placeholder{ color:#334155; }
        ::-webkit-scrollbar      { width:3px; }
        ::-webkit-scrollbar-track{ background:#12151e; }
        ::-webkit-scrollbar-thumb{ background:#1e2435; border-radius:2px; }
        select option            { background:#12151e; }
      `}</style>
    </div>
  )
}