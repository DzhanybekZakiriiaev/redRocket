/* ────────────────────────────────────────────────────────────────────────
   RightPanel.tsx — the two right-edge annotation cards.

     • EVENT STREAM   — top-right, populates downward: the diagnosis + evidence
       narrative (who went uncertain, converged, asserted a fault, passed
       evidence).
     • LINK ACTIVITY  — bottom-right: every contact open right now (all of them,
       no cap) — the continuous inter-node comms.

   Shared classes (.panel, .t-title, .dim, .acc). Slow timer, never per frame.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect, useState } from 'react'
import { useStore } from '../store'
import { clock } from '../clock'
import { tPlus } from '../trace'

function kindColor(k: string): string {
  if (k === 'FAULT') return 'var(--alert)'
  if (k === 'RESOLVED' || k === 'GOSSIP') return 'var(--accent)'
  if (k === 'CLEAR') return 'var(--ink-dim)'
  return 'var(--neutral)'
}

const card: React.CSSProperties = { padding: '8px 10px' }

export default function RightPanel() {
  const trace = useStore(s => s.trace)
  const selected = useStore(s => s.selected)
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 1e9), 150)
    return () => clearInterval(id)
  }, [])

  if (!trace) return null
  const fi = Math.max(0, Math.min(trace.nFrames - 1, Math.round(clock.frame)))
  const epoch = trace.meta.epoch_ms

  const RATE: Record<string, string> = { isl: '25 MB/S', downlink: '160 MB/S' }
  const activeLinks = trace.contactsByFrame[fi].map(i => trace.raw.contacts[i])

  type Row = { f: number; kind: string; lines: string[] }
  const rows: Row[] = []
  for (let e = trace.eventAtFrame[fi]; e >= 0 && rows.length < 4; e--) {
    const ev = trace.events[e]
    rows.push({ f: ev.f, kind: ev.kind, lines: ev.lines })
  }
  for (let k = fi; k >= Math.max(0, fi - 12) && rows.length < 10; k--)
    for (const gi of trace.gossipByFrame[k]) {
      const g = trace.raw.gossip[gi]
      rows.push({ f: k, kind: 'GOSSIP', lines: [`${g.f} → ${g.to}   PASSED ${g.n} OBS`] })
    }
  rows.sort((a, b) => b.f - a.f)
  const feed = rows.slice(0, 4)

  return (
    <>
      {/* ── EVENT STREAM — top right, populates downward ── */}
      <div style={{ position: 'fixed', right: 20, top: 46, bottom: 'calc(50% + 6px)', width: 262, zIndex: 20 }}>
        <div className="panel" style={{ ...card, height: '100%', overflow: 'hidden' }}>
          <div className="t-title">EVENT STREAM</div>
          <div className="t-sub" style={{ marginBottom: 7 }}>AUTONOMOUS DIAGNOSIS &amp; EVIDENCE</div>
          {feed.length === 0 && <div className="t-micro faint">NOMINAL — NO EVENTS</div>}
          {feed.map((r, i) => (
            <div key={i} style={{ display: 'flex', gap: 7, marginBottom: 6 }}>
              <div style={{ width: 3, flexShrink: 0, background: kindColor(r.kind) }} />
              <div style={{ flex: 1 }}>
                <div className="t-micro" style={{ color: kindColor(r.kind) }}>T+{tPlus(trace.timeAt(r.f), epoch).slice(0, 5)}  {r.kind === 'GOSSIP' ? 'EVIDENCE' : r.kind}</div>
                {r.lines.map((l, j) => (
                  <div key={j} className={j === 0 ? 't-nine' : 't-nine dim'}>{l.replace(/^\/\/\s*/, '')}</div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ── LINK ACTIVITY — bottom right, ALL open links ── */}
      <div style={{ position: 'fixed', right: 20, top: 'calc(50% + 6px)', bottom: 46, width: 262, zIndex: 20 }}>
        <div className="panel" style={{ ...card, height: '100%', overflow: 'hidden' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span className="t-title">LINK ACTIVITY</span>
            <span className="t-micro acc">{activeLinks.length} OPEN</span>
          </div>
          <div className="t-sub" style={{ marginBottom: 7 }}>CONTACTS OPEN NOW</div>
          {activeLinks.length === 0 && <div className="t-micro faint">— NO OPEN LINKS —</div>}
          {activeLinks.map((c, i) => {
            const mine = c.a === selected || c.b === selected
            return (
              <div key={i} className="t-micro" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                <span style={{ color: mine ? 'var(--ink)' : 'var(--ink-dim)' }}>{c.a} ↔ {c.b}</span>
                <span style={{ color: c.k === 'downlink' ? 'var(--neutral)' : 'var(--accent)' }}>{c.k === 'downlink' ? 'DOWNLINK' : 'ISL'} · {RATE[c.k]}</span>
              </div>
            )
          })}
        </div>
      </div>
    </>
  )
}
