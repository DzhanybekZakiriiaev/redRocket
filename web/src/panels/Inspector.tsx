/* ────────────────────────────────────────────────────────────────────────
   Inspector.tsx — [O] the selected asset's onboard diagnosis + data stream.

   Rendered as a flex child beside the roster list (see App), so the asset list
   and this detail read as one equal-height cluster. The body is three groups —
   DIAGNOSIS, DATA STREAM, TELEMETRY — spaced evenly down the full-height panel.
   Every group is fixed-height, so the layout never shifts as data streams in.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../store'
import { clock } from '../clock'
import { CAUSE_WORD, ACTION_WORD, tPlus, accuracyOf, type Cause } from '../trace'

const ll = (v: number, pos: string, neg: string) => `${Math.abs(v).toFixed(1)}°${v >= 0 ? pos : neg}`
const rowS: React.CSSProperties = { display: 'flex', justifyContent: 'space-between', marginBottom: 3 }

function Head({ children, color }: { children: React.ReactNode; color?: string }) {
  return <div className="t-micro" style={{ marginBottom: 5, color: color ?? 'var(--ink-dim)' }}>{children}</div>
}

export default function Inspector() {
  const trace = useStore(s => s.trace)
  const selected = useStore(s => s.selected)
  const [, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 1e9), 150)
    return () => clearInterval(id)
  }, [])

  const toggleAdvisor = useStore(s => s.toggleAdvisor)
  const setModelIO = useStore(s => s.setModelIO)
  const acc = useMemo(() => (trace ? accuracyOf(trace) : null), [trace])

  if (!trace) return null
  const fi = Math.max(0, Math.min(trace.nFrames - 1, Math.round(clock.frame)))
  const now = trace.timeAt(fi)
  const epoch = trace.meta.epoch_ms
  const advisor = (trace.meta.advisor || '').toUpperCase()
  const sample = trace.samples[selected]?.[fi]
  const causes = trace.meta.causes as Cause[]
  const bars = sample ? causes.map((c, i) => ({ c, p: sample.d[i] ?? 0 })).sort((a, b) => b.p - a.p) : []
  const correct = sample && sample.truth !== 'nominal' ? sample.cause === sample.truth : null

  const links = trace.contactsByFrame[fi]
    .map(i => trace.raw.contacts[i])
    .filter(c => c.a === selected || c.b === selected)
    .map(c => ({ peer: c.a === selected ? c.b : c.a, kind: c.k })).slice(0, 3)

  const signal: { f: number; peer: string; mode: string }[] = []
  for (let k = fi; k >= Math.max(0, fi - 10) && signal.length < 3; k--)
    for (const i of trace.failsByFrame[k]) {
      const x = trace.raw.fails[i]
      if (x.a === selected || x.b === selected) signal.push({ f: k, peer: x.a === selected ? x.b : x.a, mode: x.m })
    }

  const gossip: { f: number; dir: 'IN' | 'OUT'; peer: string; n: number }[] = []
  for (let k = fi; k >= Math.max(0, fi - 12) && gossip.length < 3; k--)
    for (const i of trace.gossipByFrame[k]) {
      const g = trace.raw.gossip[i]
      if (g.f === selected) gossip.push({ f: k, dir: 'OUT', peer: g.to, n: g.n })
      else if (g.to === selected) gossip.push({ f: k, dir: 'IN', peer: g.f, n: g.n })
    }

  const [lat, lon, alt] = trace.posAt(selected, fi)
  let nextT = Infinity
  for (const c of trace.raw.contacts) {
    if (c.a !== selected && c.b !== selected) continue
    if (c.t0 > now && c.t0 < nextT) nextT = c.t0
  }

  return (
    <div className="panel" style={{ width: 250, height: '100%', overflow: 'hidden', padding: '11px 13px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span className="t-title acc">{selected}</span>
        <button
          onClick={toggleAdvisor}
          className="t-micro"
          title="Click (or press V) to compare Gemma vs Bayes on the same faults"
          style={{
            cursor: 'pointer', background: 'transparent',
            border: '1px solid var(--ink-faint)', padding: '2px 7px',
            display: 'flex', gap: 6, alignItems: 'baseline', letterSpacing: '0.12em',
          }}
        >
          <span style={{ color: 'var(--accent)' }}>{advisor}</span>
          <span className="dim" style={{ fontSize: 8 }}>⇄</span>
          {acc && <span className="acc">{(acc.pct * 100).toFixed(1)}%</span>}
        </button>
      </div>
      <div className="t-sub">ONBOARD DIAGNOSIS</div>

      {sample ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between', paddingTop: 14 }}>
          {/* ── DIAGNOSIS ── */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 9 }}>
              <span className="t-chip" style={{ color: sample.state === 'FAULT' ? 'var(--alert)' : 'var(--ink)' }}>{CAUSE_WORD[sample.cause]}</span>
              <span className="t-val acc">{(sample.conf * 100).toFixed(0)}%</span>
            </div>
            {bars.map(({ c, p }, i) => (
              <div key={c} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 4 }}>
                <span className={i === 0 ? '' : 'dim'} style={{ width: 74, fontSize: 8, letterSpacing: '0.1em' }}>{CAUSE_WORD[c]}</span>
                <div style={{ flex: 1, height: 5, background: 'rgba(var(--ink-rgb),0.08)' }}>
                  <div style={{ width: `${Math.max(1, p * 100)}%`, height: '100%', background: i === 0 ? 'var(--accent)' : 'rgba(var(--ink-rgb),0.35)' }} />
                </div>
                <span className="dim" style={{ width: 22, textAlign: 'right', fontSize: 8 }}>{(p * 100).toFixed(0)}</span>
              </div>
            ))}
            {sample.rationale && (
              <div style={{ marginTop: 12 }}>
                <Head color="var(--accent)">MODEL REASONING</Head>
                <div className="t-body" style={{ fontStyle: 'italic', lineHeight: 1.4, color: 'var(--ink)' }}>
                  “{sample.rationale}”
                </div>
              </div>
            )}
            <button
              onClick={() => setModelIO(true)}
              className="t-micro"
              title="Show the exact prompt, evidence and reply behind this diagnosis (G)"
              style={{
                marginTop: 10, width: '100%', cursor: 'pointer', background: 'transparent',
                border: '1px solid var(--ink-faint)', padding: '4px 7px',
                display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                letterSpacing: '0.12em',
              }}
            >
              <span style={{ color: 'var(--accent)' }}>SHOW MODEL I/O</span>
              <span className="faint" style={{ fontSize: 8 }}>G</span>
            </button>
            <div style={{ marginTop: 12 }}>
              <Head color="var(--accent)">AUTONOMOUS RESPONSE</Head>
              <div className="t-body acc">
                {ACTION_WORD[sample.action] ?? sample.action.toUpperCase()}{sample.target ? ` → ${sample.target}` : ''}
              </div>
              {sample.truth !== 'nominal' && (
                <div className="t-micro dim" style={{ marginTop: 4 }}>
                  GROUND TRUTH {CAUSE_WORD[sample.truth]}{' '}
                  <span style={{ color: correct ? 'var(--accent)' : 'var(--alert)' }}>{correct ? '✓' : '✗'}</span>
                </div>
              )}
            </div>
          </div>

          {/* ── DATA STREAM ── */}
          <div>
            <div style={{ height: 1, background: 'var(--ink-faint)', marginBottom: 10 }} />
            <Head color="var(--ink)">ACTIVE LINKS</Head>
            <div style={{ minHeight: 42 }}>
              {links.length === 0 ? <div className="t-micro faint">— NO CONTACT —</div>
                : links.map((l, i) => (
                  <div key={i} className="t-micro" style={rowS}>
                    <span style={{ color: l.kind === 'downlink' ? 'var(--neutral)' : 'var(--accent)' }}>↔ {l.peer}</span>
                    <span className="dim">{l.kind === 'downlink' ? 'DOWNLINK' : 'ISL'}</span>
                  </div>
                ))}
            </div>

            <Head color="var(--alert)">SIGNAL</Head>
            <div style={{ minHeight: 42 }}>
              {signal.length === 0 ? <div className="t-micro faint">— NOMINAL —</div>
                : signal.map((s, i) => (
                  <div key={i} className="t-micro" style={rowS}>
                    <span className="alr">{s.mode.toUpperCase()}</span>
                    <span className="dim">{s.peer} · T+{tPlus(trace.timeAt(s.f), epoch).slice(0, 5)}</span>
                  </div>
                ))}
            </div>

            <Head color="var(--accent)">EVIDENCE EXCHANGE</Head>
            <div style={{ minHeight: 42 }}>
              {gossip.length === 0 ? <div className="t-micro faint">— NONE —</div>
                : gossip.map((g, i) => (
                  <div key={i} className="t-micro" style={rowS}>
                    <span className="acc">{g.dir} {g.peer}</span>
                    <span className="dim">{g.n} OBS · T+{tPlus(trace.timeAt(g.f), epoch).slice(0, 5)}</span>
                  </div>
                ))}
            </div>
          </div>

          {/* ── TELEMETRY ── */}
          <div>
            <div style={{ height: 1, background: 'var(--ink-faint)', marginBottom: 10 }} />
            <Head color="var(--neutral)">TELEMETRY</Head>
            <div className="t-micro" style={rowS}><span className="dim">POSITION</span><span>{ll(lat, 'N', 'S')} {ll(lon, 'E', 'W')}</span></div>
            <div className="t-micro" style={rowS}><span className="dim">ALTITUDE</span><span style={{ color: 'var(--neutral)' }}>{alt.toFixed(0)} KM</span></div>
            <div className="t-micro" style={rowS}><span className="dim">NEXT PASS</span><span style={{ color: 'var(--neutral)' }}>{nextT === Infinity ? '—' : `T+${tPlus(nextT, epoch).slice(0, 5)}`}</span></div>
            <div className="t-micro" style={rowS}><span className="dim">EARTH OWLT</span><span style={{ color: 'var(--neutral)' }}>12:30</span></div>
            <div className="t-micro" style={rowS}><span className="dim">EARTH SEES</span><span style={{ color: 'var(--neutral)' }}>T+{tPlus(now - 750000, epoch).slice(0, 5)}</span></div>
          </div>
        </div>
      ) : <div className="t-sub">NO DATA</div>}
    </div>
  )
}
