/* ────────────────────────────────────────────────────────────────────────
   Scenarios.tsx — the "simulate condition" control.

   A single button in the transport row; the list lives in a modal so it costs
   no permanent screen width. Every entry replays a fault the simulator really
   generated — the eight records in trace.faults, injected by sim/faults.py
   under seed 42. Selecting one seeks to just before onset, selects the asset
   that has to diagnose it, and drops to half speed.

   Nothing is synthesised at click time. The flicker, gossip and resolution
   that follow are the recorded output of that run, which is the point: a
   judge can scrub back and get exactly the same thing again.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect, useMemo } from 'react'
import { useStore } from '../store'
import { clock, seek } from '../clock'
import { CAUSE_WORD, tPlus, type Derived } from '../trace'

interface Scenario {
  i: number
  label: string
  site: string
  node: string
  frame: number
  t: number
  durMin: number
  severity: number
}

/** Which asset should be in the inspector when this fault opens. */
function subjectOf(d: Derived, faultIdx: number): string {
  const f = d.raw.faults[faultIdx]
  if (f.target.startsWith('SAT')) return f.target

  // A surface fault is diagnosed by whichever orbiter is talking to that site.
  // Prefer the one whose recorded ground truth actually names this cause.
  const G = d.meta.grid_ms, E = d.meta.epoch_ms
  const s = Math.max(0, Math.round((f.t_start - E) / G))
  const e = Math.min(d.nFrames - 1, Math.round((f.t_end - E) / G))
  for (let fr = s; fr <= e; fr++)
    for (const sat of d.sats) {
      const smp = d.samples[sat][fr]
      if (smp.truth === f.kind && smp.target === f.target) return sat
    }
  for (let fr = s; fr <= e; fr++)
    for (const idx of d.contactsByFrame[fr]) {
      const c = d.raw.contacts[idx]
      if (c.a === f.target) return c.b
      if (c.b === f.target) return c.a
    }
  return d.sats[0]
}

/** Land a few frames early so the onset is watched, not missed. */
const LEAD = 6

export default function Scenarios() {
  const trace = useStore(s => s.trace)
  const open = useStore(s => s.scenarioPicker)
  const setOpen = useStore(s => s.setScenarioPicker)
  const select = useStore(s => s.select)
  const setPlaying = useStore(s => s.setPlaying)
  const setSpeed = useStore(s => s.setSpeed)
  const setScenario = useStore(s => s.setScenario)
  const active = useStore(s => s.scenario)

  const scenarios = useMemo<Scenario[]>(() => {
    if (!trace) return []
    const G = trace.meta.grid_ms, E = trace.meta.epoch_ms
    return trace.raw.faults.map((f, i) => ({
      i,
      label: CAUSE_WORD[f.kind],
      site: f.target.startsWith('SAT')
        ? f.target
        : (trace.gsById[f.target]?.name ?? f.target.replace('GS_', '')),
      node: subjectOf(trace, i),
      frame: Math.max(0, Math.round((f.t_start - E) / G)),
      t: f.t_start,
      durMin: Math.round((f.t_end - f.t_start) / 60000),
      severity: f.severity,
    })).sort((a, b) => a.frame - b.frame)
  }, [trace])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, setOpen])

  if (!trace || scenarios.length === 0) return null

  const run = (s: Scenario) => {
    select(s.node)
    seek(Math.max(0, s.frame - LEAD))
    clock.speed = 0.5
    setSpeed(0.5)
    clock.playing = true
    setPlaying(true)
    setScenario(s.i)
    setOpen(false)
  }

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className={active !== null ? 'on' : ''}
        title="Replay one of the faults this run actually injected"
        style={{ whiteSpace: 'nowrap' }}
      >
        ⚑ RUN SCENARIO
      </button>

      {open && (
        <div
          onClick={() => setOpen(false)}
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(0,0,0,0.72)', display: 'grid', placeItems: 'center',
            padding: 24,
          }}
        >
          <div
            onClick={e => e.stopPropagation()}
            className="panel"
            style={{
              width: 'min(560px, 94vw)', maxHeight: '84vh', overflowY: 'auto',
              padding: '16px 18px', background: 'var(--bg, #07090b)',
              border: '1px solid var(--ink-dim)',
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <div>
                <span className="t-title" style={{ color: 'var(--alert)' }}>SIMULATE CONDITION</span>
                <span className="t-sub" style={{ marginLeft: 10 }}>{scenarios.length} RECORDED INJECTIONS</span>
              </div>
              <button onClick={() => setOpen(false)} className="t-micro"
                style={{ cursor: 'pointer', background: 'transparent', border: '1px solid var(--ink-faint)', padding: '2px 8px' }}>
                ESC ✕
              </button>
            </div>
            <div className="t-micro dim" style={{ marginTop: 5, marginBottom: 14, lineHeight: 1.5 }}>
              EACH ENTRY IS A FAULT sim/faults.py INJECTED UNDER SEED 42 — SELECTING ONE
              SEEKS TO ONSET AT 0.5×. NOTHING IS GENERATED AT CLICK TIME.
            </div>

            {scenarios.map(s => {
              const on = active === s.i
              return (
                <button
                  key={s.i}
                  onClick={() => run(s)}
                  style={{
                    width: '100%', cursor: 'pointer', textAlign: 'left',
                    background: on ? 'rgba(var(--ink-rgb),0.10)' : 'transparent',
                    border: `1px solid ${on ? 'var(--accent)' : 'var(--ink-faint)'}`,
                    padding: '8px 10px', marginBottom: 5,
                    display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                    gap: 10,
                  }}
                >
                  <span style={{ display: 'flex', gap: 8, alignItems: 'baseline', minWidth: 0 }}>
                    <span className="t-chip" style={{ color: on ? 'var(--accent)' : 'var(--ink)' }}>{s.label}</span>
                    <span className="t-micro dim">{s.site}</span>
                  </span>
                  <span style={{ display: 'flex', gap: 10, alignItems: 'baseline', flex: '0 0 auto' }}>
                    <span className="t-micro faint" style={{ fontSize: 8 }}>SEV {(s.severity * 100).toFixed(0)}%</span>
                    <span className="t-micro faint" style={{ fontSize: 8 }}>{s.durMin} MIN</span>
                    <span className="t-micro faint" style={{ fontSize: 8 }}>VIA {s.node}</span>
                    <span className="t-micro acc">T+{tPlus(s.t, trace.meta.epoch_ms).slice(0, 5)}</span>
                  </span>
                </button>
              )
            })}
          </div>
        </div>
      )}
    </>
  )
}
