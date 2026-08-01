/* [E] Roster panel — the all-assets overview.
   Rows are React DOM, rendered once. The LEDs are poked imperatively from
   the rAF loop; nothing here re-renders per frame. */

import { useEffect, useRef } from 'react'
import { useStore } from '../store'
import { onFrame } from '../clock'
import { stateColour } from '../MapField'
import type { DState } from '../trace'

const ROW = 34

function readPal() {
  const cs = getComputedStyle(document.documentElement)
  const g = (n: string) => cs.getPropertyValue(n).trim()
  return {
    ink: g('--ink'), dim: g('--ink-dim'), faint: g('--ink-faint'),
    accent: g('--accent'), alert: g('--alert'), neutral: g('--neutral'), bg: g('--bg'),
    // warn/route are part of Pal and stateColour reads --warn for UNCERTAIN.
    // Omitting them here handed stateColour an undefined colour, so the roster
    // LED silently stopped flickering on an unsettled posterior.
    warn: g('--warn') || '#F5A623', route: g('--route') || '#9B7BFF',
    gs: g('--gs') || '#46C46A',
    dimA: 0.45, faintA: 0.18,
  }
}

export default function Roster() {
  const trace = useStore(s => s.trace)
  const selected = useStore(s => s.selected)
  const select = useStore(s => s.select)
  const leds = useRef<(HTMLDivElement | null)[]>([])
  const words = useRef<(HTMLDivElement | null)[]>([])

  useEffect(() => {
    if (!trace) return
    let pal = readPal()
    const unsubPal = useStore.subscribe(() => { pal = readPal() })
    const un = onFrame((frame, wall) => {
      const fi = Math.round(Math.max(0, Math.min(trace.nFrames - 1, frame)))
      trace.sats.forEach((s, i) => {
        const st: DState = trace.samples[s][fi].state
        const el = leds.current[i]
        if (el) el.style.background = stateColour(st, pal, wall, i)
        const w = words.current[i]
        if (w && w.textContent !== st) w.textContent = st
      })
    })
    return () => { un(); unsubPal() }
  }, [trace])

  if (!trace) return null

  return (
    <div className="panel" style={{ width: 88, height: '100%', overflow: 'hidden', borderRight: 'none' }}>
      <div className="px-[9px] pt-[7px] pb-[5px]">
        <div className="t-title" style={{ fontSize: 11 }}>ASSETS</div>
        <div className="t-sub" style={{ fontSize: 7, letterSpacing: '0.14em' }}>TRACKED · {trace.sats.length}</div>
      </div>
      <div className="rule mx-[9px]" style={{ background: 'var(--ink-dim)' }} />

      <div className="mt-[3px]">
        {trace.sats.map((s, i) => {
          const isSel = s === selected
          return (
            <div
              key={s}
              onClick={() => select(s)}
              className="relative hair-b"
              style={{
                height: ROW,
                background: i % 2 === 0 ? 'rgba(232,234,237,0.03)' : 'transparent',
                outline: isSel ? '1px solid var(--accent)' : 'none',
                outlineOffset: -1,
                cursor: 'crosshair',
                paddingLeft: 18,
                display: 'flex', flexDirection: 'column', justifyContent: 'center',
              }}
            >
              {/* status LED */}
              <div
                ref={el => { leds.current[i] = el }}
                style={{ position: 'absolute', left: 8, top: ROW / 2 - 3, width: 5, height: 5, background: 'var(--ink-dim)' }}
              />
              <div className="t-body" style={{ color: isSel ? 'var(--ink)' : undefined }}>{s}</div>
              <div ref={el => { words.current[i] = el }} className="t-micro dim">NOMINAL</div>
            </div>
          )
        })}
      </div>

    </div>
  )
}
