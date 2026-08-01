/* ────────────────────────────────────────────────────────────────────────
   App.tsx — root component.

   Loads the single data source (/trace.json → Derived), primes the transport,
   and mounts the three live regions. Everything below reads the store and the
   clock; App is the only place that kicks either into motion.

   The clock loop only advances while `clock.ready` is true, so the globe stays
   frozen until the trace is derived — set it here, after loadTrace resolves.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect } from 'react'
import MapField from './MapField'
import Chrome from './Chrome'
import Roster from './panels/Roster'
import Inspector from './panels/Inspector'
import Timeline from './panels/Timeline'
import RightPanel from './panels/RightPanel'
import { loadTrace } from './trace'
import { useStore } from './store'
import { clock, startClock } from './clock'

export default function App() {
  const trace = useStore(s => s.trace)
  const setTrace = useStore(s => s.setTrace)
  const cycleGlobeMode = useStore(s => s.cycleGlobeMode)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'm' || e.key === 'M') cycleGlobeMode()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cycleGlobeMode])

  useEffect(() => {
    let alive = true
    loadTrace()
      .then(t => {
        if (!alive) return
        clock.nFrames = t.nFrames
        clock.playing = useStore.getState().playing
        clock.speed = useStore.getState().speed
        clock.ready = true          // gate lifts → the transport advances
        setTrace(t)
        startClock()
      })
      .catch(err => console.error('[trace] failed to load /trace.json', err))
    return () => { alive = false }
  }, [setTrace])

  return (
    <div style={{ position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden' }}>
      <MapField />
      <Chrome />
      <div style={{ position: 'fixed', left: 20, top: 46, bottom: 46, display: 'flex', alignItems: 'stretch', zIndex: 20 }}>
        <Roster />
        <Inspector />
      </div>
      <Timeline />
      <RightPanel />
      {!trace && (
        <div style={{
          position: 'absolute', inset: 0, display: 'grid', placeItems: 'center',
          font: '10px "Roboto Condensed", sans-serif', letterSpacing: '0.18em',
          color: 'var(--ink-dim, #888)', pointerEvents: 'none',
        }}>
          LOADING TRACE…
        </div>
      )}
    </div>
  )
}
