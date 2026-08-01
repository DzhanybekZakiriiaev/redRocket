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
import ModelIO from './panels/ModelIO'
import { loadTraceNamed, type Derived } from './trace'
import { useStore } from './store'
import { clock, startClock } from './clock'

export default function App() {
  const trace = useStore(s => s.trace)
  const setTraces = useStore(s => s.setTraces)
  const cycleGlobeMode = useStore(s => s.cycleGlobeMode)
  const toggleAdvisor = useStore(s => s.toggleAdvisor)
  const toggleModelIO = useStore(s => s.toggleModelIO)
  const toggleProjector = useStore(s => s.toggleProjector)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'm' || e.key === 'M') cycleGlobeMode()
      if (e.key === 'v' || e.key === 'V') toggleAdvisor()   // v = versus: swap the brain
      if (e.key === 'g' || e.key === 'G') toggleModelIO()   // g = gemma: show the calculation
      if (e.key === 'p' || e.key === 'P') toggleProjector() // p = projector: lift the faint inks
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [cycleGlobeMode, toggleAdvisor, toggleModelIO, toggleProjector])

  useEffect(() => {
    let alive = true
    // Load every arm of the comparison. They share one scenario, so their
    // frames line up and the toggle swaps only the diagnoses. `peer` is
    // optional: it only exists once tools/run_gemma_peer.py has been run, and
    // a clone without it must still boot, so its failure is swallowed rather
    // than taking the whole load down with it.
    const optional = (p: Promise<Derived>) => p.then(t => t, () => undefined)
    Promise.all([
      loadTraceNamed('gemma'),
      loadTraceNamed('bayes'),
      optional(loadTraceNamed('peer')),
    ])
      .then(([gemma, bayes, peer]) => {
        if (!alive) return
        clock.nFrames = gemma.nFrames
        clock.playing = useStore.getState().playing
        clock.speed = useStore.getState().speed
        clock.ready = true          // gate lifts → the transport advances
        setTraces({ gemma, bayes, peer })
        startClock()
      })
      .catch(err => console.error('[trace] failed to load advisor traces', err))
    return () => { alive = false }
  }, [setTraces])

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
      <ModelIO />
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
