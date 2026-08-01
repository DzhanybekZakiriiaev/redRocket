import { create } from 'zustand'
import type { Derived } from './trace'
import type { GlobeMode } from './globe'

type AdvisorName = 'gemma' | 'bayes'

interface S {
  trace: Derived | null                                  // the ACTIVE advisor's trace
  traces: { gemma: Derived; bayes: Derived } | null      // both, for the head-to-head
  advisor: AdvisorName
  selected: string
  playing: boolean
  speed: 0.25 | 0.5 | 1 | 2 | 8
  projector: boolean
  globeMode: GlobeMode
  setTrace: (t: Derived) => void
  setTraces: (gemma: Derived, bayes: Derived) => void
  setAdvisor: (a: AdvisorName) => void
  toggleAdvisor: () => void
  select: (id: string) => void
  setPlaying: (p: boolean) => void
  setSpeed: (s: 0.25 | 0.5 | 1 | 2 | 8) => void
  toggleProjector: () => void
  setGlobeMode: (m: GlobeMode) => void
  cycleGlobeMode: () => void
}

export const useStore = create<S>((set) => ({
  trace: null,
  traces: null,
  advisor: 'gemma',       // Gemma is the spotlight; Bayes is the comparison
  selected: 'SAT02',      // the asset that carries the demo's long ambiguity
  playing: true,
  speed: 1,
  projector: false,
  globeMode: 'wireframe',
  setTrace: (t) => set({ trace: t }),
  // Both arms run the identical scenario, so their traces are frame-aligned.
  // Switching advisor swaps only the diagnoses; globe/faults stay put.
  setTraces: (gemma, bayes) => set({ traces: { gemma, bayes }, trace: gemma, advisor: 'gemma' }),
  setAdvisor: (a) => set((st) => (st.traces ? { advisor: a, trace: st.traces[a] } : {})),
  toggleAdvisor: () => set((st) => {
    if (!st.traces) return {}
    const a: AdvisorName = st.advisor === 'gemma' ? 'bayes' : 'gemma'
    return { advisor: a, trace: st.traces[a] }
  }),
  select: (id) => set({ selected: id }),
  setPlaying: (p) => set({ playing: p }),
  setSpeed: (s) => set({ speed: s }),
  toggleProjector: () => set((st) => {
    const on = !st.projector
    document.documentElement.classList.toggle('projector', on)
    return { projector: on }
  }),
  setGlobeMode: (m) => set({ globeMode: m }),
  cycleGlobeMode: () => set((st) => {
    const order: GlobeMode[] = ['mars', 'wireframe', 'earth']
    return { globeMode: order[(order.indexOf(st.globeMode) + 1) % order.length] }
  }),
}))
