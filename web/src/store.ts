import { create } from 'zustand'
import type { Derived } from './trace'
import type { GlobeMode } from './globe'

interface S {
  trace: Derived | null
  selected: string
  playing: boolean
  speed: 0.25 | 0.5 | 1 | 2 | 8
  projector: boolean
  globeMode: GlobeMode
  setTrace: (t: Derived) => void
  select: (id: string) => void
  setPlaying: (p: boolean) => void
  setSpeed: (s: 0.25 | 0.5 | 1 | 2 | 8) => void
  toggleProjector: () => void
  setGlobeMode: (m: GlobeMode) => void
  cycleGlobeMode: () => void
}

export const useStore = create<S>((set) => ({
  trace: null,
  selected: 'SAT02',      // the asset that carries the demo's long ambiguity
  playing: true,
  speed: 1,
  projector: false,
  globeMode: 'wireframe',
  setTrace: (t) => set({ trace: t }),
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
