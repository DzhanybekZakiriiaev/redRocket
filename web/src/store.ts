import { create } from 'zustand'
import type { AdvisorArm, Derived } from './trace'
import type { GlobeMode } from './globe'

type AdvisorName = AdvisorArm

/** Cycle order for the V key. Arms whose trace was not built are skipped. */
const ARM_ORDER: AdvisorArm[] = ['gemma', 'peer', 'bayes']

interface S {
  trace: Derived | null                                  // the ACTIVE arm's trace
  traces: Partial<Record<AdvisorArm, Derived>> | null    // every arm that loaded
  advisor: AdvisorName
  selected: string
  playing: boolean
  speed: 0.25 | 0.5 | 1 | 2 | 8
  projector: boolean
  globeMode: GlobeMode
  /** the model-I/O overlay — "show the calculation behind the screen" */
  modelIO: boolean
  /** index into raw.faults of the scenario last jumped to, for button highlight */
  scenario: number | null
  /** the scenario picker overlay */
  scenarioPicker: boolean
  setScenarioPicker: (v: boolean) => void
  toggleModelIO: () => void
  setModelIO: (v: boolean) => void
  setScenario: (i: number | null) => void
  setTrace: (t: Derived) => void
  setTraces: (t: Partial<Record<AdvisorArm, Derived>>) => void
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
  modelIO: false,
  scenario: null,
  scenarioPicker: false,
  setScenarioPicker: (v) => set({ scenarioPicker: v }),
  toggleModelIO: () => set((st) => ({ modelIO: !st.modelIO })),
  setModelIO: (v) => set({ modelIO: v }),
  setScenario: (i) => set({ scenario: i }),
  setTrace: (t) => set({ trace: t }),
  // Every arm runs the identical scenario, so the traces are frame-aligned.
  // Switching arm swaps only the diagnoses; globe/faults stay put.
  //
  // Defaults to 'gemma' (reasoning-first) rather than the newest arm. 'peer'
  // trades 15 points of accuracy for the ability to express uncertainty; that
  // is a deliberate demo choice, not something to inherit from load order.
  setTraces: (t) => set({
    traces: t,
    trace: t.gemma ?? t.peer ?? t.bayes ?? null,
    advisor: t.gemma ? 'gemma' : t.peer ? 'peer' : 'bayes',
  }),
  setAdvisor: (a) => set((st) => (st.traces?.[a] ? { advisor: a, trace: st.traces[a]! } : {})),
  toggleAdvisor: () => set((st) => {
    if (!st.traces) return {}
    const avail = ARM_ORDER.filter(a => st.traces![a])
    if (avail.length < 2) return {}
    const next = avail[(avail.indexOf(st.advisor) + 1) % avail.length]
    return { advisor: next, trace: st.traces![next]! }
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
