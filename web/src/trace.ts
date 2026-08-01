/* ────────────────────────────────────────────────────────────────────────
   trace.ts — the single data source.

   One fetch of /trace.json, then everything the renderer needs is derived
   once into frame-indexed lookup tables. Nothing here runs per frame; the
   draw loop only reads arrays.
   ──────────────────────────────────────────────────────────────────────── */

export type Cause =
  | 'nominal' | 'weather' | 'node_down' | 'pointing' | 'buffer' | 'stale_sched'

export interface Meta {
  epoch_ms: number
  grid_ms: number
  n_frames: number
  horizon_ms: number
  causes: Cause[]
  advisor: string
  config_hash?: string
}

export interface SatNode { id: string; type: 'sat' }
export interface GsNode {
  id: string; name: string; type: 'gs'
  lat: number; lon: number; operational: boolean
}
export type Node = SatNode | GsNode

export interface Contact { a: string; b: string; t0: number; t1: number; k: 'isl' | 'downlink' }
export interface Fault {
  t_start: number; t_end: number; target: string
  kind: Cause; severity: number; shift_ms: number
}
export interface Belief {
  t: number; d: number[]; a: string; tg: string; x: Cause; truth: Cause
  /** advisor's plain-English reason — empty for Bayes/Null, a sentence for Gemma */
  r?: string
}
export interface Gossip { t: number; f: string; to: string; n: number }
export interface Fail { t: number; a: string; b: string; m: 'silent' | 'degraded' | 'late' }

export interface RawTrace {
  meta: Meta
  nodes: Node[]
  positions: Record<string, [number, number, number][]>
  contacts: Contact[]
  faults: Fault[]
  beliefs: Record<string, Belief[]>
  gossip: Gossip[]
  fails: Fail[]
  strip: Record<string, string>
  summary?: Record<string, number | string>
}

/* ── Epistemic state — colour encodes this, never the cause ────────────── */
export type DState = 'NOMINAL' | 'UNCERTAIN' | 'RESOLVED' | 'FAULT' | 'STALE'

/**
 * Confidence threshold. Below this the posterior is treated as unsettled and
 * the node flickers. Tuned against this trace: 0.65 catches the three genuine
 * ambiguity episodes (SAT01 @09:45, SAT06 @15:00, SAT02 @18:40) while leaving
 * the confident diagnoses steady.
 */
export const CONF = 0.65

/** How long (in grid frames = minutes) a node holds RESOLVED teal after it
 *  collapses out of UNCERTAIN. Long enough to read at 8x. */
export const RESOLVE_HOLD = 90

/* Cause → the state word that sits under the asset chip. Text, never colour. */
export const CAUSE_WORD: Record<Cause, string> = {
  nominal:     'NOMINAL',
  weather:     'STATION WX',
  node_down:   'SILENT',
  pointing:    'DEGRADED',
  buffer:      'BUFFER',
  stale_sched: 'SCHED',
}

export const ACTION_WORD: Record<string, string> = {
  none: 'MONITOR', wait: 'HOLD', reroute: 'REROUTE', blacklist: 'BLACKLIST',
}

export interface SatSample {
  state: DState
  cause: Cause
  conf: number
  d: number[]
  action: string
  target: string
  truth: Cause
  /** advisor's plain-English reason — empty for Bayes, a sentence for Gemma */
  rationale: string
}

export interface Derived {
  meta: Meta
  raw: RawTrace
  sats: string[]
  gs: GsNode[]
  gsById: Record<string, GsNode>
  nFrames: number

  /** frame -> unix ms */
  timeAt: (f: number) => number
  /** interpolated [lat, lon, altKm] at fractional frame */
  posAt: (sat: string, f: number) => [number, number, number]

  /** contact index list per frame */
  contactsByFrame: number[][]
  /** gossip index list per frame */
  gossipByFrame: number[][]
  /** fail index list per frame */
  failsByFrame: number[][]
  /** fault index list per frame (ground truth) */
  faultsByFrame: number[][]

  /** sat -> per-frame derived epistemic sample (1441 entries, held from the
   *  5-minute belief grid) */
  samples: Record<string, SatSample[]>
  /** frame -> true when every node's posterior is settled */
  diagnosable: Uint8Array
  /** decoded contact strips, "A|B" -> per-frame 0/1 */
  strip: Record<string, Uint8Array>

  /** chronological narrative events for the alert box [G] */
  events: TraceEvent[]
  /** frame -> index of the most recent event at or before it, -1 if none */
  eventAtFrame: Int32Array
}

export interface TraceEvent {
  f: number
  kind: 'AMBIGUOUS' | 'RESOLVED' | 'FAULT' | 'CLEAR' | 'ACTION'
  node: string
  lines: string[]
}

/* ── RLE strip: "12T,3F,40T" -> Uint8Array, three lines ────────────────── */
export function decodeStrip(rle: string, n: number): Uint8Array {
  const out = new Uint8Array(n)
  let i = 0
  for (const run of rle.split(',')) {
    const k = parseInt(run, 10)
    const v = run[run.length - 1] === 'T' ? 1 : 0
    for (let j = 0; j < k && i < n; j++) out[i++] = v
  }
  return out
}

const pad2 = (n: number) => String(n).padStart(2, '0')

/** Mission-elapsed clock, "HH:MM:SS", from epoch. */
export function tPlus(ms: number, epoch: number): string {
  const s = Math.max(0, Math.floor((ms - epoch) / 1000))
  return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor((s % 3600) / 60))}:${pad2(s % 60)}`
}
export function hhmm(ms: number, epoch: number): string {
  const s = Math.max(0, Math.floor((ms - epoch) / 1000))
  return `${pad2(Math.floor(s / 3600))}:${pad2(Math.floor((s % 3600) / 60))}`
}

function stateOf(b: Belief): { state: DState; conf: number } {
  let conf = 0
  for (const p of b.d) if (p > conf) conf = p
  if (conf < CONF) return { state: 'UNCERTAIN', conf }
  return { state: b.x === 'nominal' ? 'NOMINAL' : 'FAULT', conf }
}

export function derive(raw: RawTrace): Derived {
  const { meta } = raw
  const N = meta.n_frames
  const E = meta.epoch_ms
  const G = meta.grid_ms

  const sats = raw.nodes.filter((n): n is SatNode => n.type === 'sat').map(n => n.id)
  const gs = raw.nodes.filter((n): n is GsNode => n.type === 'gs')
  const gsById: Record<string, GsNode> = {}
  for (const g of gs) gsById[g.id] = g

  const timeAt = (f: number) => E + f * G
  const f0 = (t: number) => Math.max(0, Math.min(N - 1, Math.round((t - E) / G)))

  /* positions, interpolated across the dateline correctly */
  const posAt = (sat: string, f: number): [number, number, number] => {
    const p = raw.positions[sat]
    const i = Math.max(0, Math.min(N - 1, Math.floor(f)))
    const j = Math.min(N - 1, i + 1)
    const u = Math.max(0, Math.min(1, f - i))
    const a = p[i], b = p[j]
    let dLon = b[1] - a[1]
    if (dLon > 180) dLon -= 360
    if (dLon < -180) dLon += 360
    return [a[0] + (b[0] - a[0]) * u, a[1] + dLon * u, a[2] + (b[2] - a[2]) * u]
  }

  /* frame-indexed buckets */
  const contactsByFrame: number[][] = Array.from({ length: N }, () => [])
  raw.contacts.forEach((c, idx) => {
    const s = f0(c.t0), e = f0(c.t1)
    for (let f = s; f <= e; f++) contactsByFrame[f].push(idx)
  })

  const gossipByFrame: number[][] = Array.from({ length: N }, () => [])
  raw.gossip.forEach((g, idx) => gossipByFrame[f0(g.t)].push(idx))

  const failsByFrame: number[][] = Array.from({ length: N }, () => [])
  raw.fails.forEach((x, idx) => failsByFrame[f0(x.t)].push(idx))

  const faultsByFrame: number[][] = Array.from({ length: N }, () => [])
  raw.faults.forEach((x, idx) => {
    const s = f0(x.t_start), e = f0(x.t_end)
    for (let f = s; f <= e; f++) faultsByFrame[f].push(idx)
  })

  /* strips */
  const strip: Record<string, Uint8Array> = {}
  for (const [k, v] of Object.entries(raw.strip)) strip[k] = decodeStrip(v, N)

  /* ── epistemic samples, expanded from the 5-minute belief grid ──────── */
  const samples: Record<string, SatSample[]> = {}
  const events: TraceEvent[] = []

  for (const s of sats) {
    const seq = raw.beliefs[s] ?? []
    const arr: SatSample[] = new Array(N)
    let bi = 0
    let lastUncertainEnd = -1e9
    let wasUncertain = false
    let prevState: DState | null = null
    let prevCause: Cause | null = null

    for (let f = 0; f < N; f++) {
      const t = timeAt(f)
      while (bi + 1 < seq.length && seq[bi + 1].t <= t) bi++
      const b = seq[bi]
      if (!b) {
        arr[f] = {
          state: 'NOMINAL', cause: 'nominal', conf: 1,
          d: [1, 0, 0, 0, 0, 0], action: 'none', target: '', truth: 'nominal',
          rationale: '',
        }
        continue
      }
      const { state: base, conf } = stateOf(b)

      if (base === 'UNCERTAIN') { wasUncertain = true; lastUncertainEnd = f }
      else if (wasUncertain) { wasUncertain = false }

      /* RESOLVED = a posterior that has just collapsed out of ambiguity. It
         is the payoff of the flicker, so it gets its own colour and window. */
      let state: DState = base
      if (base !== 'UNCERTAIN' && f - lastUncertainEnd > 0 && f - lastUncertainEnd <= RESOLVE_HOLD) {
        state = 'RESOLVED'
      }

      arr[f] = {
        state, conf, d: b.d, cause: b.x,
        action: b.a, target: b.tg, truth: b.truth,
        rationale: b.r ?? '',
      }

      /* narrative events on transition */
      if (state !== prevState || b.x !== prevCause) {
        if (state === 'UNCERTAIN' && prevState !== 'UNCERTAIN') {
          const top = b.d
            .map((p, i) => [p, meta.causes[i]] as [number, Cause])
            .sort((x, y) => y[0] - x[0])
            .slice(0, 2)
          events.push({
            f, kind: 'AMBIGUOUS', node: s,
            lines: [
              `// ${s} POSTERIOR UNSETTLED`,
              `// ${CAUSE_WORD[top[0][1]]} ${(top[0][0] * 100).toFixed(0)}% / ${CAUSE_WORD[top[1][1]]} ${(top[1][0] * 100).toFixed(0)}%`,
              `// AWAITING PEER EVIDENCE`,
            ],
          })
        } else if (state === 'RESOLVED' && prevState === 'UNCERTAIN') {
          events.push({
            f, kind: 'RESOLVED', node: s,
            lines: [
              `// ${s} BELIEF CONVERGED`,
              `// CAUSE ${CAUSE_WORD[b.x]} P=${(conf * 100).toFixed(0)}%`,
              `// POLICY ${ACTION_WORD[b.a] ?? b.a.toUpperCase()} VIA ${b.tg}`,
            ],
          })
        } else if (state === 'FAULT' && prevState !== 'FAULT' && prevState !== 'RESOLVED') {
          events.push({
            f, kind: 'FAULT', node: s,
            lines: [
              `// ${s} FAULT ASSERTED`,
              `// ${CAUSE_WORD[b.x]} P=${(conf * 100).toFixed(0)}%`,
              `// POLICY ${ACTION_WORD[b.a] ?? b.a.toUpperCase()} VIA ${b.tg}`,
            ],
          })
        } else if (state === 'NOMINAL' && (prevState === 'FAULT' || prevState === 'RESOLVED')) {
          events.push({
            f, kind: 'CLEAR', node: s,
            lines: [`// ${s} RETURNED TO NOMINAL`, `// LINK BUDGET RECOVERED`, `// NO FURTHER ACTION`],
          })
        }
        prevState = state
        prevCause = b.x
      }
    }
    samples[s] = arr
  }

  /* diagnosability — solid where no posterior in the constellation is split */
  const diagnosable = new Uint8Array(N)
  for (let f = 0; f < N; f++) {
    let ok = 1
    for (const s of sats) if (samples[s][f].state === 'UNCERTAIN') { ok = 0; break }
    diagnosable[f] = ok
  }

  events.sort((a, b) => a.f - b.f)
  const eventAtFrame = new Int32Array(N).fill(-1)
  {
    let e = -1
    for (let f = 0; f < N; f++) {
      while (e + 1 < events.length && events[e + 1].f <= f) e++
      eventAtFrame[f] = e
    }
  }

  return {
    meta, raw, sats, gs, gsById, nFrames: N,
    timeAt, posAt,
    contactsByFrame, gossipByFrame, failsByFrame, faultsByFrame,
    samples, diagnosable, strip, events, eventAtFrame,
  }
}

export async function loadTrace(): Promise<Derived> {
  const res = await fetch('/trace.json')
  if (!res.ok) throw new Error(`trace.json ${res.status}`)
  return derive(await res.json())
}

/** Load one advisor's baked trace, e.g. /trace_gemma.json. */
export async function loadTraceNamed(advisor: 'gemma' | 'bayes'): Promise<Derived> {
  const res = await fetch(`/trace_${advisor}.json`)
  if (!res.ok) throw new Error(`trace_${advisor}.json ${res.status}`)
  return derive(await res.json())
}

/** Diagnosis accuracy: of the belief samples whose reported link was genuinely
 *  faulted (truth ≠ nominal), how often did the top cause match the truth.
 *  Computed from the raw 5-minute grid so it matches the offline figure. */
export function accuracyOf(d: Derived): { correct: number; total: number; pct: number } {
  let correct = 0, total = 0
  for (const series of Object.values(d.raw.beliefs))
    for (const b of series)
      if (b.truth !== 'nominal') { total++; if (b.x === b.truth) correct++ }
  return { correct, total, pct: total ? correct / total : 0 }
}
