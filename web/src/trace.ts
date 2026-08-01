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
/**
 * The exact input the advisor was given for one decision, plus what came back.
 *
 * NOT derived in the browser — emitted by `tools/replay_evidence.py`, which
 * replays the baked run through the real GemmaAdvisor prompt builder. Absent
 * from a trace that predates that tool, in which case the model-I/O panel says
 * so rather than reconstructing an approximation and passing it off as real.
 */
export interface ModelIO {
  /** the link this decision was about */
  peer: string
  /** the literal string sent to the model, byte for byte */
  prompt: string
  /** the model's raw reply, before parsing */
  raw?: string
  /** the individual evidence fields, already split out of the prompt */
  f?: Record<string, string | number>
  /**
   * Elimination steps the model produced BEFORE naming a cause.
   *
   * Present only on the reasoning-first arm (sim/advisor/gemma_cot.py). These
   * are sampled ahead of the cause token, so they genuinely conditioned the
   * answer. The baseline arm's `rationale` is emitted AFTER the cause and is
   * therefore a justification written in hindsight — the UI must not present
   * the two as the same kind of thing.
   */
  steps?: string[]
  /**
   * Other nodes' diagnoses of the SAME target, as this node saw them.
   *
   * Present only on the peer cross-check arm. Each entry is one independent
   * node's conclusion — its cause, its confidence and the unix ms at which it
   * reached it — so the panel can show how stale each second opinion was.
   * Emitted as `[]` when no peer had diagnosed this target recently, which is
   * a meaningful state (the diagnosis went unchecked), not a missing field.
   */
  peers?: { node: string; cause: string; conf: number; t: number }[]
  /**
   * Did the peers agree with this node's cause?
   *
   * true when every peer named the same cause, or when there were no peers at
   * all. false means independent nodes reached different conclusions from the
   * same target — the reason the posterior below is spread across the
   * competing causes instead of committing to one.
   */
  consensus?: boolean
}

export interface Belief {
  t: number; d: number[]; a: string; tg: string; x: Cause; truth: Cause
  /** advisor's plain-English reason — empty for Bayes/Null, a sentence for Gemma */
  r?: string
  /** exact model input/output for this decision — present only in enriched traces */
  ev?: ModelIO
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
  weather:     'GS WEATHER',
  node_down:   'NODE DOWN',
  pointing:    'MISPOINT',
  buffer:      'BUFFER',
  stale_sched: 'SCHED DRIFT',
}

export const ACTION_WORD: Record<string, string> = {
  none: 'MONITOR', wait: 'HOLD', reroute: 'REROUTE', throttle: 'THROTTLE', blacklist: 'BLACKLIST',
}

/* Action → colour. Each policy response reads at a glance by hue:
   monitor = teal (calm) · hold/throttle = amber (holding back, caution) ·
   reroute = violet (steering onto a new path) · blacklist = red (dropping an
   asset). Kept beside ACTION_WORD so word and colour never drift apart. */
export const ACTION_COLOR: Record<string, string> = {
  none:      'var(--accent)',
  wait:      'var(--warn)',
  throttle:  'var(--warn)',
  reroute:   'var(--route)',
  blacklist: 'var(--alert)',
}
export const actionColor = (a: string): string => ACTION_COLOR[a] ?? 'var(--accent)'

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
  /** exact model input/output, when the trace carries it */
  ev?: ModelIO
  /** unix ms of the belief sample this frame was held from — the decision's
   *  own timestamp, not the frame's, so the model-I/O panel can label it */
  sampledAt: number
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

  /** true when this trace carries verbatim model I/O on at least one sample.
   *  Distinguishes "the trace predates tools/replay_evidence.py" from "this
   *  particular decision had no prompt because nothing was adverse" — the
   *  model-I/O panel has to say something different in each case. */
  hasEvidence: boolean

  /** true when this trace carries peer cross-check results on at least one
   *  sample. Distinguishes "this arm never cross-checked anything" — where the
   *  panel must stay silent rather than invent an absence — from "this
   *  particular decision found no peer to check against", which is worth
   *  saying out loud. */
  hasPeerCheck: boolean

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
          rationale: '', sampledAt: t,
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
        rationale: b.r ?? '', ev: b.ev, sampledAt: b.t,
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

  let hasEvidence = false
  for (const series of Object.values(raw.beliefs)) {
    if (series.some(b => b.ev)) { hasEvidence = true; break }
  }

  let hasPeerCheck = false
  for (const series of Object.values(raw.beliefs)) {
    if (series.some(b => b.ev?.peers !== undefined)) { hasPeerCheck = true; break }
  }

  return {
    meta, raw, sats, gs, gsById, nFrames: N,
    timeAt, posAt,
    contactsByFrame, gossipByFrame, failsByFrame, faultsByFrame,
    samples, diagnosable, strip, events, eventAtFrame,
    hasEvidence, hasPeerCheck,
  }
}

export async function loadTrace(): Promise<Derived> {
  const res = await fetch('/trace.json')
  if (!res.ok) throw new Error(`trace.json ${res.status}`)
  return derive(await res.json())
}

/** Cheapest check that a parsed body is a trace and not, say, the dev server's
 *  index.html fallback for a file that was never built. */
function isRawTrace(b: unknown): b is RawTrace {
  return typeof b === 'object' && b !== null && 'meta' in b && 'beliefs' in b
}

/** The three arms the display can hold at once. */
export type AdvisorArm = 'gemma' | 'peer' | 'bayes'

/**
 * Load one arm's baked trace.
 *
 *   gemma — reasoning-first (trace_gemma_cot.json), falling back to the
 *     baseline. Carries the elimination steps the model produced *before*
 *     naming a cause, and differs from the baseline on 2 of 2304 samples
 *     (0.1%), so it is the default: richest output at no measured cost.
 *
 *   peer — peer cross-check. Deliberately NOT the default. It is the only arm
 *     that can hold a belief open because independent nodes disagreed (98
 *     samples below the flicker threshold, against 0 for every other Gemma
 *     arm) — but it also scores 50.8% against gemma's 65.8%, because confident
 *     peers propagate their errors as readily as their corrections. That
 *     trade is the point of the arm, and it should be shown on purpose rather
 *     than inherited by whoever happens to load the page.
 *
 *   bayes — the no-LLM control.
 *
 * Missing files fall through, so a clone that has run none of the enrichment
 * tools still works.
 */
export async function loadTraceNamed(advisor: AdvisorArm): Promise<Derived> {
  const candidates =
    advisor === 'gemma' ? ['/trace_gemma_cot.json', '/trace_gemma.json']
    : advisor === 'peer' ? ['/trace_gemma_peer.json']
    : ['/trace_bayes.json']

  let lastStatus = 0
  for (const url of candidates) {
    let body: unknown
    try {
      const res = await fetch(url)
      if (!res.ok) { lastStatus = res.status; continue }
      body = await res.json()
    } catch {
      /* A dev server with SPA fallback answers a missing file with index.html
         and a 200, so status alone is not proof the trace exists. A body that
         will not parse as JSON means the same thing a 404 does: try the next. */
      lastStatus = 404
      continue
    }
    if (!isRawTrace(body)) { lastStatus = 404; continue }
    return derive(body)
  }
  throw new Error(`trace_${advisor}.json ${lastStatus}`)
}

/**
 * Diagnosis accuracy, scored on a test set that does not depend on the arm.
 *
 * The obvious version — count samples where `truth !== 'nominal'` — is not
 * comparable across arms, because `truth` describes whichever link that arm
 * chose to report. The Bayes filter names a link on every decision, so it
 * accumulates ~1089 scorable samples against Gemma's ~179, and most of the
 * extra ones are the same easy situation repeated. Comparing two percentages
 * computed over different populations is not a measurement, and the first
 * question a careful reader asks is "same test set?"
 *
 * So the question asked here is arm-independent: at each decision, was a fault
 * genuinely active on THIS node (from the trace's ground-truth fault records),
 * and if so did this arm name its cause? Every arm is then scored on the same
 * decision points with the same n.
 *
 * Known limit, worth stating rather than hiding: only faults whose target is a
 * satellite are counted (6 of the 8 injected in this run), because a fault at a
 * ground station has no single node to attribute it to. That exclusion applies
 * identically to every arm, so the comparison stays fair — it is simply not
 * "accuracy over all faults".
 */
export function accuracyOf(d: Derived): { correct: number; total: number; pct: number } {
  let correct = 0, total = 0
  const faults = d.raw.faults
  for (const [node, series] of Object.entries(d.raw.beliefs)) {
    const mine = faults.filter(f => f.target === node)
    if (mine.length === 0) continue
    for (const b of series) {
      const f = mine.find(x => b.t >= x.t_start && b.t <= x.t_end)
      if (!f) continue
      total++
      if (b.x === f.kind) correct++
    }
  }
  return { correct, total, pct: total ? correct / total : 0 }
}
