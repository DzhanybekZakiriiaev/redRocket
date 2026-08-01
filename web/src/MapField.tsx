/* ────────────────────────────────────────────────────────────────────────
   MapField.tsx — [F]. Everything that moves every frame.

   Two stacked canvases: WebGL for the textured sphere, 2D for all vector
   chrome, markers, arcs and pulses. React mounts them once and then never
   touches them again — the rAF loop draws imperatively.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { Globe, coastlines, satRadius, type LonLat, type GlobeMode } from './globe'
import { flickerOn, onFrame } from './clock'
import { useStore } from './store'
import { CAUSE_WORD, type Derived, type DState } from './trace'
import { roleOf, ROLE_TAG } from './sites'

/* ── link declutter ──────────────────────────────────────────────────────
   Eight satellites with a near-complete ISL mesh puts 14-15 arcs on screen at
   once, which reads as a ball of yarn and buries the links that matter. The
   simulator still runs every one of them — this only governs what is DRAWN.

   Kept: all downlinks (a surface contact is where faults surface), plus any
   ISL touching the selected asset, plus any ISL that is currently failing.
   Dropped: background satellite-to-satellite chatter with nothing riding on
   it. Set SHOW_ALL_ISL true to put the full mesh back. */
const SHOW_ALL_ISL = false

const FONT = '"Roboto Condensed Variable", "Roboto Condensed", sans-serif'
const f8  = `400 8px ${FONT}`
const f15 = `700 15px ${FONT}`

interface Pal {
  ink: string; dim: string; faint: string; accent: string; alert: string
  warn: string; route: string; neutral: string; bg: string
  dimA: number; faintA: number
}

function readPalette(): Pal {
  const cs = getComputedStyle(document.documentElement)
  const g = (n: string) => cs.getPropertyValue(n).trim()
  return {
    ink: g('--ink'), dim: g('--ink-dim'), faint: g('--ink-faint'),
    accent: g('--accent'), alert: g('--alert'),
    warn: g('--warn') || '#F5A623', route: g('--route') || '#9B7BFF',
    neutral: g('--neutral'),
    bg: g('--bg'),
    dimA: parseFloat(g('--a-dim')) || 0.45,
    faintA: parseFloat(g('--a-faint')) || 0.18,
  }
}

/** Colour for an epistemic state. This is the only place colour is decided. */
function stateColour(st: DState, pal: Pal, wall: number, phase: number): string {
  switch (st) {
    case 'NOMINAL':   return pal.dim
    case 'RESOLVED':  return pal.accent
    case 'FAULT':     return pal.alert
    case 'STALE':     return pal.neutral
    // amber, not red — "still deciding between two causes", distinct from a
    // confirmed fault. Flickers so an unsettled node reads as unsettled.
    case 'UNCERTAIN': return flickerOn(wall, phase) ? pal.warn : pal.ink
  }
}
export { stateColour }

const idx2 = (id: string) => id.slice(-2)
const hash = (s: string) => {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) }
  return ((h >>> 0) % 1000) / 1000
}

/* Deterministic animation windows, measured in grid frames so scrubbing is
   exact in both directions. */
const ARC_IN = 10
const ARC_OUT = 7
const FAIL_LIFE = 3

interface P { x: number; y: number; behind: boolean }

export default function MapField() {
  const glRef = useRef<HTMLCanvasElement>(null)
  const ovRef = useRef<HTMLCanvasElement>(null)
  const hitRef = useRef<{ id: string; x: number; y: number }[]>([])

  useEffect(() => {
    const glc = glRef.current!, ovc = ovRef.current!
    const ctx = ovc.getContext('2d')!
    const globe = new Globe(glc, { mode: useStore.getState().globeMode })
    let lastMode: GlobeMode = useStore.getState().globeMode
    const coasts = coastlines()

    /* pre-tessellate the graticule: meridians every 15 deg, parallels every 15 */
    const grat: LonLat[][] = []
    for (let lon = -180; lon < 180; lon += 15) {
      const l: LonLat[] = []
      for (let lat = -90; lat <= 90; lat += 3) l.push([lon, lat])
      grat.push(l)
    }
    for (let lat = -75; lat <= 75; lat += 15) {
      const l: LonLat[] = []
      for (let lon = -180; lon <= 180; lon += 3) l.push([lon, lat])
      grat.push(l)
    }

    let pal = readPalette()
    let W = 0, H = 0, DPR = 1
    let dotPat: CanvasPattern | null = null

    const makeDots = () => {
      const c = document.createElement('canvas')
      c.width = 14; c.height = 14
      const g = c.getContext('2d')!
      g.fillStyle = 'rgba(232,234,237,0.075)'
      g.fillRect(0, 0, 1, 1)
      dotPat = ctx.createPattern(c, 'repeat')
    }

    const resize = () => {
      W = window.innerWidth; H = window.innerHeight
      DPR = Math.min(2, window.devicePixelRatio || 1)
      ovc.width = Math.round(W * DPR); ovc.height = Math.round(H * DPR)
      ovc.style.width = W + 'px'; ovc.style.height = H + 'px'
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0)
      globe.resize(W, H, DPR)
      pal = readPalette()
      makeDots()
    }
    resize()
    window.addEventListener('resize', resize)
    const unsubPal = useStore.subscribe((st) => {
      pal = readPalette()
      if (st.globeMode !== lastMode) { lastMode = st.globeMode; globe.setMode(st.globeMode) }
    })

    const tmp = new THREE.Vector3()
    const proj =(lat: number, lon: number, r: number, o: P): boolean => {
      const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180
      const c = Math.cos(la)
      tmp.set(r * c * Math.cos(lo), r * Math.sin(la), -r * c * Math.sin(lo))
      return globe.project(tmp, o)
    }

    /* ── polyline drawing with limb clipping ─────────────────────────── */
    const strokeGeo = (lines: LonLat[][], colour: string, alpha: number,
                       side: 'front' | 'back' = 'front') => {
      ctx.save()
      ctx.strokeStyle = colour
      ctx.globalAlpha = alpha
      ctx.lineWidth = 1
      ctx.beginPath()
      const q: P = { x: 0, y: 0, behind: false }
      for (const line of lines) {
        let open = false
        for (const [lon, lat] of line) {
          const ok = proj(lat, lon, 1.001, q)
          const vis = ok && (side === 'front' ? !q.behind : q.behind)
          if (!vis) { open = false; continue }
          if (!open) { ctx.moveTo(q.x, q.y); open = true } else ctx.lineTo(q.x, q.y)
        }
      }
      ctx.stroke()
      ctx.restore()
    }

    /* ── glyphs ─────────────────────────────────────────────────────── */
    const triangle = (x: number, y: number, s: number, colour: string) => {
      ctx.beginPath()
      ctx.moveTo(x, y - s * 0.62)
      ctx.lineTo(x + s * 0.55, y + s * 0.4)
      ctx.lineTo(x - s * 0.55, y + s * 0.4)
      ctx.closePath()
      ctx.fillStyle = colour
      ctx.fill()
    }
    const diamond = (x: number, y: number, s: number, colour: string) => {
      ctx.beginPath()
      ctx.moveTo(x, y - s / 2); ctx.lineTo(x + s / 2, y)
      ctx.lineTo(x, y + s / 2); ctx.lineTo(x - s / 2, y)
      ctx.closePath()
      ctx.fillStyle = colour
      ctx.fill()
    }
    const label =(t: string, x: number, y: number, font: string, colour: string, align: CanvasTextAlign = 'left', track = '0.12em') => {
      ctx.font = font
      ;(ctx as unknown as { letterSpacing: string }).letterSpacing = track
      ctx.fillStyle = colour
      ctx.textAlign = align
      ctx.textBaseline = 'alphabetic'
      ctx.fillText(t, x, y)
      ;(ctx as unknown as { letterSpacing: string }).letterSpacing = '0em'
    }

    /* quadratic arc between two screen points, bowed away from frame centre */
    const arcPath = (ax: number, ay: number, bx: number, by: number) => {
      const mx = (ax + bx) / 2, my = (ay + by) / 2
      const dx = bx - ax, dy = by - ay
      const len = Math.hypot(dx, dy) || 1
      const nx = -dy / len, ny = dx / len
      const k = Math.min(90, len * 0.16)
      return { cx: mx + nx * k, cy: my + ny * k }
    }
    const qPoint = (ax: number, ay: number, cx: number, cy: number, bx: number, by: number, t: number) => {
      const u = 1 - t
      return {
        x: u * u * ax + 2 * u * t * cx + t * t * bx,
        y: u * u * ay + 2 * u * t * cy + t * t * by,
        tx: 2 * u * (cx - ax) + 2 * t * (bx - cx),
        ty: 2 * u * (cy - ay) + 2 * t * (by - cy),
      }
    }

    /* ── per-node screen position resolution ──────────────────────────── */
    const posOf = (tr: Derived, id: string, f: number, o: P): boolean => {
      if (id.startsWith('SAT')) {
        const [lat, lon, alt] = tr.posAt(id, f)
        return proj(lat, lon, satRadius(alt), o)
      }
      const g = tr.gsById[id]
      if (!g) return false
      return proj(g.lat, g.lon, 1.004, o)
    }

    /* ── orbital ring for one satellite ────────────────────────────────────
       The orbit plane is estimated from the current sub-satellite point and one
       a short arc away; a great circle in that plane, at the sat's altitude, is
       the orbit. Near hemisphere bright + a soft accent glow; far hemisphere
       faint, so the ring reads as passing behind the planet. */
    const oVec = new THREE.Vector3()
    const oPt: P = { x: 0, y: 0, behind: false }
    const vA = new THREE.Vector3(), vB = new THREE.Vector3()
    const nrm = new THREE.Vector3(), uu = new THREE.Vector3(), ww = new THREE.Vector3()
    const toLocal = (lat: number, lon: number, out: THREE.Vector3) => {
      const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180
      const c = Math.cos(la)
      out.set(c * Math.cos(lo), Math.sin(la), -c * Math.sin(lo))
    }
    const drawOrbit = (tr: Derived, s: string, f: number) => {
      const [la0, lo0, alt] = tr.posAt(s, f)
      const fB = f + 18 <= tr.nFrames - 1 ? f + 18 : f - 18
      const [la1, lo1] = tr.posAt(s, fB)
      toLocal(la0, lo0, vA)
      toLocal(la1, lo1, vB)
      nrm.crossVectors(vA, vB)
      if (nrm.length() < 0.15) return          // arc too short → unstable plane
      nrm.normalize()
      uu.copy(vA).normalize()
      ww.crossVectors(nrm, uu).normalize()
      const rr = satRadius(alt)
      const N = 96
      for (const side of ['back', 'front'] as const) {
        ctx.save()
        // orbits are STRUCTURE — dim white, not accent, so teal stays meaningful
        ctx.strokeStyle = pal.ink
        ctx.globalAlpha = side === 'front' ? 0.20 : 0.06
        ctx.lineWidth = 1
        ctx.beginPath()
        let open = false
        for (let i = 0; i <= N; i++) {
          const th = (i / N) * Math.PI * 2
          const c = Math.cos(th), sn = Math.sin(th)
          oVec.set(
            (c * uu.x + sn * ww.x) * rr,
            (c * uu.y + sn * ww.y) * rr,
            (c * uu.z + sn * ww.z) * rr,
          )
          const ok = globe.project(oVec, oPt)
          const vis = ok && (side === 'front' ? !oPt.behind : oPt.behind)
          if (!vis) { open = false; continue }
          if (!open) { ctx.moveTo(oPt.x, oPt.y); open = true } else ctx.lineTo(oPt.x, oPt.y)
        }
        ctx.stroke()
        ctx.restore()
      }
    }

    /* ── the frame ────────────────────────────────────────────────────── */
    const draw = (frame: number, wall: number, dt: number) => {
      globe.render(dt)
      const tr = useStore.getState().trace
      ctx.clearRect(0, 0, W, H)
      if (!tr) return

      const f = Math.max(0, Math.min(tr.nFrames - 1, frame))
      const fi = Math.round(f)
      const sel = useStore.getState().selected
      const cx = W / 2, cy = H / 2

      /* 1 ── graticule + coastlines. In wireframe mode the graticule IS the
         surface (bright, with a faint see-through far hemisphere); over a
         textured planet it stays faint so it reads as an overlaid grid. */
      if (lastMode === 'wireframe') {
        // pure graticule sphere — no coastlines (no "land specks")
        strokeGeo(grat, pal.ink, 0.10, 'back')
        strokeGeo(grat, pal.ink, 0.32, 'front')
      } else {
        const gcol = lastMode === 'mars' ? 'rgba(255,228,208,1)' : pal.ink
        strokeGeo(grat, gcol, 0.12, 'front')
        if (lastMode === 'earth') strokeGeo(coasts, pal.ink, 0.22, 'front')
      }

      /* 1b ── orbital rings, accented */
      for (const s of tr.sats) drawOrbit(tr, s, f)

      /* 2 ── dotted overlay grid, above the globe */
      if (dotPat) {
        ctx.save()
        ctx.fillStyle = dotPat
        ctx.fillRect(0, 0, W, H)
        ctx.restore()
      }

      /* 3 ── faint crosshair through the globe centre. The reticle rings and
         cardinal chevrons were removed — over a contained globe they read as
         clutter sitting in front of the planet. */
      ctx.save()
      ctx.strokeStyle = pal.faint
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.55
      ctx.beginPath()
      ctx.moveTo(0, Math.round(cy) + 0.5); ctx.lineTo(W, Math.round(cy) + 0.5)
      ctx.moveTo(Math.round(cx) + 0.5, 0); ctx.lineTo(Math.round(cx) + 0.5, H)
      ctx.stroke()
      ctx.restore()

      /* ── 5 ── nodes: resolve screen positions once ─────────────────── */
      const pos: Record<string, P> = {}
      for (const s of tr.sats) {
        const o: P = { x: 0, y: 0, behind: false }
        if (posOf(tr, s, f, o)) pos[s] = o
      }
      for (const g of tr.gs) {
        const o: P = { x: 0, y: 0, behind: false }
        if (posOf(tr, g.id, f, o)) pos[g.id] = o
      }

      /* ── 6 ── ground truth faults, as filled diamonds ───────────────── */
      const activeFaults = tr.faultsByFrame[fi]
      const faultTargets = new Set<string>()
      for (const i of activeFaults) faultTargets.add(tr.raw.faults[i].target)

      /* ── 7 ── surface sites, always drawn.
         Previously a station appeared only while faulted, which meant the
         network had no visible ground segment at all until something broke —
         the globe read as eight satellites orbiting nothing. Every site now
         holds a persistent marker, shaped by role (see sites.ts: a display
         convention, NOT simulator state). Operational sites carry a label;
         the seven non-operational ones stay as bare dim ticks so they add
         density without competing for attention. A fault still promotes its
         site to a red triangle inside a ring. */
      for (const g of tr.gs) {
        const q = pos[g.id]
        if (!q || q.behind) continue
        const faulted = faultTargets.has(g.id)
        const role = roleOf(g.id)

        if (!g.operational) {
          ctx.save()
          ctx.fillStyle = pal.ink
          ctx.globalAlpha = 0.22
          ctx.fillRect(q.x - 1.5, q.y - 1.5, 3, 3)
          ctx.restore()
          continue
        }

        const col = faulted ? pal.alert : pal.accent
        ctx.save()
        ctx.globalAlpha = faulted ? 1 : 0.85
        if (role === 'ROVER') {
          // rover — a small diamond, visually distinct from the dish triangles
          ctx.fillStyle = col
          ctx.beginPath()
          ctx.moveTo(q.x, q.y - 5); ctx.lineTo(q.x + 5, q.y)
          ctx.lineTo(q.x, q.y + 5); ctx.lineTo(q.x - 5, q.y)
          ctx.closePath(); ctx.fill()
        } else {
          triangle(q.x, q.y, faulted ? 11 : 8, col)
          if (role === 'DSN') {
            // deep-space complex — the high-gain dish gets a collar arc
            ctx.strokeStyle = col
            ctx.globalAlpha = 0.55
            ctx.lineWidth = 1
            ctx.beginPath()
            ctx.arc(q.x, q.y, 9, Math.PI * 1.15, Math.PI * 1.85)
            ctx.stroke()
          }
        }
        ctx.restore()

        if (faulted) {
          ctx.save(); ctx.strokeStyle = pal.alert; ctx.globalAlpha = 0.7; ctx.lineWidth = 1
          ctx.beginPath(); ctx.arc(q.x, q.y, 16, 0, Math.PI * 2); ctx.stroke(); ctx.restore()
        }

        ctx.save()
        ctx.font = f8
        ctx.textAlign = 'center'
        ctx.textBaseline = 'top'
        ctx.fillStyle = faulted ? pal.alert : pal.dim
        ctx.globalAlpha = faulted ? 1 : 0.8
        ctx.fillText(g.name, q.x, q.y + 9)
        ctx.globalAlpha = 0.45
        ctx.fillText(ROLE_TAG[role], q.x, q.y + 18)
        ctx.restore()
      }

      /* ── 8 ── failed contacts: red dashed. A broken link is an error, and
         every error on this globe is red. Dashed so the absence still reads. */
      ctx.save()
      ctx.setLineDash([3, 4])
      ctx.lineWidth = 1
      ctx.globalAlpha = 0.6
      ctx.strokeStyle = pal.alert
      for (let k = Math.max(0, fi - FAIL_LIFE); k <= fi; k++) {
        for (const i of tr.failsByFrame[k]) {
          const x = tr.raw.fails[i]
          const A = pos[x.a], B = pos[x.b]
          if (!A || !B || A.behind || B.behind) continue
          const c = arcPath(A.x, A.y, B.x, B.y)
          ctx.beginPath()
          ctx.moveTo(A.x, A.y)
          ctx.quadraticCurveTo(c.cx, c.cy, B.x, B.y)
          ctx.stroke()
        }
      }
      ctx.setLineDash([])
      ctx.restore()

      /* ── 9 ── contact arcs: the network breathing ───────────────────── */
      const arcs: { a: string; b: string; ax: number; ay: number; bx: number; by: number; cx: number; cy: number; dn: boolean }[] = []
      /* pairs currently in failure — these stay drawn however dense the mesh */
      const failing = new Set<string>()
      for (let k = Math.max(0, fi - FAIL_LIFE); k <= fi; k++)
        for (const i of tr.failsByFrame[k]) {
          const x = tr.raw.fails[i]
          failing.add(x.a < x.b ? `${x.a}|${x.b}` : `${x.b}|${x.a}`)
        }

      ctx.save()
      ctx.lineWidth = 1
      for (const i of tr.contactsByFrame[fi]) {
        const c = tr.raw.contacts[i]

        if (!SHOW_ALL_ISL && c.k === 'isl'
          && c.a !== sel && c.b !== sel
          && !failing.has(c.a < c.b ? `${c.a}|${c.b}` : `${c.b}|${c.a}`)) continue

        const A = pos[c.a], B = pos[c.b]
        if (!A || !B || A.behind || B.behind) continue
        if (Math.hypot(B.x - A.x, B.y - A.y) > Math.max(W, H) * 1.1) continue

        const s = Math.round((c.t0 - tr.meta.epoch_ms) / tr.meta.grid_ms)
        const e = Math.round((c.t1 - tr.meta.epoch_ms) / tr.meta.grid_ms)
        const grow = Math.max(0, Math.min(1, (f - s) / ARC_IN))
        const fade = Math.max(0, Math.min(1, (e - f) / ARC_OUT))
        if (grow <= 0) continue

        const cp = arcPath(A.x, A.y, B.x, B.y)
        const involved = c.a === sel || c.b === sel
        ctx.globalAlpha = (involved ? 0.9 : 0.42) * fade
        ctx.strokeStyle = pal.accent

        ctx.beginPath()
        if (grow >= 1) {
          ctx.moveTo(A.x, A.y)
          ctx.quadraticCurveTo(cp.cx, cp.cy, B.x, B.y)
        } else {
          // partial draw-in
          const N = 18
          for (let n = 0; n <= N * grow; n++) {
            const q = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, n / N)
            if (n === 0) ctx.moveTo(q.x, q.y); else ctx.lineTo(q.x, q.y)
          }
        }
        ctx.stroke()
        arcs.push({ a: c.a, b: c.b, ax: A.x, ay: A.y, bx: B.x, by: B.y, cx: cp.cx, cy: cp.cy, dn: c.k === 'downlink' })

        /* bundle traffic — filled dots running the arc */
        if (grow >= 1 && fade >= 1) {
          const seed = hash(c.a + c.b)
          for (let d = 0; d < 2; d++) {
            const t = (f * 0.045 + seed + d * 0.5) % 1
            const q = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, t)
            ctx.globalAlpha = involved ? 0.95 : 0.5
            ctx.fillStyle = pal.accent
            ctx.beginPath(); ctx.arc(q.x, q.y, 2.1, 0, Math.PI * 2); ctx.fill()
          }
        }

        /* downlink arcs terminate in a numbered box */
        if (c.k === 'downlink' && grow >= 1) {
          const q = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, 0.5)
          ctx.globalAlpha = 0.8 * fade
          ctx.strokeStyle = pal.dim
          ctx.strokeRect(Math.round(q.x - 5) + 0.5, Math.round(q.y - 4) + 0.5, 10, 8)
          ctx.globalAlpha = 1
          label(idx2(c.a.startsWith('SAT') ? c.a : c.b), q.x, q.y + 3, `400 7px ${FONT}`, pal.dim, 'center', '0em')
        }
      }
      ctx.restore()

      /* ── 10 ── gossip glyphs removed — evidence exchange is narrated in the
         EVENT STREAM panel instead of arrows on the globe. */

      /* ── 10b ── ACTIVE REROUTES. When a node diagnoses a fault it steers
         traffic to `target`; that decision was invisible on the globe before
         (it lived only in the inspector). Draw the new path bright violet with
         a running pulse + arrowhead so a reroute is obvious from across a room.
         Violet is used for nothing else, so any violet on screen == a reroute. */
      const shortId = (id: string) =>
        id.startsWith('SAT') ? idx2(id) : (tr.gsById[id]?.name ?? id.replace('GS_', ''))
      for (const s of tr.sats) {
        const sm = tr.samples[s][fi]
        if (sm.action !== 'reroute' || !sm.target) continue
        const A = pos[s], B = pos[sm.target]
        if (!A || !B || A.behind || B.behind) continue
        if (Math.hypot(B.x - A.x, B.y - A.y) > Math.max(W, H) * 1.1) continue
        const cp = arcPath(A.x, A.y, B.x, B.y)

        ctx.save()
        // the new route — thick, glowing violet
        ctx.strokeStyle = pal.route
        ctx.shadowColor = pal.route
        ctx.shadowBlur = 8
        ctx.globalAlpha = 0.92
        ctx.lineWidth = 1.6
        ctx.beginPath()
        ctx.moveTo(A.x, A.y)
        ctx.quadraticCurveTo(cp.cx, cp.cy, B.x, B.y)
        ctx.stroke()

        // traffic now flowing the new way
        ctx.fillStyle = pal.route
        const tt = (f * 0.06 + hash(s + sm.target)) % 1
        const q = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, tt)
        ctx.globalAlpha = 1
        ctx.beginPath(); ctx.arc(q.x, q.y, 2.8, 0, Math.PI * 2); ctx.fill()

        // arrowhead into the target — direction from the curve tangent near B
        const h = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, 0.9)
        const ang = Math.atan2(B.y - h.y, B.x - h.x)
        ctx.translate(B.x, B.y); ctx.rotate(ang)
        ctx.beginPath(); ctx.moveTo(2, 0); ctx.lineTo(-8, -4); ctx.lineTo(-8, 4)
        ctx.closePath(); ctx.fill()
        ctx.restore()

        // label at the arc apex
        const mid = qPoint(A.x, A.y, cp.cx, cp.cy, B.x, B.y, 0.5)
        label(`↻ ${shortId(s)} → ${shortId(sm.target)}`,
          mid.x, mid.y - 7, f8, pal.route, 'center', '0.14em')
      }

      /* ── 11 ── ground truth fault diamonds ──────────────────────────── */
      for (const i of activeFaults) {
        const ft = tr.raw.faults[i]
        const q = pos[ft.target]
        if (!q || q.behind) continue
        diamond(q.x, q.y - 22, 13, pal.alert)
      }

      /* ── 12 ── asset chips. THE FLICKER LIVES HERE ──────────────────── */
      const hits: { id: string; x: number; y: number }[] = []
      tr.sats.forEach((s, si) => {
        const q = pos[s]
        if (!q) return
        hits.push({ id: s, x: q.x, y: q.y })
        if (q.behind) return

        const sm = tr.samples[s][fi]
        const col = stateColour(sm.state, pal, wall, si)
        const isSel = s === sel

        // exact marker point + leader to the chip
        ctx.save()
        ctx.strokeStyle = pal.ink
        ctx.globalAlpha = 0.8
        ctx.lineWidth = 1
        ctx.beginPath()
        ctx.moveTo(q.x - 4, q.y); ctx.lineTo(q.x + 4, q.y)
        ctx.moveTo(q.x, q.y - 4); ctx.lineTo(q.x, q.y + 4)
        ctx.stroke()

        const chx = Math.round(q.x + 26), chy = Math.round(q.y - 34)
        ctx.globalAlpha = 0.45
        ctx.beginPath()
        ctx.moveTo(q.x + 5, q.y - 3); ctx.lineTo(chx - 2, chy + 22)
        ctx.stroke()
        ctx.restore()

        // satellite body colour follows its epistemic state, so state reads
        // from the globe itself — red=fault, amber=uncertain, teal=resolved.
        const faulted = sm.state === 'FAULT'
        const uncertain = sm.state === 'UNCERTAIN'
        const resolved = sm.state === 'RESOLVED'
        let bodyCol = pal.ink
        if (faulted) bodyCol = pal.alert
        else if (uncertain) bodyCol = flickerOn(wall, si) ? pal.warn : pal.ink
        else if (resolved || isSel) bodyCol = pal.accent
        const big = faulted || uncertain || isSel
        ctx.save()
        ctx.fillStyle = bodyCol
        ctx.shadowColor = bodyCol
        ctx.shadowBlur = faulted ? 10 : big ? 8 : 3
        ctx.beginPath(); ctx.arc(q.x, q.y, big ? 3.4 : 2.2, 0, Math.PI * 2); ctx.fill()
        ctx.restore()
        // attention ring — red for fault, amber for uncertain. Resolved and
        // nominal stay ringless so the two "needs-attention" states pop.
        const ringCol = faulted ? pal.alert : uncertain ? pal.warn : null
        if (ringCol) {
          ctx.save(); ctx.strokeStyle = ringCol; ctx.globalAlpha = 0.75; ctx.lineWidth = 1
          ctx.beginPath(); ctx.arc(q.x, q.y, 11, 0, Math.PI * 2); ctx.stroke(); ctx.restore()
        }

        // range rings on a sparse subset only
        if (si === 1 || si === 5) {
          ctx.save()
          ctx.strokeStyle = pal.faint
          ctx.lineWidth = 1
          ctx.beginPath(); ctx.arc(q.x, q.y, 74, 0, Math.PI * 2); ctx.stroke()
          ctx.restore()
        }

        // status word above the chip — only when something is wrong (no NOMINAL)
        if (sm.state !== 'NOMINAL') {
          const wcol = sm.state === 'FAULT' ? pal.alert
            : sm.state === 'UNCERTAIN' ? pal.warn : pal.ink
          label(CAUSE_WORD[sm.cause], chx, chy - 6, f8, wcol, 'left', '0.18em')
        }

        // the chip itself
        ctx.save()
        ctx.fillStyle = pal.ink
        roundRect(ctx, chx, chy, 38, 22, 2)
        ctx.fill()
        ctx.restore()
        label(idx2(s), chx + 19, chy + 16, f15, '#0A0B0D', 'center', '0em')

        // state circle + dot — the single most important pixel on screen
        const dcx = chx + 38 + 12, dcy = chy + 11
        ctx.save()
        ctx.strokeStyle = pal.dim
        ctx.lineWidth = 1
        ctx.beginPath(); ctx.arc(dcx, dcy, 8, 0, Math.PI * 2); ctx.stroke()
        ctx.fillStyle = col
        ctx.beginPath(); ctx.arc(dcx, dcy, 3, 0, Math.PI * 2); ctx.fill()
        ctx.restore()

        if (isSel) {
          ctx.save()
          ctx.strokeStyle = pal.accent
          ctx.lineWidth = 1
          ctx.strokeRect(Math.round(chx) - 6.5, Math.round(chy) - 5.5, 38 + 32, 22 + 11)
          ctx.restore()
        }
      })
      hitRef.current = hits

      /* ── 13 ── corner scale + cursor telemetry on the map itself ────── */
      ctx.globalAlpha = 1
    }

    /* ── mouse: drag to spin, wheel to zoom, click (no drag) to select ──── */
    let dragging = false, lx = 0, ly = 0, moved = 0
    const onDown = (e: MouseEvent) => {
      dragging = true; lx = e.clientX; ly = e.clientY; moved = 0
      ovc.style.cursor = 'grabbing'
    }
    const onMove = (e: MouseEvent) => {
      if (!dragging) return
      const dx = e.clientX - lx, dy = e.clientY - ly
      lx = e.clientX; ly = e.clientY
      moved += Math.abs(dx) + Math.abs(dy)
      globe.rotateBy(dx, dy)
    }
    const onUp = (e: MouseEvent) => {
      if (dragging && moved < 5) {            // a click, not a drag → select
        const hits = hitRef.current
        let best: string | null = null, bd = 60
        for (const h of hits) {
          const d = Math.hypot(h.x - e.clientX, h.y - e.clientY)
          if (d < bd) { bd = d; best = h.id }
        }
        if (best) useStore.getState().select(best)
      }
      dragging = false
      ovc.style.cursor = 'grab'
    }
    const onWheel = (e: WheelEvent) => { e.preventDefault(); globe.zoomBy(e.deltaY) }
    ovc.style.cursor = 'grab'
    ovc.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    ovc.addEventListener('wheel', onWheel, { passive: false })

    const un = onFrame(draw)
    return () => {
      un(); unsubPal()
      window.removeEventListener('resize', resize)
      ovc.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      ovc.removeEventListener('wheel', onWheel)
      globe.renderer.dispose()
    }
  }, [])

  return (
    <>
      <canvas ref={glRef} className="fixed inset-0 w-full h-full" style={{ zIndex: 0 }} />
      <canvas ref={ovRef} className="fixed inset-0" style={{ zIndex: 1 }} />
    </>
  )
}

function roundRect(g: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  g.beginPath()
  g.moveTo(x + r, y)
  g.arcTo(x + w, y, x + w, y + h, r)
  g.arcTo(x + w, y + h, x, y + h, r)
  g.arcTo(x, y + h, x, y, r)
  g.arcTo(x, y, x + w, y, r)
  g.closePath()
}
