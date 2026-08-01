/* ────────────────────────────────────────────────────────────────────────
   clock.ts — the transport.

   The rAF loop writes `frame` HERE, into a plain module-level mutable, and
   notifies subscribers that poke the DOM directly. It never touches React
   state; if it did, the whole tree would re-render 60x a second and stall.
   ──────────────────────────────────────────────────────────────────────── */

/** Grid frames advanced per wall-clock second at 1x. 1441 frames / 6 = ~4 min
 *  for the full 24 h day, so 8x replays the day in 30 s. */
export const BASE_FPS = 6

export const clock = {
  frame: 0,
  nFrames: 1441,
  playing: true,
  speed: 1 as 0.25 | 0.5 | 1 | 2 | 8,
  /** wall-clock ms since load — drives the flicker, which must stay at a hard
   *  400 ms regardless of playback speed */
  wall: 0,
  /** true once the trace has been derived */
  ready: false,
}

export type FrameFn = (frame: number, wall: number, dt: number) => void

const subs = new Set<FrameFn>()
export function onFrame(fn: FrameFn): () => void {
  subs.add(fn)
  return () => { subs.delete(fn) }
}

let raf = 0
let last = 0

function tick(now: number) {
  raf = requestAnimationFrame(tick)
  const dt = last ? Math.min(100, now - last) : 16
  last = now
  clock.wall = now

  if (clock.playing && clock.ready) {
    clock.frame += (dt / 1000) * BASE_FPS * clock.speed
    if (clock.frame >= clock.nFrames - 1) clock.frame = 0   // loop the day
  }
  for (const fn of subs) fn(clock.frame, now, dt)
}

export function startClock() {
  if (!raf) raf = requestAnimationFrame(tick)
}

export function seek(f: number) {
  clock.frame = Math.max(0, Math.min(clock.nFrames - 1, f))
}

/* ── the flicker ────────────────────────────────────────────────────────
   Hard 400 ms alternation, no crossfade, no easing. Phase is offset per
   node so eight uncertain assets read as eight independent posteriors
   rather than one synchronised UI blink.                                  */
export const FLICKER_MS = 400
export function flickerOn(wall: number, phase = 0): boolean {
  return Math.floor((wall + phase * 97) / FLICKER_MS) % 2 === 0
}
