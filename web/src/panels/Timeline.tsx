/* [P] Timeline + diagnosability band.
   The band is painted once per trace. The playhead and the readout are
   written by the rAF loop straight into the DOM. */

import { useEffect, useRef } from 'react'
import { useStore } from '../store'
import { clock, onFrame, seek } from '../clock'
import { tPlus } from '../trace'

export default function Timeline() {
  const trace = useStore(s => s.trace)
  const playing = useStore(s => s.playing)
  const speed = useStore(s => s.speed)
  const setPlaying = useStore(s => s.setPlaying)
  const setSpeed = useStore(s => s.setSpeed)

  const wrap = useRef<HTMLDivElement>(null)
  const band = useRef<HTMLCanvasElement>(null)
  const head = useRef<HTMLDivElement>(null)
  const clockEl = useRef<HTMLDivElement>(null)
  const diagEl = useRef<HTMLSpanElement>(null)
  const range = useRef<HTMLInputElement>(null)
  const dragging = useRef(false)

  /* paint the diagnosability band */
  useEffect(() => {
    if (!trace) return
    const paint = () => {
      const c = band.current, w = wrap.current
      if (!c || !w) return
      const W = w.clientWidth, H = 8
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      c.width = Math.round(W * dpr); c.height = Math.round(H * dpr)
      c.style.width = W + 'px'; c.style.height = H + 'px'
      const g = c.getContext('2d')!
      g.setTransform(dpr, 0, 0, dpr, 0, 0)
      g.clearRect(0, 0, W, H)

      const cs = getComputedStyle(document.documentElement)
      const dim = cs.getPropertyValue('--ink-dim').trim()

      // hatch tile, matching warning strip [J]
      const t = document.createElement('canvas')
      t.width = 4; t.height = 4
      const tg = t.getContext('2d')!
      tg.strokeStyle = 'rgba(232,234,237,0.32)'
      tg.lineWidth = 1
      tg.beginPath()
      tg.moveTo(-1, 5); tg.lineTo(5, -1)
      tg.moveTo(3, 5); tg.lineTo(5, 3)
      tg.stroke()
      const pat = g.createPattern(t, 'repeat')!

      const n = trace.nFrames
      let i = 0
      while (i < n) {
        const v = trace.diagnosable[i]
        let j = i
        while (j < n && trace.diagnosable[j] === v) j++
        const x0 = (i / n) * W, x1 = (j / n) * W
        g.fillStyle = v ? dim : pat
        g.fillRect(x0, 0, Math.max(1, x1 - x0), H)
        i = j
      }
      g.strokeStyle = 'rgba(232,234,237,0.18)'
      g.lineWidth = 1
      g.strokeRect(0.5, 0.5, W - 1, H - 1)
    }
    paint()
    window.addEventListener('resize', paint)
    const unsub = useStore.subscribe(paint)
    return () => { window.removeEventListener('resize', paint); unsub() }
  }, [trace])

  /* playhead + clock, driven imperatively */
  useEffect(() => {
    if (!trace) return
    const un = onFrame((frame) => {
      const w = wrap.current
      if (!w) return
      const pct = frame / (trace.nFrames - 1)
      if (head.current) head.current.style.transform = `translateX(${pct * w.clientWidth}px)`
      if (clockEl.current) clockEl.current.textContent = 'T+' + tPlus(trace.timeAt(frame), trace.meta.epoch_ms)
      if (diagEl.current) {
        const fi = Math.round(frame)
        const ok = trace.diagnosable[fi] === 1
        diagEl.current.textContent = ok ? 'IDENTIFIABLE' : 'NOT IDENTIFIABLE'
        diagEl.current.className = ok ? 't-micro dim' : 't-micro alr'
      }
      if (range.current && !dragging.current) range.current.value = String(Math.round(frame))
    })
    return () => un()
  }, [trace])

  if (!trace) return null

  const ticks = 13

  return (
    <div className="fixed" style={{ left: 372, right: 296, bottom: 52, zIndex: 20 }}>
      <div className="flex justify-between items-end mb-[3px]">
        <div className="t-micro dim">DIAGNOSABILITY / OBSERVABLE WINDOW</div>
        <span ref={diagEl} className="t-micro dim">IDENTIFIABLE</span>
      </div>

      <div ref={wrap} className="relative">
        <canvas ref={band} style={{ display: 'block', width: '100%', height: 8 }} />

        {/* 1px rule with descending ticks — matches scale bar [M] exactly */}
        <div className="relative" style={{ marginTop: 7 }}>
          <div style={{ height: 1, background: 'var(--ink-dim)' }} />
          <div className="relative" style={{ height: 8 }}>
            {Array.from({ length: ticks }).map((_, i) => (
              <div
                key={i}
                style={{
                  position: 'absolute', left: `${(i / (ticks - 1)) * 100}%`,
                  top: 0, width: 1, height: i % 2 === 0 ? 7 : 4,
                  background: 'var(--ink-faint)',
                }}
              />
            ))}
          </div>
          <div className="relative" style={{ height: 9 }}>
            {Array.from({ length: ticks }).map((_, i) =>
              i % 2 === 0 ? (
                <div
                  key={i}
                  className="absolute t-micro faint"
                  style={{
                    left: `${(i / (ticks - 1)) * 100}%`, top: -2,
                    transform: i === 0 ? 'none' : i === ticks - 1 ? 'translateX(-100%)' : 'translateX(-50%)',
                    fontSize: 7,
                  }}
                >
                  {String(i * 2).padStart(2, '0')}
                </div>
              ) : null,
            )}
          </div>
        </div>

        {/* playhead — the only --accent element here */}
        <div
          ref={head}
          style={{
            position: 'absolute', left: 0, top: -3, width: 1, height: 34,
            background: 'var(--accent)', pointerEvents: 'none',
          }}
        />

        {/* scrubber sits transparently over the band */}
        <input
          ref={range}
          type="range"
          min={0}
          max={trace.nFrames - 1}
          defaultValue={0}
          onPointerDown={() => { dragging.current = true }}
          onPointerUp={() => { dragging.current = false }}
          onChange={e => seek(Number(e.target.value))}
          style={{ position: 'absolute', left: 0, top: -3, width: '100%', opacity: 0.001 }}
        />
      </div>

      <div className="flex flex-wrap items-center gap-x-[8px] gap-y-[4px] mt-[6px]">
        <button
          className={playing ? 'on' : ''}
          onClick={() => { const p = !playing; setPlaying(p); clock.playing = p }}
          style={{ minWidth: 44 }}
        >
          {playing ? '▮▮ PAUSE' : '▶ PLAY'}
        </button>
        <div className="flex gap-[4px]">
          {([0.25, 0.5, 1, 2, 8] as const).map(s => (
            <button
              key={s}
              className={speed === s ? 'on' : ''}
              onClick={() => { setSpeed(s); clock.speed = s }}
            >
              {s}&times;
            </button>
          ))}
        </div>
        <div ref={clockEl} className="t-val" style={{ letterSpacing: '0.08em', marginLeft: 'auto' }}>T+00:00:00</div>
      </div>
    </div>
  )
}
