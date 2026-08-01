/* ────────────────────────────────────────────────────────────────────────
   Chrome.tsx — tactical terminal frame (UI-SPEC 4, 5, 7).

   Not just a border: a faint containment frame, corner brackets with corner
   nodes, edge tick-rulers, top/bottom status bars (menu words · server status ·
   grid labels), scattered crop marks, and the noise + scanline + sweep texture.
   All pointer-events:none, above the globe, below the floating panels.
   ──────────────────────────────────────────────────────────────────────── */

import { useEffect, useState } from 'react'

const FR = 14          // frame inset from the viewport edge
const ARM = 26         // corner-bracket arm length

const NOISE =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")"

const NO_EV: React.CSSProperties = { pointerEvents: 'none' }
const RULER = 'repeating-linear-gradient(90deg, var(--ink-faint) 0 1px, transparent 1px 26px)'
const RULER_V = 'repeating-linear-gradient(0deg, var(--ink-faint) 0 1px, transparent 1px 26px)'

function Crop({ left, top }: { left: string; top: string }) {
  return (
    <svg style={{ position: 'fixed', left, top, width: 8, height: 8, overflow: 'visible', zIndex: 12, ...NO_EV }}>
      <line x1="4" y1="0" x2="4" y2="8" stroke="var(--ink-faint)" strokeWidth="1" />
      <line x1="0" y1="4" x2="8" y2="4" stroke="var(--ink-faint)" strokeWidth="1" />
    </svg>
  )
}

function LiveClock() {
  const [t, setT] = useState('--:--:--')
  useEffect(() => {
    const upd = () => setT(new Date().toISOString().slice(11, 19))
    upd()
    const id = setInterval(upd, 1000)
    return () => clearInterval(id)
  }, [])
  return <>{t} UTC</>
}

export default function Chrome() {
  const c = '1px solid var(--ink-dim)'
  const bkt: React.CSSProperties = { position: 'fixed', width: ARM, height: ARM, zIndex: 12, ...NO_EV }
  const node: React.CSSProperties = { position: 'fixed', width: 3, height: 3, background: 'var(--accent)', zIndex: 13, ...NO_EV }

  return (
    <>
      {/* faint containment frame */}
      <div style={{ position: 'fixed', inset: FR, border: '1px solid var(--ink-faint)', zIndex: 11, ...NO_EV }} />

      {/* corner brackets + corner nodes */}
      <div style={{ ...bkt, top: FR - 1, left: FR - 1, borderTop: c, borderLeft: c }} />
      <div style={{ ...bkt, top: FR - 1, right: FR - 1, borderTop: c, borderRight: c }} />
      <div style={{ ...bkt, bottom: FR - 1, left: FR - 1, borderBottom: c, borderLeft: c }} />
      <div style={{ ...bkt, bottom: FR - 1, right: FR - 1, borderBottom: c, borderRight: c }} />
      <div style={{ ...node, top: FR - 1, left: FR - 1 }} />
      <div style={{ ...node, top: FR - 1, right: FR - 1 }} />
      <div style={{ ...node, bottom: FR - 1, left: FR - 1 }} />
      <div style={{ ...node, bottom: FR - 1, right: FR - 1 }} />

      {/* edge tick rulers */}
      <div style={{ position: 'fixed', top: FR, left: FR + 44, right: FR + 44, height: 5, background: RULER, zIndex: 12, ...NO_EV }} />
      <div style={{ position: 'fixed', bottom: FR, left: FR + 44, right: FR + 44, height: 5, background: RULER, zIndex: 12, ...NO_EV }} />
      <div style={{ position: 'fixed', left: FR, top: FR + 44, bottom: FR + 44, width: 5, background: RULER_V, zIndex: 12, ...NO_EV }} />
      <div style={{ position: 'fixed', right: FR, top: FR + 44, bottom: FR + 44, width: 5, background: RULER_V, zIndex: 12, ...NO_EV }} />

      {/* top status bar — menu words + server status */}
      <div style={{ position: 'fixed', top: FR + 9, left: FR + 16, right: FR + 16, display: 'flex', justifyContent: 'space-between', zIndex: 12, ...NO_EV }}>
        <div className="t-micro dim" style={{ letterSpacing: '0.24em' }}>IRIDIUM-NEXT · SGP4 PROPAGATED</div>
        <div className="t-micro dim" style={{ display: 'flex', alignItems: 'center', gap: 6, letterSpacing: '0.2em' }}>
          <LiveClock />
          <span style={{ width: 5, height: 5, background: 'var(--accent)', display: 'inline-block' }} />
        </div>
      </div>

      {/* bottom status bar — grid + mission id */}
      <div style={{ position: 'fixed', bottom: FR + 9, left: FR + 16, right: FR + 16, display: 'flex', justifyContent: 'space-between', zIndex: 12, ...NO_EV }}>
        <div className="t-micro faint" style={{ letterSpacing: '0.24em' }}>8 SV · 2 PLANES · 5 GND · 4500 KM ISL</div>
        <div className="t-micro faint" style={{ letterSpacing: '0.2em' }}>REDROCKET · ONBOARD DIAGNOSIS NET</div>
      </div>

      {/* crop marks */}
      <Crop left="30%" top="24px" />
      <Crop left="64%" top="24px" />
      <Crop left="calc(100% - 26px)" top="52%" />
      <Crop left="24px" top="60%" />

      {/* texture */}
      <div className="noise" style={{ backgroundImage: NOISE }} />
      <div className="scanlines" />
      <div className="sweep" />
    </>
  )
}
