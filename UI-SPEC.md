# UI Specification — Tactical Orbital Display

> **Amendments live in [PLAN.md §6](PLAN.md).** Two things override or extend this document:
> 1. **§6.3 Colour semantics** — the palette carries two semantic colours; the fault taxonomy has six causes. Colour encodes *epistemic state* (nominal / uncertain / resolved / fault / stale); the cause itself is text. Do not add colours to this palette.
> 2. **§6.5 Timeline** — this spec has no time control. Region **[P]** is added bottom-centre: scrubber plus diagnosability band, in this document's visual language.
>
> Everything else here is authoritative for appearance.

Emulate a military tactical command display: dense, monochrome, hairline-thin, with sparse accent colour used only where it carries meaning. Reference register is film-production FUI (fictional user interface) for a command-and-control screen — **not** a dashboard, **not** pixel art, **not** a SaaS admin panel.

The governing principle: **the screen is an instrument, not a page.** Everything floats over a live map. Nothing has a solid card background. Nothing is rounded. Nothing glows.

---

## 1. Palette

Exactly six values. Do not add a seventh.

```css
--bg:        #0A0B0D;  /* near-black, very slightly blue */
--ink:       #E8EAED;  /* off-white — all text, all lines */
--ink-dim:   rgba(232,234,237,0.45);  /* secondary text, minor rules */
--ink-faint: rgba(232,234,237,0.18);  /* grid, dividers, inactive */
--accent:    #2EC4C4;  /* teal — friendly / active / selected */
--alert:     #E03434;  /* red — fault / warning / hostile */
```

Optional seventh, used for exactly one marker class: `--neutral: #E8DCC0` (pale cream).

**Rules:**
- The map, all chrome, all text: monochrome. Greyscale only.
- Teal and red are **semantic**, never decorative. If a colour appears, it means something.
- No gradients. No glows. No drop shadows. No `border-radius` above 2px.
- Panel backgrounds: `rgba(10,11,13,0.82)` — always translucent, map visible beneath.

---

## 2. Typography

One family, condensed grotesque, used at every level:

```css
font-family: "Roboto Condensed", "Barlow Condensed", "Oswald", sans-serif;
```

| Role | Size | Weight | Tracking | Case |
|---|---|---|---|---|
| Panel title | 13px | 700 | 0.08em | UPPER |
| Panel subtitle | 9px | 400 | 0.14em | UPPER |
| Body / label | 10px | 400 | 0.06em | UPPER |
| Data value | 11px | 500 | 0.02em | as-is |
| Micro label | 8px | 400 | 0.18em | UPPER |
| Marker chip number | 15px | 700 | 0 | as-is |

**Everything is uppercase except numeric values.** Line-height 1.35 throughout. Never exceed 15px except marker chips.

---

## 3. Line and stroke rules

- All rules and borders: **1px**. Never 2px, never 3px.
- Panel dividers: `--ink-faint`. Panel outlines: `--ink-dim`.
- Zero border-radius everywhere except marker chips (2px).
- No bevels, no insets, no rivets, no ornamental frames.

---

## 4. Layout

Full-viewport, absolutely positioned regions over a full-bleed canvas. No CSS grid gutters between regions — they float independently with generous black between them.

```
┌──────────────────────────────────────────────────────────────────────┐
│ [A] menu bar          [B] tabs        [C] server status              │
│ ┌────────────┐                              ┌──────────────────────┐ │
│ │[D] mission │                              │ [G] alert box        │ │
│ │    header  │                              └──────────────────────┘ │
│ └────────────┘        ╭────────────────╮     [H] coord lock          │
│ ┌────────────┐       ╱                  ╲                            │
│ │[E] roster  │      │   [F] MAP FIELD    │   [I] data table          │
│ │  01 ▸▸▸▸   │      │   wireframe globe  │                           │
│ │  02 ▸▸▸▸   │      │   markers + arcs   │   ▨▨▨ [J] warning strip   │
│ │  03 ▸▸▸▸   │       ╲                  ╱                            │
│ │  ...       │        ╰────────────────╯     [K] mode select         │
│ └────────────┘                                                       │
│ ┌────────┐          [P] timeline + diagnosability  ┌────────────────┐ │
│ │[L] nav │        [M] scale   [N] link status      │ [O] feed inset │ │
│ └────────┘                                         └────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

**Density gradient:** dense at centre and left, sparse at the right and edges. The right side is mostly black with small floating text clusters — resist the urge to fill it.

**Outer frame:** a 1px `--ink-dim` rule inset 12px from the viewport edge, with 40–80px gaps broken into it at irregular intervals. Corner brackets (two 24px strokes forming an L) at all four corners, drawn *outside* the frame line.

---

## 5. Component inventory

### [A] Menu bar
Top-left through centre. Horizontal list of 5–7 uppercase words at 10px, `--ink-dim`, 24px apart. Purely decorative — no interaction required. Words: `MISSION  EDIT  PRESET  SELECT  VIEW  WINDOW`.

### [B] View tabs
Centre-top. 3 tabs, 10px uppercase. Active tab has an `--ink` 1px box around it with 6px padding and `--ink` text; inactive are `--ink-dim`, no box. A small `+` glyph sits before and after the tab group. Labels: `ORB  GND  LNK`.

### [C] Server status
Top-right, single line, 9px: `LINK SERVER: STATUS ONLINE`. Right-aligned.

### [D] Mission header
Top-left, below the menu. Title `SITE MAP` at panel-title size with a 1px `--ink-dim` rule beneath it running ~180px. Under the rule, 5 lines at 8px `--ink-dim`, `label: value` pairs, tight line-height:

```
LOCATION: LOW EARTH ORBIT
CONSTELLATION: <name>
MISSION TIME: 04:08:16
OPERATOR: AUTONOMOUS
OBJECTIVE: LINK DIAGNOSIS
```

### [E] Roster panel — **the primary component**
Left edge, ~200px wide, vertically centred.

- Title: `TRACKED ASSETS` (13px/700). Subtitle beneath: `CONSTELLATION STATUS TRACKER` (9px, `--ink-dim`).
- 8–20 rows. Each row is 34px tall:
  - Left: 2-digit index (`01`) at 9px, `--ink-dim`, in a 22px column with a 1px right divider.
  - Centre: asset designation at 10px uppercase `--ink`.
  - Right: 2-digit index repeated at 9px `--ink-dim` (an intentional redundancy that reads as instrumentation).
- **Sheared left edge.** Each row's left edge is cut on a diagonal rather than square — a shallow parallelogram, roughly 8–10px of horizontal offset over the 34px height. This is one of the most recognisable details in the reference and it is easy to miss. `clip-path: polygon(...)` on the row, or an SVG backing shape.
- Rows alternate: even rows `rgba(232,234,237,0.03)` fill, odd rows transparent.
- 1px `--ink-faint` divider between rows.
- **Status LED:** a 5px square at the row's left edge, coloured by that asset's current diagnosis state. This is where colour enters.
- Selected row: 1px `--accent` outline, text goes `--ink`.

### [F] Map field — the globe

> ⚠️ **Largest departure from the reference. Read this before building.**
>
> The reference image uses a **photographic satellite basemap** — a dark, desaturated aerial view of a city, bleeding off all four edges. A very large share of the composition's density and mid-tone range comes from that photo. The vector chrome reads as *overlay*, and it only reads that way because there is something underneath it.
>
> A pure wireframe sphere on flat black loses all of it. Worse, the project's view is **global** where the reference is **local** — a whole Earth centred in frame leaves large empty black margins the reference never has.
>
> **Do both of these:**
> 1. **Texture the globe** with desaturated satellite imagery (NASA Blue Marble, pushed to monochrome and darkened to roughly 10–25% luminance). Keep the graticule and continent outlines *on top* of it.
> 2. **Crop, don't fit.** Scale the globe so it overflows the viewport on all edges, exactly as the reference photo does. Never render a complete circle floating in black — that is the single change that will make it stop looking like the reference.

Full-bleed behind everything.

- **Textured sphere with wireframe overlay.** Desaturated satellite basemap beneath; latitude lines every 15°, longitude every 15°, 1px, `--ink-faint`; continents as 1px `--ink-dim` outlines, no fills.
- **Dotted overlay grid**: 1px dots on a 14px pitch at `rgba(232,234,237,0.06)`, covering the whole viewport, sitting *above* the globe.
- **Reticle:** two concentric circles near the viewport centre, 1px `--ink-dim`, radii computed from `min(vw,vh)` — roughly 0.32× and 0.12×. Bearing labels at 9px just outside the outer circle. **The reference uses `360 / 240 / 090 / 180`, not evenly spaced cardinals** — the irregularity is deliberate and reads as instrumentation.
- **Per-marker range rings.** Individual assets may carry their own thin circle, ~60–90px radius, 1px `--ink-faint`, centred on the chip. Apply to 2–3 markers only, never all — the reference uses them sparsely.
- **Crosshair rules:** one vertical and one horizontal 1px `--ink-faint` line through the viewport centre, broken where they pass through panels.

### Marker system (on the map)

1. **Asset chip** — a white `--ink` filled rectangle, 2px radius, ~38×22px, containing the asset's 2-digit index in near-black. Immediately right of it, a 16px circle outline containing a 6px filled dot coloured by diagnosis state.
   - **Above** the chip: a status word at 8px `--ink-dim` — reference uses `LOCKED`; ours uses `NOMINAL` / `DEGRADED` / `SILENT` / `STATION WX`.
   - **Below** the chip: a metric at 8px `--ink-dim` — reference uses `DISTANCE 191M`; ours uses `NEXT CONTACT 14:22`.
   - **Below-left**, outside the box: a second, *larger* figure at ~15px `--ink` (the reference shows `33`, `02`). Use the asset's contact count or queue depth. This size contrast is a distinctive part of the look — do not flatten it to one type size.
2. **Ground station** — a 12px teal `--accent` triangle, apex up. Label at 8px beside it.
3. **Fault marker** — **a filled red `--alert` diamond** (rotated square), ~14px across. *Not* a plus or cross — the reference uses solid diamonds and they read very differently at a glance.
4. **Neutral marker** — a pale cream `--neutral` **quatrefoil** (four-lobed clover), ~12px, filled. The reference scatters 5–6 of these. In our build, use them for ground stations currently under a weather fault.

### Connector arcs
1px `--accent` lines from asset chips to their contact partner, with a slight quadratic curve. Each terminates in a small 10×8px rectangle outline containing a 2-digit number. Arcs animate: draw in over 300ms when a contact window opens, fade out over 200ms when it closes.

### Chevrons
Large open `V`, `Λ`, `<`, `>` glyphs in `--ink`, **1px stroke, 60–80px** — larger and thinner than first specified. The reference places them at the **cardinal points of the reticle**, adjacent to the bearing labels, not scattered at random. Four is right; six is cluttered. The reference's most distinctive decorative element — include them.

### [G] Alert box
Top-right. Bracket-framed (corner brackets only, not a full box), ~230px wide. Header line: `// SYSTEM UPDATE` with a 5px `--alert` dot to its right. Below, 3 lines of 8px `--ink-dim` text, each prefixed `//`, describing the current event. Final line always: `// STANDBY FOR FURTHER INSTRUCTIONS`.

### [H] Coordinate lock
Right side, below the alert box. Two-column micro table, 8px:

```
LOCATION LOCK    W 142.0901   W 142.0901
ORB IN           N  35.4223   N  35.4223
```

### [I] Data table
Right side. Three columns of 6 rows. Each cell: a 2-letter code (`AA`, `AB`…), then a numeric value at 8px. Values are static decorative telemetry — they do not need to mean anything, but should tick occasionally to feel live.

### [J] Warning strip
Right side, full-width of the right column, 22px tall. Diagonal hatch fill (45° repeating-linear-gradient, `--ink-faint` on transparent, 4px pitch), 1px `--ink-dim` outline. A small triangle glyph at the left end, an `×` at the right end, and the word `WARNING` in `--alert` at 11px/700, right-of-centre. Hidden until a fault is active.

### [K] Mode select
Right side, lower. Label `MODE SELECT` at 8px `--ink-dim`, then 2 lines of 9px `--ink`. Below it a 4-item vertical list (`BELIEF / TRACE / LINK / EPHEM`), one item boxed in 1px `--ink` to show selection.

### [L] Navigation cluster
Bottom-left. A `ZOOM` label with two 26px circle-outline buttons (`+`, `−`) and a percentage readout at 11px. Beside it, a `NAVIGATION` label above a 56px circle outline containing four directional carets and a centre dot. Below both: a row of five 22px square outline buttons containing simple 1px glyphs.

### [M] Scale bar
Bottom-centre-left. A horizontal 1px rule ~110px with 5 tick marks descending from it, numeric labels at 7px beneath, and the unit label `KM` at 8px above the left end.

### [N] Link status
Bottom-centre, centred. Two lines, 9px, `--ink-dim`:

```
CONNECTION . . . GOOD
COMMS OPEN
```

Text swaps to `--alert` and `COMMS LOST` when a fault is active.

### [O] Feed inset
Bottom-right, ~250×130px. 1px `--ink-dim` outline with a solid `--accent` triangle occupying the top-right corner (a 16px right-triangle). Header strip above: `02  FEED: <asset designation>` at 8px. Inside: the belief breakdown or an OpNav frame. A 3px-tall waveform strip along the bottom edge.

### [P] Timeline + diagnosability *(added — see PLAN.md §6.5)*
Bottom-centre, full width of the map field.

```
 ▮▮▮▮░░░░▮▮▮▮▮▮▮▮░░░░░░▮▮▮▮▮▮        ← diagnosability, 8px tall
 ├────┴────┴────┴────┴────┴────┤      ← 1px rule, tick marks descending
        ▲                              ← playhead, 1px --accent vertical
 ▶ ⏸   1×  2×  8×          T+04:17:22
```

Solid `--ink-dim` where identifiable, 45° diagonal hatch where not (same treatment as [J]). Rule and ticks match [M] exactly. Playhead is the only `--accent` element. Transport controls at 9px uppercase, active speed boxed in 1px `--ink` per [B].

### Crop marks
Small `+` glyphs, 8px, 1px stroke, `--ink-faint`, scattered at ~12 positions: panel corners, frame intersections, and 3–4 arbitrary points in empty space. **Do not skip these** — they are disproportionately responsible for the aesthetic.

---

## 6. Motion

Restrained and mechanical. No easing curves softer than `ease-out`. No bouncing, no scaling, no fades longer than 300ms.

- Globe rotates continuously, very slowly (~0.02°/frame).
- Contact arcs draw in / fade out as windows open and close.
- Status LEDs and diagnosis dots **snap** between colours — no transition.
- An uncertain node's dot alternates between two colours on a hard 400ms interval (no crossfade). **This is the single most important animation in the interface.**
- Data table values tick every 2–4s.
- Optional: a 1px horizontal scanline sweeping the viewport every 8s at 4% opacity.

---

## 7. Texture

- Full-viewport noise overlay: a tiled PNG or SVG `feTurbulence`, `opacity: 0.04`, `mix-blend-mode: overlay`, `pointer-events: none`.
- Horizontal scanlines: `repeating-linear-gradient(rgba(0,0,0,0.14) 0 1px, transparent 1px 3px)`, `pointer-events: none`.

Both sit above everything at the top of the stacking context.

---

## 8. Anti-patterns — do not do these

- No rounded cards, no soft shadows, no gradient fills.
- No colour outside the six tokens. No purple, no orange, no blue-violet.
- No glow, no bloom, no neon.
- No icon libraries (Lucide, Font Awesome, Material). All glyphs are drawn with 1px SVG strokes.
- No text above 15px anywhere.
- No solid panel backgrounds — always translucent over the map.
- No emoji.
- No bevels, rivets, or ornamental borders. Flat hairlines only.
- No centred layouts. Everything is edge-anchored.
- Do not fill the right side. Emptiness is part of the composition.

---

## 9. Build order

1. Full-bleed wireframe globe with rotation.
2. Dotted overlay grid, reticle, crosshair rules.
3. Roster panel (left) — the primary interactive surface.
4. Asset chips and connector arcs on the map.
5. Frame, corner brackets, crop marks.
6. Right-side clusters (alert, coords, data table, warning strip).
7. Bottom clusters (nav, scale, link status, feed inset).
8. Noise and scanline overlays last.

Steps 1–4 carry the demo. Steps 5–8 carry the aesthetic. If time runs short, cut from the middle of 6–7, never from 5.

**Time estimates and the hard cut line are in [PLAN.md §6.8](PLAN.md).**
