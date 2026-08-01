# Deep-Space Reskin — Context & Handoff Brief

> **Read this first.** It is the single source of truth for the LEO → deep-space
> pivot: why we're doing it, what is load-bearing and must NOT be touched, what
> actually changes, how Gemma plugs in, and the pitch. Everything here is
> grounded in the code as it exists on 2026-08-01. Where it says "preserve," it
> means the thing is already correct and moving it will cost you.

---

## 0. The decision, in three lines

1. **We are keeping the entire simulator and reskinning it as a deep-space
   relay network** (recommended body: **Mars**; lunar/cislunar is a documented
   fallback). This is a *reskin + renarrate*, **not** a physics rewrite.
2. **The reskin is mostly labels, copy, one globe texture, and finishing the one
   module that was always going to be the star: the Gemma advisor.**
3. **The project fits deep space *better* than it ever fit LEO.** The pivot makes
   the story more honest, not less.

---

## 1. Why LEO → deep space makes sense (this is the core argument — internalize it)

The project's central premise is:

> *A scheduled contact goes silent. That one observable can mean five different
> things with five opposite right answers, and the spacecraft **cannot ask
> anyone** — so it must diagnose the cause **autonomously, onboard**, and act.*

That premise is **weak for LEO and strong for deep space.**

- **In LEO, you *can* ask.** A low-Earth satellite gets a ground pass every
  ~30–90 minutes. "It can't phone home" is not really true — help is one orbit
  away. A judge who thinks for ten seconds notices.
- **In deep space, you genuinely cannot ask.** One-way light-time to Mars is
  **4–24 minutes** (round-trip up to ~48 min). Earth physically cannot be in the
  fault-response loop. Onboard autonomous diagnosis stops being a nice-to-have
  and becomes **mandatory**. The premise becomes true.

And the machinery we already built is **deep-space-native to begin with**:

- **The baseline — Contact Graph Routing over DTN bundles — was invented for
  deep space.** CGR / DTN / the "Interplanetary Internet" come out of NASA JPL
  and Vint Cerf's work; ION (the reference CGR implementation we follow) is a
  deep-space networking stack. We've been dressing deep-space routing in LEO
  clothes. Taking the costume off makes everything *more* coherent.

**How it maps onto the hackathon track (Track 2 — Trajectory & Orbit / Deep
Space Navigation).** The track asks for tools that assist with:

| Track phrase | What we already have |
|---|---|
| "autonomous navigation when communicating with Earth is delayed" | The entire thesis — onboard diagnosis under light-time delay |
| "orbital anomaly detection" | The **pointing / attitude (star-tracker)** fault is exactly an orbital/attitude anomaly the node detects itself |
| "real-time telemetry analysis" | Per-node belief panels reading live telemetry into a diagnosis |
| "edge-deployed models" | **Gemma-3-4b running onboard each spacecraft** is the textbook edge story |
| "multimodal vision tools" | Gemma-3-4b is multimodal — optional stretch: feed it a star-tracker frame or a telemetry plot image |

We hit four of the five track keywords without inventing anything new. The pivot
is aligning the *framing* to machinery that was already pointed this way.

---

## 2. The scenario (recommended: Mars relay)

**Body:** Mars. **Nodes:** a small relay network judges will instantly recognize —
relay **orbiters** (think MRO / MAVEN / Odyssey / Mars Express class), **surface
assets** (rovers, landers, beacons), and **Earth's Deep Space Network (DSN)** as
the ultimate sink reached *only through the relays*. Orbiters cross-link (ISL)
and relay for the surface; that is our gossip + contact-graph structure, unchanged.

**Why Mars over the Moon:** the delay is the whole point of the track, and Mars's
4–24 min one-way light-time makes "Earth can't help in the loop" *true*. The Moon
(~1.3 s one-way) keeps the geometry slightly more faithful to the current 12
surface sites, but its delay is too small to force autonomy — the story goes soft.

**Lunar/cislunar is the documented fallback.** If Mars framing feels like a
stretch for the surface-station count, switching to a LunaNet-style lunar relay
is a find-replace on the body texture + delay numbers + a few labels. Same
geometry, weaker delay story. Decide once and commit.

**Reskin honesty note:** the underlying propagation is still the tested
Earth/Iridium geometry producing a valid time-varying contact graph. We are
**not** claiming true heliocentric ephemerides. No judge audits TLEs; they audit
the story and the demo. The orbits look like relay orbits around a body — that is
all the scene needs. Spend the saved risk budget on Gemma and the frontend.

---

## 3. What to PRESERVE (do not touch — it's correct and load-bearing)

Everything in this section already works and is tested. Reskinning means
relabeling around it, not reopening it.

- **The geometry engine — `sim/contacts.py`, `sim/diagnosability.py`.** TLE +
  SGP4 → contact windows → the diagnosability strip. Locked after a real
  27-config sweep (`n_sats=8, n_planes=2, isl_max_km=4500, 12 stations, 10°
  mask, 24 h`). It emits a valid time-varying contact graph. **Reskin the labels
  on top of this; do not reopen the math.**
- **The diagnosis core — `sim/advisor/bayes.py` + `sim/advisor/base.py`.** The
  Bayes filter, the shared `COST` matrix, the emission model (WEATHER and
  POINTING deliberately share an attenuation row because they're unidentifiable
  from one link — that subtlety is the whole intellectual point), the per-cause
  transition kernel. This is the accuracy anchor Gemma gets measured against.
  **125 tests + 13 tracked defects sit on top of these enum names and
  constants.** Renaming the enum *values* will ripple through all of it.
- **The diagnosability strip.** The offline "when are causes even separable"
  computation, brute-force cross-checked, mixed-dominant at 92.7%. **This is the
  most impressive and most defensible thing in the project.** Preserve it exactly.
- **The gossip mechanism.** Compact structured status vectors, `(origin, seq)`
  dedup, the measured 100× belief swing driven purely by evidence that physically
  travelled. Preserve.
- **The trace contract + frontend data pipeline.** `out/trace.json` (598 KB, one
  fetch, keys: `meta / nodes / positions / contacts / faults / beliefs / gossip
  / fails / strip / summary`). RLE strip encoding. The frontend reads exactly one
  file. Don't add a second data source.
- **The test suite.** `python -m pytest tests/ -q` → 123 passed, 13 xfailed.
  Keep it green; it's your proof the reskin didn't break the sim.

**The golden rule of this reskin:** *rename in the display layer and the docs,
not in the code enums.* Keep `Cause.WEATHER / POINTING / NODE_DOWN / BUFFER /
STALE_SCHED` as internal identifiers (the Bayes constants, tests, and trace all
key off them) and map them to deep-space display names in the frontend + pitch.
A clean enum rename is a day of chasing test breakage for zero demo value.

---

## 4. What to CHANGE (the actual reskin work, smallest-first)

### 4.1 Fault taxonomy — remap display names (code enums stay)

Same structure — one silence, several causes, opposite right answers — just
deep-space clothing. This is a **display + docs** remap; the internal enum keeps
its current name.

| Internal enum (keep) | Deep-space display name | What it is | Right response |
|---|---|---|---|
| `WEATHER` | **Solar conjunction / plasma scintillation** (or DSN weather) | Sun/atmosphere corrupts the link; self-heals over time | **Wait** |
| `NODE_DOWN` | **Safe-mode / fault-protection trip** | Craft dropped to safe mode; absorbing until ground intervenes | **Blacklist / route around** |
| `POINTING` | **High-gain antenna pointing / attitude (star-tracker) anomaly** | Mispointed HGA or attitude error — an *orbital anomaly* | **Reroute** |
| `BUFFER` | **Onboard solid-state recorder saturation** | SSR full; more traffic makes it worse | **Throttle** |
| `STALE_SCHED` | **Stale uplinked sequence / ephemeris drift** | Nav uncertainty shifts the predicted contact window | **Re-sync** |

Every one still produces the same observable: **a scheduled contact that doesn't
open.** That invariant is what the whole diagnosis problem rests on — it survives
the reskin untouched.

### 4.2 Copy / narrative (docs + UI strings)

- `PLAN.md` §1 "What the system does" — reframe LEO → Mars relay, swap the cause
  table for the deep-space one above, keep the mechanism description verbatim.
- Title / one-liners everywhere (PLAN, PROGRESS, README, UI chrome).
- `data/ground_stations.json` — relabel the 12 sites as Mars surface assets /
  DSN complexes. Positions can stay; only names/labels change.

### 4.3 The globe body (`web/src/globe.ts`)

Swap the Earth basemap for **Mars** (or Moon). It's a `MeshBasicMaterial`
textured sphere with a monochrome equirectangular basemap and `CROP = 0.78` so
it overflows the viewport. Retexture only — the projection, crop, and overlay
math are shared with the 2D canvas and must not move. This is the one real
frontend change the reskin requires.

### 4.4 Light-time delay (narrative, optional-but-recommended)

Add a one-way light-time figure (~13 min nominal for Mars) to the copy and,
if cheap, a visible "Earth round-trip: 26 min" readout. You do **not** need to
re-time the sim for this — it's a framing number that makes the autonomy story
land. If you want it in the mechanics, it's a delay applied to the Earth/DSN leg
only, not to inter-orbiter gossip.

### 4.5 Fault physics constants (`sim/faults.py`, optional tuning)

Conjunction self-heals over *days*, not the 5-min weather dwell. For a 2-minute
demo on a 24 h scenario, **leave the dwell short** so the self-heal is visible —
narrate it as "the geometry clears," not "rain stops." Don't rebalance the
physics unless you have spare time; it changes nothing a judge can see.

---

## 5. Gemma — the required deliverable (currently unbuilt)

**Status:** the advisor swap-point (`sim/advisor/base.py`, the `Advisor`
protocol: `observe` / `receive` / `decide`) is clean and interchangeable. The
`null` (baseline) and `bayes` advisors exist. **`gemma.py` does not exist yet.**
This is the one module the hackathon *requires* and it's the missing one.

**Model:** "gemma 4" almost certainly means **`gemma-3-4b-it`** — the 4-billion-
param model. It's the right pick for two reasons: it's the edge-deployment sweet
spot, and it's **multimodal (vision)**, which lines up with the track's
"multimodal vision tools." Served via the Gemini API / Google AI Studio with an
API key (also OpenRouter / Together). *There is no Gemma 4; if a different model
was meant, confirm before building.*

**Where it plugs in:** Gemma **is** the advisor. It drops into the empty slot
behind `base.py`, consumes `Observation` objects + gossiped `Evidence` (compact
structured status vectors — **no natural language crosses the link**), and emits
a diagnosis + a `Policy`. The `Policy.rationale` field is **already reserved** for
Gemma's natural-language explanation — the Bayes filter leaves it empty on
purpose. So Gemma completes the design; it doesn't fight it.

**Keep the Bayes filter.** It stays as the honest accuracy anchor. "Our LLM is
measured against a decision-theoretically optimal Bayes filter on the same cost
matrix" is a *strength* — show the comparison, don't hide it.

**The edge-deployment framing (resolves the api-key tension).** The track rewards
edge models; we're calling a hosted API. Say it straight: *"Gemma-3-4b is small
enough to run on the spacecraft's flight computer. We call the API for the demo,
but the entire point is that it's edge-sized and onboard-deployable — each node
runs its own instance, and only compact status vectors cross the link."* That's
honest and it's the winning story. The PLAN already asserts exactly this
("Gemma runs locally on each node").

**Interpretability is the contribution.** Per the project description, the
novelty is not the routing — it's the **interpretable diagnosis layer**: a
diagnosis a flight controller can *read*. That's Gemma's rationale text, surfaced
on demand in the per-node belief panel. Lead with it.

**Optional multimodal stretch (only if ahead):** hand Gemma-3-4b a star-tracker
frame or a rendered telemetry plot for the pointing/attitude anomaly. Directly
claims the track's "multimodal vision" keyword. Skip unless the core demo is solid.

---

## 6. The honest risks — say these to yourself, not to the judges

From the project's own `PROGRESS.md`, independent of the reskin:

1. **The frontend has never once been looked at.** It typechecks; nobody knows a
   globe actually renders. **This is the #1 demo risk.** First move on any build
   session: `cd web && npm run dev` and look at the screen.
2. **The Gemma advisor doesn't exist** (see §5). #1 requirement gap.
3. **Don't quote the identifiability numbers** (0.778 vs 0.577) — n=9 in one bin,
   underpowered. Say "accuracy is materially higher where the geometry permits
   diagnosis" and leave the number off the slide until the power run finishes.
4. **The delivered/dropped race is a dead heat.** The network is over-provisioned;
   both arms deliver identically. **Keep that counter off screen** until traffic
   is resized, or cut it. Do not show a tied race and call it a win.

---

## 7. The pitch (golden info)

**One-liner options:**
- *"A second opinion for spacecraft that can't phone home."*
- *"Onboard fault diagnosis for deep-space relays — when Earth is 20 minutes away."*

**The 2-minute demo, in the order it should play:**
1. **The constellation breathing.** Mars, relay orbiters on their arcs, contact
   links opening and closing before anything breaks. Already better-looking than
   most of the room.
2. **Inject a fault** (a *conjunction/weather*-class fade at a relay, **not** node
   death — node death is independent detection wearing the costume of distributed
   inference, and a sharp judge sees through it). Everything looks fine for a beat.
3. **One halo goes uncertain and flickers between two causes** — the node that
   noticed can't yet tell conjunction from a pointing error. Its neighbours are
   calm, oblivious.
4. **A relay contact opens, a gossip pulse crosses the arc, the neighbour's halo
   resolves** — then the next one. You physically watch a diagnosis spread across
   the constellation at the speed of orbital mechanics.
5. **The diagnosability band underneath turns out to have predicted exactly where
   the flickering happened.** This is the beat that makes people sit up.
6. **Scrub back and replay it slower.** Judges love controlling a demo.

**The intellectual hook judges remember (lead with this if you lead with one
thing):** *"We can prove, before running any inference, when two causes are
physically indistinguishable from a single link. Where our strip is red, no model
— not ours, not anyone's — can tell them apart, and Gemma correctly stays
uncertain there. Where it's green, a neighbour's gossip resolves it. We're not
claiming a magic classifier; we're showing exactly where diagnosis is possible
and letting evidence do the rest."* Honesty is the differentiator.

**Why it's ours / prior art to acknowledge (pre-empt the skeptical judge):**
epidemic and link-state gossip are old; POMDP routing under uncertain contact
plans already gets near-oracular delivery. **The contribution is the interpretable
diagnosis layer, not the routing.** Naming the prior art unprompted reads as
competence, not weakness.

**Track-keyword checklist to hit out loud:** autonomous under delay ✓ · orbital
anomaly detection (the pointing/attitude fault) ✓ · real-time telemetry analysis
(belief panels) ✓ · edge-deployed model (Gemma-3-4b onboard) ✓ · multimodal
vision (stretch) ◻.

---

## 8. Open decisions the next agent must make

1. **Mars vs lunar** — recommendation: Mars (delay story). Commit before writing copy.
2. **Display-relabel vs enum rename** — recommendation: display-relabel, keep code
   enums (protects 125 tests + Bayes constants). Only rename enums if you have a
   spare day and a reason.
3. **Build order** — recommendation: (a) verify the frontend renders, (b) build
   `gemma.py`, (c) apply the reskin copy/labels/texture. Frontend first because
   it's the biggest unknown; Gemma second because it's the requirement; cosmetics
   last because they're cheap and low-risk.
4. **Multimodal stretch** — only if the core demo is already solid.

---

## 9. File map (where the reskin work actually lands)

| File | Reskin action |
|---|---|
| `PLAN.md` §1, title | Reframe LEO → Mars relay; swap cause table for §4.1 above |
| `PROGRESS.md` | Record the pivot; Gemma + frontend-render remain the two gaps |
| `data/ground_stations.json` | Relabel 12 sites as Mars surface assets / DSN (positions can stay) |
| `web/src/globe.ts` | Retexture Earth → Mars basemap; leave projection/crop untouched |
| `sim/advisor/gemma.py` | **Create** — Gemma advisor behind `base.py`, emits diagnosis + `Policy.rationale` |
| `web/src/panels/*`, UI strings | Deep-space display labels; surface Gemma rationale in the belief panel |
| `sim/faults.py` | Optional constant tuning only; do not rebalance unless time allows |
| `sim/types.py`, `bayes.py`, tests | **Do not touch enum values** — reskin is display-layer |

---

*Written 2026-08-01 as the handoff brief for the deep-space reskin. Grounded in
the codebase state described in `PROGRESS.md` (simulator complete + tested;
frontend written but never visually verified; Gemma advisor unbuilt).*
