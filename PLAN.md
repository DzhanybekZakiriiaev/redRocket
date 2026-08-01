# Second Opinion — Build Specification

A Mars relay network where scheduled contacts fail, and each orbiter works out *why* on its own — because the round trip to Earth is 8–48 minutes and no one on the ground can answer in time.

> **Deep-space reskin (C-011).** This document was originally written for a LEO Earth constellation. The project is now framed as a **Mars relay network**: the 8 satellites are relay orbiters, the 5 operational "ground stations" are Mars surface sites, downlinks are orbiter↔surface proximity links, and crosslinks (ISLs) are unchanged. The offstage Deep Space Network is *why* autonomy is required — not a node in the sim. **The reskin is presentation, not rebuild:** the simulator, SGP4 geometry, contact plan, diagnosis layer, and `trace.json` are byte-for-byte unchanged; the Earth/Iridium ephemeris is kept as a visual stand-in. Where a section below still says "satellite," "ground station," or "LEO," read the deep-space equivalent per §5.3 and C-011. Sections not yet reworded are LEO-native by history, not by intent.

---

## Contents

1. [What the system does](#1-what-the-system-does)
2. [End-to-end data flow](#2-end-to-end-data-flow)
3. [Module map](#3-module-map)
4. [Interfaces — what connects to what](#4-interfaces--what-connects-to-what)
5. [Component specifications](#5-component-specifications)
6. [Screens](#6-screens)
7. [Requirements](#7-requirements)
8. [Development stages](#8-development-stages)
9. [Reference data](#9-reference-data)
10. [Changes](#10-changes)

---

## 1. What the system does

Mars relay orbiters can only communicate during scheduled contact windows — a surface pass over a rover or lander, or a crosslink with another orbiter — a few minutes at a time, minutes-to-hours apart. When a scheduled contact fails to open, that single observable can mean several different things:

| Cause | Correct response |
|---|---|
| Dust storm at the surface site | **Wait** — it clears over hours to days |
| Peer orbiter has failed (safe-mode / bus fault) | **Blacklist** — route around it permanently |
| Peer's onboard recorder is full | **Throttle** — sending more makes it worse |
| High-gain antenna mispoint / star-tracker drift | **Reroute** — use a different peer |
| Stale uplinked sequence / ephemeris drift | **Re-sync** — the contact is happening at a different time |

Same silence. Opposite right answers. **And the orbiter cannot ask anyone** — the round trip to Earth is 8–48 minutes depending on where the two planets sit in their orbits, so the Deep Space Network can neither see the failure nor answer in time. Each orbiter must form its own opinion about the cause, act on it, and pass what it saw to the next orbiter it meets. That is the deep-space autonomy problem, and it is the project's thesis verbatim.

**The system gives each node the ability to form and revise an opinion about the cause, and act on it.**

Three things run per node:
1. **Observe** — record what happened at each scheduled contact (opened? degraded? silent? late?)
2. **Infer** — maintain a probability distribution over causes, updated from local observations and from evidence gossiped by neighbours during successful contacts
3. **Act** — pick the response that matches the most likely cause

A fourth piece is computed offline from the schedule alone:

4. **Diagnosability** — determine, before any inference runs, *when the causes are separable at all*. If a failed link is the only thing happening on both ends at that moment, two different causes produce byte-identical evidence and no method can tell them apart. This is computable from the contact plan and is rendered as a strip under the timeline.

---

## 2. End-to-end data flow

```
data/iridium.tle ─────────┬──────────────────────────────────────┐
data/ground_stations.json ┤                                      │
                          ▼                                      │
                   contacts.py                                   │
                          │                                      │
                          ▼                                      │
                   contact_plan.jsonl                            │
                     │         │                                 │
          ┌──────────┘         └──────────┐                      │
          ▼                               ▼                      │
  diagnosability.py                  faults.py ◄── seed          │
          │                               │                      │
          ▼                               ▼                      │
  strip.json                       fault_trace.jsonl             │
          │                               │                      │
          │                               ▼                      │
          │                          engine.py ◄── traffic.py    │
          │                          ┌────┴─────┐                │
          │                          │  router  │                │
          │                          │ observe  │                │
          │                          │  gossip  │                │
          │                          │  policy  │                │
          │                          └────┬─────┘                │
          │                               │                      │
          │                    advisor (swap point)              │
          │                     ├── null                         │
          │                     ├── bayes                        │
          │                     └── gemma ──► cache.sqlite       │
          │                                    └──► llama-server │
          │                               │                      │
          │                               ▼                      │
          └──────────────────────► trace.json ◄──────────────────┘
                                         │
                                         ▼
                                    web/ frontend
```

**The frontend reads exactly one file: `trace.json`.** Positions, contacts, faults, beliefs, gossip and the diagnosability strip are all baked in by the simulator. No TLE parsing, no SGP4, no second fetch — and nothing to go wrong on demo day.

---

## 3. Module map

```
data/
  iridium.tle                 committed from CelesTrak, never fetched at runtime
  ground_stations.json        hand-authored, 10–15 real stations

sim/
  config.py                   ScenarioConfig — frozen, hashable, single source of truth
  contacts.py                 TLE + stations → ContactPlan
  diagnosability.py           ContactPlan → Strip            [depends on nothing else]
  faults.py                   ContactPlan + seed → FaultTrace
  traffic.py                  ScenarioConfig + seed → TrafficTrace
  observe.py                  contact outcome + node state → Observation
  gossip.py                   evidence exchange + dedup
  policy.py                   Policy → router constraints
  router.py                   first-contact forwarding
  engine.py                   event loop
  trace.py                    event log writer
  advisor/
    base.py                   Advisor protocol
    null.py                   no diagnosis (baseline)
    bayes.py                  discrete Bayes filter
    gemma.py                  LLM advisor
    llama_client.py           HTTP + SQLite response cache
  run.py                      CLI entry point

web/
  src/
    map/                      [F] three.js textured sphere + 2D overlay canvas
    roster/                   [E] left panel — all assets + state LEDs
    timeline/                 [P] scrubber + diagnosability band
    chrome/                   [A–D][G–N] frame, brackets, crop marks, clusters
    feed/                     [O] belief breakdown for selected asset
    lib/trace.ts              trace loader
    lib/project.ts            three.js camera projection + occlusion
    store.ts                  zustand
```

Region IDs in brackets refer to `UI-SPEC.md` §4. **Canvas draws everything that moves; React DOM draws everything static.**

**Rule: `policy.py` is the only path from a diagnosis to a routing decision.** Nothing else in the simulator reads the advisor's output. This keeps the comparison between advisors honest and makes the swap point real.

**Rule: the advisor never sees `FaultTrace`.** It receives `Observation` objects only. Everything else is ground truth and would leak the answer.

---

## 4. Interfaces — what connects to what

These six types are the contract. Freeze them before writing anything else; every module boundary is one of them.

### 4.1 `Contact` — output of `contacts.py`

> **All times in every type below are absolute Unix epoch milliseconds (UTC)** — not offsets from the scenario epoch. The frontend passes them straight to `new Date(t)` with no separate epoch to ship or get wrong.

```python
Contact(
  id:        str,        # stable, "SAT01>GS_SVALBARD@1785542400"  (src>dst@unix_seconds)
  src:       str,        # node id
  dst:       str,        # node id
  t_open:    int,        # absolute Unix ms, UTC
  t_close:   int,
  rate_bps:  int,
  kind:      str,        # "isl" | "downlink"
  max_elev:  float,      # degrees, peak elevation — drives the 1/sin(el) profile
)
```
Contacts are **unidirectional**; emit both directions. Contacts on the same node pair must not overlap — merge if the generator produces overlaps.

### 4.2 `FaultEvent` — output of `faults.py`

```python
FaultEvent(
  t_start:  int,
  t_end:    int,
  target:   str,         # node id, or ground-station id
  kind:     str,         # "weather" | "node_down" | "pointing" | "buffer" | "stale_sched"
  severity: float,       # 0..1, fraction of capacity lost
  shift_ms: int,         # nonzero only for stale_sched
)
```
**Ground truth. Consumed by `engine.py` to decide what actually happens, and by `metrics` for scoring. Never by an advisor.**

### 4.3 `Observation` — what a node actually sees

This is the most important type in the system. Everything the diagnosis layer knows arrives through it.

```python
Observation(
  t:              int,
  node:           str,       # the observing node
  peer:           str,
  contact_id:     str,
  outcome:        str,       # "ok" | "silent" | "degraded" | "late"
  measured_rate:  int,       # 0 if silent
  expected_rate:  int,
  elevation_deg:  float,
  channel:        str,       # "primary" | "beacon"  — two-channel discriminator
  queue_bytes:    int,       # local queue depth toward this peer
  ms_since_ok:    int,       # elapsed silence on this link
  shift_observed: int,       # ms offset if carrier detected outside window
  peer_degree:    int,       # concurrent contacts this peer has right now
  self_degree:    int,       # concurrent contacts we have right now
)
```

`peer_degree` and `self_degree` are what make the diagnosability condition observable to the node itself, not just to the offline analysis.

### 4.4 `Policy` — what an advisor returns

```python
Policy(
  belief:     dict[str, float],   # 6 causes → probability, sums to 1
  action:     str,                # "none"|"wait"|"reroute"|"throttle"|"blacklist"
  target:     str,                # peer the action applies to
  until:      int,                # ms; action expiry
  confidence: float,
  rationale:  str,                # one line, may be empty for non-LLM advisors
)
```

### 4.5 `Evidence` — what crosses the link between nodes

```python
Evidence(
  origin:   str,     # who observed it (NOT who forwarded it)
  seq:      int,     # per-origin counter — the dedup key
  t:        int,     # when observed
  peer:     str,
  outcome:  str,     # quantized from Observation.outcome
  degree:   int,
)
```

**Gossip evidence, not beliefs.** Forwarding posteriors double-counts: A's opinion reaches C directly and via B, C treats it as two independent confirmations, and the network converges confidently wrong. Shipping raw observations with an `(origin, seq)` dedup set avoids the problem entirely rather than correcting for it.

Size budget: ~24 bytes packed. Count these in any bandwidth figure.

### 4.6 `Event` — trace records for the frontend

```jsonc
{"t": 1738400, "type": "contact_open",  "id": "...", "src": "...", "dst": "..."}
{"t": 1738400, "type": "contact_fail",  "id": "...", "mode": "silent"}
{"t": 1738400, "type": "bundle_tx",     "id": "...", "contact": "...", "bytes": 1024}
{"t": 1738400, "type": "belief",        "node": "SAT03", "dist": [0.1,0.6,...]}
{"t": 1738400, "type": "policy",        "node": "SAT03", "action": "wait", "target": "..."}
{"t": 1738400, "type": "gossip",        "from": "SAT03", "to": "SAT07", "n": 4}
```

Beliefs are emitted on a **60 s grid**, not per-event — the frontend interpolates. Everything else is exact.

### 4.7 `Strip` — output of `diagnosability.py`

```python
Strip(
  t_grid:      list[int],
  per_link:    dict[str, list[bool]],          # link id → identifiable at each t
  per_pair:    dict[tuple[str,str], list[bool]] # cause pair → separable at each t
)
```

Computed from `ContactPlan` alone. No simulator, no LLM, no faults. Runs in under a second.

---

## 5. Component specifications

### 5.1 `contacts.py` — contact plan generation

**Input:** `data/iridium.tle`, `data/ground_stations.json`, epoch, horizon.
**Output:** `contact_plan.jsonl`.

- **Library:** Skyfield ≥1.54. Assert `from sgp4.api import accelerated` is true — the pure-Python fallback is ~100× slower.
- **Constellation:** ~12 Iridium NEXT satellites drawn from **two or three adjacent orbital planes** — not spread evenly across all six. See the topology note below; this choice is load-bearing.
- **Sat→ground:** `sat.find_events(gs, t0, t1, altitude_degrees=10.0)`. Skyfield can emit culminate-before-rise and returns wrong set times above 45°. **Pair rise→set with a state machine and discard unpaired events** — do not assume triplets.
- **Sat→sat:** no helper exists. Vectorized grazing-ray test against a sphere of radius `6378.137 + 80` km, AND `range < ISL_MAX_KM`. Six lines of numpy.
- **Elevation:** `(sat - gs).at(t).altaz()`. Vector subtraction — never `.observe()`.
- **Rates:** constants per link kind, scaled by `clip(sin(elevation), 0.3, 1.0)`.
- Full 48 h plan regenerates in under a second. Regenerate wholesale; never patch incrementally.

**Topology — sample by plane, not evenly. This is the single most consequential configuration choice.**

Two requirements pull apart: gossip needs connectivity to propagate, identifiability needs sparsity to have structure. Sampling evenly across all six planes fails the first requirement — measured at 15 satellites spread that way: mean ISL degree **1.41**, and **68.6% of pairs never connect at all**. At 6–8 satellites sampled the same way the constellation is near-disconnected and gossip has no path to travel, which kills the propagation visual outright.

The fix is *which* satellites, not how many. **Draw from two or three adjacent planes.** Iridium's in-plane spacing is ~4090 km, inside the 4500 km crosslink range, so in-plane neighbours are connected almost continuously — a reliable chain for evidence to walk. Cross-plane links open and close, preserving intermittency.

This separates the two jobs cleanly:

| Layer | Role | Character |
|---|---|---|
| **In-plane ISL chain** | The gossip highway | Near-continuous, always has a path |
| **Cross-plane ISLs** | Intermittent structure | Open and close on orbital geometry |
| **Ground-station downlinks** | Where faults live and where identifiability bites | Sparse, scheduled, frequently degree-1 |

**Verify in hour 1** by sweeping `{8, 12, 16} sats × {2, 3} planes × {3000, 4500, 6000} km ISL` and computing the strip for each. Accept a configuration only if the strip shows visible red structure on the downlinks **and** a gossip path reaches most nodes within the scenario window. Ten-minute loop. **LOCKED: 8 satellites, 2 planes, 4500 km, 5 operational stations** (C-007, C-010). The sweep is complete; do not re-run it.

### 5.2 `diagnosability.py` — the separability condition

Two causes are distinguishable only if they predict different observations. Weather at ground station `g` degrades every contact touching `g`. Pointing error at satellite `s` degrades every contact touching `s`. If during the fault window `s` has exactly one active contact and it is with `g`, and `g` has exactly one active contact and it is with `s`, then both hypotheses predict **the identical observation set** — the posterior ratio equals the prior ratio forever, for any estimator.

```
identifiable(s, g, t)  ⟺  deg(g, t) > 1  ∨  deg(s, t) > 1
```

where `deg(n, t)` is the number of contacts involving `n` open at time `t`.

Per-pair conditions:

| Cause pair | Separable when |
|---|---|
| weather ↔ pointing | the concurrency condition above |
| weather ↔ node_down | concurrency, plus elapsed-duration prior |
| node_down transient ↔ permanent | **never at a single t** — only elapsed silence separates them |
| buffer ↔ pointing | requires traffic variation or an active throttle probe |
| stale_sched ↔ anything | requires out-of-window listening enabled |

~20 lines over the contact plan. **Build this before the simulator.** It has no dependencies, it validates the topology, and it tells you whether the rest of the project can work at all.

### 5.3 `faults.py` — injection

Five causes plus NOMINAL. Deterministic from `(ScenarioConfig, seed)`, materialized to a file before the run so every advisor consumes byte-identical faults.

**Deep-space reskin (C-011): the code enum names are unchanged — only their presentation changes.** The physics table below is LEO-authored and still governs the simulation exactly; the map here is what the UI and the pitch call each cause. Because signatures are preserved, the entire diagnosis story (including the demo beat in §6.6) carries over unmodified.

| Code enum (unchanged) | Deep-space label | What it is on Mars |
|---|---|---|
| `weather` | **DUST STORM** | Regional/global dust obscuration at a surface site; degrades every orbiter pass over that site, low-elevation edges first; clears over hours–days. Per-*site*, like weather is per-station |
| `pointing` | **ANTENNA MISPOINT** | Orbiter high-gain antenna mispoint or star-tracker drift; per-orbiter, primary channel only — the low-gain beacon stays nominal |
| `node_down` | **SAFE-MODE / BUS FAULT** | Orbiter drops to safe mode or loses its bus; every link including the beacon goes silent; absorbing until commanded recovery |
| `buffer` | **RECORDER FULL** | Onboard solid-state recorder saturates; correlates with offered traffic; clears immediately under throttle |
| `stale_sched` | **STALE SEQUENCE / EPHEMERIS DRIFT** | A stale uplinked sequence or drifted ephemeris shifts the predicted pass; the contact is shifted, not absent |

"Ground station" everywhere below reads as **Mars surface site** (rover / lander / candidate landing site); "downlink" reads as **orbiter↔surface proximity link**; "ISL / crosslink" is unchanged (orbiter↔orbiter, the gossip highway). The demo fault stays `weather` — now a **dust storm** — for exactly the reasons in §6.6: an orbiter that fails a surface pass cannot tell dust-at-the-site from its-own-mispoint until it meets a second orbiter that also failed at that site. That is genuine distributed inference, not independent detection.

| Kind | Model | Distinguishing signature |
|---|---|---|
| `weather` | ITU-R P.618 chain; fade duration lognormal, median ≈5 min; slope ≤0.5 dB/s | Hits **all satellites at that ground station**. Clips low-elevation edges first (path ∝ 1/sin el). Frequency-dependent |
| `node_down` | Total, binary. Permanent, or transient (watchdog seconds–minutes, reboot ~1 day, safe mode hours–days) | Hits **every link of that node**, including its beacon |
| `pointing` | `L_dB ≈ 12(θ/θ₃dB)²`; total only when θ ≳ θ₃dB | Hits **all peers of that satellite**, primary channel only — **beacon stays nominal** |
| `buffer` | Fractional loss keyed to queue occupancy | **Correlates with offered traffic.** Clears immediately under throttle. Physical layer clean |
| `stale_sched` | Contact opens at `t + shift_ms`, shift 1–5 min | Contact is **shifted, not absent**. Same shift across all that node's contacts. Requires out-of-window listening |

**The whole point is that all five produce the same immediate observable — a contact that doesn't open.** Separation comes from the cross-sectional and temporal pattern, never from a single observation.

### 5.4 `advisor/` — the swap point

```python
class Advisor(Protocol):
    def observe(self, obs: Observation) -> None: ...
    def receive(self, ev: list[Evidence]) -> None: ...
    def decide(self, node: str, t: int) -> Policy: ...
```

Three implementations behind one interface. `NullAdvisor` must run the **identical code path** — same observation construction, same policy enforcer, empty policy — or the comparison measures more than the advisor.

**`bayes.py`** — discrete filter over 6 states, per link. ~80 lines of numpy.

Transition kernel is action-conditioned, and the per-state dynamics are the point:

| State | Dynamics |
|---|---|
| `NOMINAL` | absorbing until a fault begins |
| `weather` | self-healing, geometric dwell, `p_stay ≈ 0.85` |
| `node_down` | **absorbing** — no action escapes it |
| `pointing` | semi-absorbing; exits only under a corrective action |
| `buffer` | exits under `throttle`, persists under `wait` |
| `stale_sched` | absorbing until a schedule refresh |

Factored emission `p(o|s) = Π p(o_k|s)` over the `Observation` fields. Recursion: `b⁻ = Tᵀb`; `b ∝ b⁻ ⊙ L(o)`. Action selection: `argmin_a Σ_s b(s)·C[s][a]` against an explicit cost matrix.

This is a complete working diagnosis system with no LLM involved. **Build it before the Gemma path** — it de-risks everything downstream and it is the number the LLM has to justify itself against.

**`gemma.py`** — belief extraction via single-token logprobs.

A grammar cannot enforce that five floats sum to 1. Do not ask the model to write a probability vector; it will emit valid JSON with broken semantics.

```
prompt ends:  "...Most likely cause (A-F):"
grammar:      root ::= [A-F]
n_predict:    1
n_probs:      6
```

Read the six token logprobs, softmax over exactly those tokens → a vector guaranteed to lie on the simplex. Python runs the Bayes recursion against the prior. Optionally a second call for `action` + `rationale` under a JSON schema; deriving the action in Python from the posterior is equally valid and halves the call count.

Serving: **one `llama-server` process with N slots**, not N processes. Each node is a prompt plus Python-side state.

```bash
llama-server -m gemma-3-4b-it-q4_0.gguf -ngl 99 -c 32768 -np 8 -cb --cache-reuse 256 --temp 0 --top-k 1 --seed 42 --host 127.0.0.1 --port 8080
```

Verify logprobs are exposed before building on them — use the native `/completion` endpoint, not the OpenAI-compatible one:

```bash
curl -s http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d "{\"prompt\":\"Most likely cause (A-F):\",\"grammar\":\"root ::= [A-F]\",\"n_predict\":1,\"n_probs\":6,\"temperature\":0}"
```

Pass condition: a `completion_probabilities` array with six `{tok_str, prob}` entries.

**`llama_client.py`** — every call keyed by `sha256(model_hash ‖ prompt_template_version ‖ canonical_json(observation_window) ‖ decode_params)` into SQLite. Temperature 0 is *not* determinism — CUDA reduction kernels are not batch-invariant, so identical prompts batched differently can yield different tokens. The cache is the determinism mechanism. In replay mode, **raise on cache miss** rather than silently calling live.

### 5.5 `gossip.py`

On each successful contact, exchange `Evidence` tuples not already in the peer's seen-set, newest first, up to a byte budget. Dedup by `(origin, seq)`.

Apply an **age discount** `λ^Δt` that depends on the cause: stale evidence about self-healing weather is nearly worthless, stale evidence about an absorbing node failure is nearly as good as fresh. This asymmetry is the interesting part of doing inference over a delay-tolerant link.

### 5.6 `router.py` and `policy.py`

**First-contact forwarding:** send on the next available contact toward the destination. ~30 lines. A standard DTN baseline.

Full SABR (CCSDS 734.3-B-1, earliest-arrival Dijkstra over a contact graph) is specified in the reference notes but is **not on the build path** — the two-pass search/forwarding structure is a multi-hour item and the project's claim does not depend on routing quality.

`policy.py` applies the advisor's action as an overlay, never by mutating the contact plan:

| Action | Effect |
|---|---|
| `wait` | hold bundle in queue; do not re-plan |
| `reroute` | exclude this peer for this destination until `until` |
| `throttle` | reduce offered rate to this peer by a factor |
| `blacklist` | exclude this peer entirely until `until` |

---

## 6. Screens

**One screen.** No navigation, no tabs that change anything, no routing. Everything visible at once, floating over a live map.

The visual register is specified in full in `UI-SPEC.md` — tactical command display, six colour tokens, 1px hairlines, condensed uppercase type, nothing rounded, nothing glowing, translucent panels over the map. **That document is authoritative for appearance.** This section records only decisions, conflicts resolved, and what the UI spec omits.

### 6.1 Component map

| ID | Component | Carries |
|---|---|---|
| **[E]** | Roster panel, left | **All-assets overview** — every satellite, its state LED, its designation |
| **[F]** | Map field | Wireframe globe, asset chips, contact arcs, gossip pulses |
| **[G]** | Alert box, top-right | Current fault event, in `//` prefixed lines |
| **[J]** | Warning strip | Appears only while a fault is unresolved |
| **[O]** | Feed inset, bottom-right | **Belief breakdown for the selected asset** — replaces the click-to-inspect panel |
| **[P]** | *Timeline + diagnosability strip, bottom* | **Not in the UI spec — see §6.5** |
| [A][B][C][D][H][I][K][L][M][N] | Chrome | Aesthetic. No data dependency. Cut from the middle if time runs short |

**The roster is the "overview of all devices" screen.** Twelve rows at 34px is 408px — fits the left column comfortably, and every asset's diagnosis state is visible simultaneously without clicking anything.

### 6.2 Globe — textured, cropped, and the renderer choice

**[REVISED after seeing the reference image.]** My earlier call was d3-geo on the grounds that a bare wireframe is a 2D projection problem. The reference shows that was wrong in one respect: **the map is a photographic satellite basemap bleeding off every edge**, and a large share of the composition's density and mid-tone range comes from it. A wireframe on flat black loses that, and our view is global where the reference is local — so a complete circle centred in frame would leave black margins the reference never has.

**Two requirements, independent of renderer:**

1. **Texture the sphere** with desaturated satellite imagery (Blue Marble, monochrome, 10–25% luminance), graticule and coastlines drawn *over* it.
2. **Crop, don't fit.** Overflow the viewport on all edges. Never a complete circle floating in black — this single change matters more than any other for matching the reference.

**Renderer: Three.js. Decided — d3-geo is out, do not revisit.**

`SphereGeometry` + `MeshBasicMaterial` (ignores lighting, so zero material work) + a texture, ~30 lines. Satellite screen positions via `vector.project(camera)`. Occlusion is the depth buffer, free. The d3-geo alternative needed a per-pixel raster reprojection loop, which is the one genuinely fiddly part of this build, to save a dependency that costs nothing.

**`satellite.js` is also out.** Positions are baked into the trace by the simulator on the 60 s grid and interpolated in the browser — 8 satellites × 1441 frames is trivial, and it removes TLE parsing, SGP4, and a whole class of "why is the orbit wrong" debugging from demo day. The Python side already knows exactly where everything is.

Iridium at 780 km ⇒ render radius ×1.12.


### 6.3 Colour semantics — the one real conflict, and its resolution

**The palette has two semantic colours. The taxonomy has six causes. These cannot be reconciled by adding colours** — doing so breaks the spec's own governing rule that colour must carry meaning, and six pastel cause-colours would be decoration.

**Encode *epistemic state* in colour. Encode *cause* in text.**

| Node state | Rendering | Meaning |
|---|---|---|
| `NOMINAL` | `--ink-dim`, steady | Nothing wrong |
| `UNCERTAIN` | **alternates `--ink` / `--alert` on a hard 400 ms interval, no crossfade** | Belief is split across causes |
| `RESOLVED` | `--accent` teal, steady | Confident diagnosis, action taken |
| `FAULT` | `--alert` red, steady | Confirmed fault on this node |
| `STALE` | `--neutral` cream, steady | Belief resting only on aged gossip |

The cause itself appears as the state word beneath the asset chip — `NOMINAL` / `DEGRADED` / `SILENT` / `STATION WX` / `BUFFER` / `SCHED` — and in full in the feed inset [O].

**This improves the demo rather than compromising it.** The money shot is no longer *colour A becomes colour B*, which is arbitrary and needs explaining. It is **flickering becomes steady** — a node visibly not knowing, then knowing. That reads instantly, with no legend and no narration, and the spread across the constellation becomes a wave of flickers resolving one after another.

The flicker is already identified in the UI spec as "the single most important animation in the interface." It is. Build it first and tune it before anything else.

**Consequence:** the earlier five-colour cause palette is dead. Do not carry it forward.

### 6.4 Markers and arcs

Per the UI spec: asset chip (white rectangle, 2px radius, 2-digit index) with a 16px circle outline beside it containing a 6px state dot; ground stations as 12px teal triangles; faults as 14px red crosses.

Two additions the spec does not cover:

- **Gossip pulse.** A contact arc carries bundle traffic *and* evidence exchange. These must be distinguishable. Bundle traffic: a filled dot travelling the arc. Gossip: a **1px bracket glyph `⟩`** travelling the arc, `--accent`. Different shape, not different colour.
- **Arc weight on failure.** When a scheduled contact fails to open, draw the arc anyway as a 1px `--ink-faint` **dashed** line for 2 s, then remove it. The absence must be visible — a contact that silently never appears reads as nothing happening.

### 6.5 Timeline and diagnosability strip — missing from the UI spec

The UI spec has no time control. The demo requires one: *scrub back and replay the moment slower* is a core beat.

Add region **[P]**, bottom-centre, full width of the map field, in the spec's visual language:

```
 ▮▮▮▮░░░░▮▮▮▮▮▮▮▮░░░░░░▮▮▮▮▮▮        ← diagnosability, 8px tall
 ├────┴────┴────┴────┴────┴────┤      ← 1px rule, tick marks descending
        ▲                              ← playhead, 1px --accent vertical
 ▶ ⏸   1×  2×  8×          T+04:17:22
```

- **Diagnosability band**, 8px: solid `--ink-dim` where the currently-faulted link is identifiable, 45° diagonal hatch (same treatment as warning strip [J]) where it is not. Hatch already means "caution" in this visual language — reuse it rather than introducing a colour.
- **Rule and ticks** match scale bar [M] exactly: 1px, ticks descending, 7px labels beneath.
- **Playhead** is the only `--accent` element, 1px vertical.
- Transport controls at 9px uppercase; active speed boxed in 1px `--ink`, per tab treatment [B].

### 6.6 Which fault to inject — the demo beat

**Inject `weather`, not `node_down`.** This is the difference between showing distributed inference and showing something that only looks like it.

**Why node death fails.** If a satellite dies, every peer that attempts contact observes silence *directly*. Each diagnoses independently from its own observation. Red spreads across the constellation and looks exactly like the money shot — but nothing was inferred from anyone else and no evidence needed to travel. It is independent detection wearing the costume of distributed inference.

**Why weather works.** A satellite that fails a downlink genuinely cannot distinguish *weather at the ground station* from *its own pointing error* — the observation is byte-identical. It flickers. Then it meets a peer that failed at **the same ground station**. That single fact collapses the ambiguity, because pointing error is per-satellite and weather is per-station. The flicker snaps to steady. Then that node tells the next.

Three things follow:

1. Flicker-then-resolve is genuine inference, visibly distinct from independent detection.
2. It is exactly the case the diagnosability condition governs (§5.2) — the band goes solid at the moment the second contact supplies the concurrency. Theory and picture align on screen without a word.
3. Resolution requires information that physically travelled, which is the entire premise.

| Beat | Fault | Shows |
|---|---|---|
| 1 | `weather` at a ground station | Ambiguity → gossip → convergence. **The mechanism** |
| 2 | `node_down` | Total silence, permanent, blacklist, reroute. **The clean payoff** |

If only one fits: **weather.** Then scrub back and replay it slower.

### 6.7 Counters — CUT

Both arms currently deliver identically (573 bundles, 0 dropped): contact capacity exceeds offered load by orders of magnitude, so a delivered/dropped readout would show a dead heat and actively undercut the demo. Cut rather than left as a gated maybe. Revisit only if traffic sizing lands and the gap is large enough to read from the back of a room. See C-010.

### 6.8 Build order and time budget

The UI spec's own build order is correct. Realistic costs for one person:

| Step | Work | Est |
|---|---|---|
| 1 | Globe: three.js textured sphere, cropped to overflow, rotation | 45 m |
| 2 | Dotted grid, reticle, crosshairs | 25 m |
| 3 | **Roster panel + state LEDs** | 40 m |
| 4 | **Asset chips, arcs, flicker, gossip pulses** | 60 m |
| **P** | **Timeline + diagnosability band** | 30 m |
| 5 | Frame, corner brackets, crop marks | 20 m |
| 6 | Right clusters: alert, coords, data table, warning strip | 45 m |
| 7 | Bottom clusters: nav, scale, link status, feed inset | 45 m |
| 8 | Noise + scanlines | 10 m |

**Steps 1–4 plus P are the demo: ~3 h 20 m.** Steps 5–8 are the aesthetic: ~2 h.

**Hard cut line:** if step 4 is not finished, stop building chrome. Steps 5 and 8 together are 30 minutes and deliver most of the visual identity — corner brackets, crop marks, noise, and scanlines are disproportionately responsible for the look. **Cut 6 and 7 before cutting 5 or 8.** The nav cluster [L], mode select [K], and data table [I] are pure decoration and can be dropped entirely without anyone noticing.

### 6.9 Projector safety — resolve before demo day

The palette is tuned for a laptop panel. Projectors crush dark values and lose low-alpha detail.

- `--ink-faint` at `0.18` on `#0A0B0D` will likely **vanish**. The graticule and dotted grid go with it.
- The noise overlay at `0.04` will definitely vanish. Harmless.
- Reticle radii are specified in absolute px (230 / 85). On a 1280×720 projector versus a 1920×1080 panel these land differently. **Compute from `min(vw, vh)`, not fixed px.**

**Make this a one-line fix:** keep all six values as CSS custom properties on `:root` and add a `.projector` class that overrides `--ink-faint` to `0.30` and `--ink-dim` to `0.58`. Test on the actual projector, toggle the class, move on. Do not rebalance the palette by hand at 16:40.

### 6.10 Stack

```bash
npm create vite@latest web -- --template react-ts
```
```bash
npm i three zustand && npm i -D tailwindcss@4 @tailwindcss/vite@4
```

Tailwind v4 needs no config file. Keep the project off OneDrive and out of paths containing spaces.

**Rendering split:** everything that moves every frame — globe, arcs, chips, pulses — draws imperatively to a single `<canvas>`. Everything static or event-driven — roster, panels, frame, crop marks — is React DOM. **The rAF loop writes time to a ref or a zustand transient subscription, never React state**, or the tree re-renders 60×/s.

Font: `Roboto Condensed` or `Barlow Condensed`. **Download and self-host the woff2 before the event** — a Google Fonts CDN call is a runtime network dependency and the whole page falls back to the wrong metrics without it.

## 7. Requirements

### 7.1 Functional

| # | Requirement | Verified by |
|---|---|---|
| F1 | Generate a contact plan from real TLEs and real ground-station coordinates | Plan contains >100 contacts over 24 h; durations 5–12 min |
| F2 | Compute the diagnosability strip from the contact plan alone | Runs with no simulator, no faults, no LLM; <1 s |
| F3 | Inject five fault kinds, deterministically from a seed | Same seed → byte-identical `fault_trace.jsonl` |
| F4 | All five faults produce the same immediate observable | `Observation.outcome` distribution is identical across kinds at first occurrence |
| F5 | Each node maintains a normalized belief over 6 causes | `sum(belief) == 1.0 ± 1e-6` at every emission |
| F6 | Belief updates from local observation without any peer contact | Single-node scenario still converges on `node_down` |
| F7 | Evidence propagates between nodes on successful contact only | No belief change at a node between its contacts, absent local observation |
| F8 | Gossiped evidence is deduplicated by origin | Same `(origin, seq)` received twice does not change the posterior |
| F9 | Each node emits an action matched to its belief | Action equals `argmin` of expected cost under the cost matrix |
| F10 | Actions change routing behaviour | `blacklist` removes the peer from forwarding candidates |
| F11 | A run replays in the browser from a static trace | No network calls after initial load |
| F12 | Advisors are swappable without touching the simulator | `--advisor null\|bayes\|gemma` produces three runs from one code path |

### 7.2 Data

| # | Requirement | Source |
|---|---|---|
| D1 | TLE set, committed, never fetched at runtime | CelesTrak `gp.php?GROUP=iridium-NEXT&FORMAT=tle` — **pass `FORMAT` explicitly**, it defaults to CSV |
| D2 | 10–15 ground stations with real coordinates | Hand-authored JSON. §9 |
| D3 | Gemma 3 4B QAT Q4_0 GGUF, ~3.3 GB, downloaded in advance | HuggingFace `google/gemma-3-4b-it-qat-q4_0-gguf` |
| D4 | llama.cpp CUDA binaries + matching `cudart` | Match the CUDA version to the binary or it fails at startup |
| D5 | Skyfield timescale cache pre-warmed | Avoids a runtime download |

### 7.3 Technical

| # | Requirement | Note |
|---|---|---|
| T1 | A run is fully reproducible from `(ScenarioConfig, seed)` | Named RNG streams per subsystem: `traffic`, `faults`, `channel`, `advisor`, `gossip` |
| T2 | Traffic and fault traces are materialized before the run | Otherwise advisor RNG consumption shifts arrival times and runs stop being comparable |
| T3 | Every LLM call is cached; replay performs zero inference | Cache miss in replay mode is an error, not a silent live call |
| T4 | Simulation time is integer milliseconds | Never float — accumulated error is a determinism bug |
| T5 | No iteration over runtime-ordered sets or dicts | Sort by node id |
| T6 | The advisor holds no reference to the simulator or fault trace | Enforce with a test |
| T7 | Belief frames downsampled to 60 s for the trace | Full resolution is ~78 MB; downsampled is ~1.3 MB |
| T8 | Trace loads in one fetch | Target <3 MB gzipped |

---

## 8. Development stages

Dependency-ordered. Stages 1 and 6 have no dependencies on each other and run in parallel from the start.

```
S0 ─┬─ S1 ── S2 ── S3 ── S4 ── S5 ─┐
    │                              ├─ S8
    └─ S6 ── S7 ────────────────────┘
```

### S0 — Contracts

Freeze §4's six types. Write a hand-authored fake `trace.json` with two nodes, one fault, and three belief frames.

**Exit:** the frontend can render something from the fake trace, and the simulator has a target format. **Nothing else starts until this is done** — the trace schema is the only real integration risk in the project.

### S1 — Contact plan + diagnosability

`contacts.py`, `diagnosability.py`. Run the topology sweep from §5.1 and lock the configuration.

**Exit:** `contact_plan.jsonl` and `strip.json` exist; the strip has visible red structure. **This stage alone produces a result even if everything after it fails.**

### S2 — Faults and observation

`faults.py`, `observe.py`. Emit real traces with no diagnosis attached.

**Exit:** F3 and F4 pass — same seed reproduces, and all five faults look identical at first occurrence.

### S3 — Bayes advisor

`advisor/base.py`, `advisor/null.py`, `advisor/bayes.py`, `policy.py`, `router.py`, `engine.py`.

**Exit:** a complete working diagnosis system with no LLM. F5, F6, F9, F10 pass. Confusion matrix is diagonal-dominant in identifiable windows and visibly confused in unidentifiable ones — **this is the check that the whole premise holds.**

### S4 — Gemma advisor

`advisor/gemma.py`, `llama_client.py`. Verify logprob exposure *before* writing the advisor.

**Exit:** F12 passes — three advisors, one code path. Cache populated; replay does zero inference.

### S5 — Gossip

`gossip.py`. Evidence tuples, seen-set dedup, age discount.

**Exit:** F7 and F8 pass. Belief visibly converges across nodes after contact.

### S6 — Map field *(parallel with S1–S3)*

`UI-SPEC.md` build steps 1–2 plus asset chips: textured sphere overflowing the viewport, graticule and coastlines over it, dotted grid, reticle, crosshairs, ground-station triangles, contact arcs opening and closing, bundle pulses.

**Exit:** the network breathes, driven by TLEs alone with no simulator dependency.

### S7 — State visuals *(parallel)*

Roster panel [E] with status LEDs, **the flicker** (build and tune this first — it is the demo), gossip bracket-glyph pulses, timeline + diagnosability band [P]. Then chrome per `UI-SPEC.md` steps 5 and 8. Feed inset [O] last.

**Exit:** the fake trace from S0 drives a complete-looking screen.

### S8 — Integration

Point the frontend at real output. Tune fault timing so the interesting moment is early and obvious. Add a "jump to fault" control.

**Exit:** one run, one screen, end to end.

### Cut order, if time runs short

1. **Chrome clusters [I][K][L] and feed inset [O]** — pure decoration plus a question-and-answer asset nobody clicks in two minutes. **Keep the frame, brackets, crop marks, and noise** (`UI-SPEC.md` steps 5 and 8, 30 min total) — they carry most of the visual identity
2. **Delivered/dropped counters** — gated on the numbers being convincing anyway (§6.7)
3. **Gemma (S4)** — the Bayes advisor is a complete working diagnosis system; the halos look identical
4. **Diagnosability strip** — cheap to build and nothing else replaces it, so this is a reluctant cut

**Never cut:**

- **S1** — the only stage with no dependencies and the only one that stands alone.
- **S5 (gossip)** — *changed from an earlier draft.* Gossip was listed as first-to-cut on the reasoning that a single node diagnosing alone still shows the mechanism. That is no longer true: §6.8 selects `weather` as the demo fault precisely because its resolution **requires** evidence that physically travelled between nodes. Without gossip, the weather case never resolves and the demo has no payoff. Gossip is now load-bearing.

---

## 9. Reference data

### Ground stations

| Name | Lat | Lon | Alt (m) |
|---|---|---|---|
| Svalbard (SvalSat) | 78.2297 | 15.4075 | 458 |
| Fairbanks (Poker Flat) | 64.8590 | −147.8490 | 320 |
| Wallops Island | 37.9400 | −75.4600 | 12 |
| Punta Arenas | −52.9390 | −70.8510 | 35 |
| Awarua, NZ | −46.5290 | 168.3800 | 3 |

Add 5–10 more by hand from ESA Estrack and KSAT published station lists for visual density on the globe.

**Make Svalbard and Fairbanks killable.** Measured: all five stations → 532 contacts/24 h, median duration 9.5 min, median gap 31 min. Remove the polar pair → 198 contacts, median gap 67 min, p90 270 min, max 392 min. That 3× gap inflation is the best node-failure scenario in the configuration.

### Link rates

| Link | Rate |
|---|---|
| Ka-band ISL (Iridium-class) | 25 Mbps |
| X-band downlink | 160 Mbps |
| S-band beacon | 2 Mbps |

### Traffic

Poisson arrivals per source, many-to-one toward a ground sink. Bundles 1 kB fixed. TTL 3–6 h so it spans several contact opportunities. **Size buffers so the fault-free case drops ≈0 and the faulted case drops >0** — otherwise faults are invisible in the metrics.

### Fault physics constants

| Fault | Constants |
|---|---|
| weather | fade duration lognormal, median ≈5 min, range 10 s–5000 s; slope ≤0.5 dB/s; rain cells 5–10 km, decorrelated ≥100 km; Ka ≫ X by ~f^2.3 |
| node_down | watchdog seconds–minutes; SEU reboot ~1 day; safe mode hours–days; eclipse brownout periodic at ~90 min |
| pointing | `L_dB ≈ 12(θ/θ₃dB)²`; θ₃dB ≈ 5.3° at X-band for a 0.5 m dish; 5° error ⇒ 12 dB ⇒ link dead |
| buffer | loss fraction monotonic in queue occupancy; clears under throttle |
| stale_sched | shift 1–5 min; same sign and magnitude across all of that node's contacts; grows with plan age |

---

## 10. Changes

Amendments to this plan made after implementation began. Newest first.

A change belongs here when the *plan* was wrong or underspecified — not when
code merely got written. Discoveries live in `FINDINGS.md`; this section
records what the plan now says differently as a result.

| # | Section | Change | Driver |
|---|---|---|---|
| C-011 | §1 / §6 | **About to reskin from LEO to deep space.** The scenario framing and visual register are moving from a low-Earth-orbit satellite network to a deep-space setting. Pending — records the intent; section-level rewrites (globe/basemap, framing copy, topology assumptions) follow. | design |
| C-010 | §9 / §6.4 | **Operational stations reduced to 5** (the file keeps 12 for globe density, seven rendered dimmed as other networks). Twelve operational gave 47% per-satellite coverage and a 17-minute median gap — not a delay-tolerant network, and the "you cannot ask anyone" premise silently stopped being true. Five gives 21% coverage, 31.7 min median gap, p90 80 min, **and** the best strip available: 100% of downlinks mixed, identifiability 0.521. **Also: the delivered/dropped counter must stay off screen until traffic is sized** — both arms currently deliver identically, so it would show a dead heat. | FINDINGS F-017 |
| C-009 | §5.2 | **The invariant applies to the likelihood, not the posterior.** §5.2 says the ratio between two unidentifiable causes "equals the prior ratio for all t and all observations". That is right for the observation likelihood and wrong for the posterior: the transition kernel encodes genuinely different dwell times (weather ~5 min, pointing persistent), so a fault still running after ten observations really is more likely pointing. Forbidding the drift would forbid the model from knowing weather is brief. Measured residual drift ~3.3× over ten observations, from dynamics alone — the instantaneous ratio is exactly 1.0. | FINDINGS F-016 |
| C-008 | §7 F4 | **F4 was overstated and contradicted §6.** It required all five causes to produce the same observable; §6's separability table says `stale_sched` produces a *shifted* contact as its signature. §6 is right. F4 now applies to the four **absence-producing** causes (weather, node_down, pointing, buffer). `stale_sched` is **conditionally** identifiable — `LATE` inside the listening window, `SILENT` and fully confusable outside it — and is reported separately. Verified over 400 real contacts at generator-drawn severities: genuine overlap among all four, no cause nameable from one observation. | FINDINGS F-010 |
| C-007 | §5.1 | **Topology locked: `n_sats=8, n_planes=2, isl_max_km=4500`.** Was 12 satellites. Eight now wins on every measured axis — gossip reach 1.000/1.000, downlink identifiability 0.6247 (closest to mid-band), 92.7% of links mixed (highest of 8/12/16), and least globe clutter. `isl_max_km=4500` is the smallest value clearing the gossip gate and clears it completely; 6000 km is no better on any gossip metric and only adds churn. Sweep: 27 configs, cross-checked against an independent Dijkstra earliest-arrival, zero mismatches. | FINDINGS F-008 |
| C-006 | §5.1 | Satellites must be ordered by **true argument of latitude at the scenario epoch**, never by `model.mo`. Mean anomaly is stated at each satellite's own TLE epoch and CelesTrak staggers those ~9 min apart, so `mo` sorts by epoch, not orbital position. | FINDINGS F-008 |
| C-005 | §5.1 | **The topology tradeoff does not exist on the `isl_max_km` axis.** Measured 0.6149 downlink identifiability at 4500 / 6000 / 8000 km — identical to four decimals. Once degree is scoped by link kind (C-002), a downlink's identifiability depends only on other *downlinks*, so crosslink range cannot touch it. `isl_max_km` is a pure gossip-propagation knob; `n_sats` / `n_planes` / station count are what move the strip. Sweep these as two independent one-dimensional questions, not a joint 27-config ranking. | FINDINGS F-006 |
| C-004 | §4 | **All times are absolute Unix epoch milliseconds (UTC)**, not offsets from the scenario epoch. Frontend uses `new Date(t)` directly with no epoch to ship alongside. Contact id format corrected to `src>dst@unix_seconds`. | test suite |
| C-003 | §6.5 | Region **[P]** must render the strip for **the currently faulted link**, never a network-wide aggregate. "Is any link identifiable right now" is true at essentially every instant and renders as a solid green bar carrying no information. | FINDINGS F-002 |
| C-002 | §5.2 | The identifiability condition is scoped **by link kind**, not computed over all contacts. `deg` counts only contacts of the same kind as the link being evaluated. The condition is not about concurrency but about **concurrency on a shared failure mode** — a healthy crosslink cannot discriminate weather-at-station from pointing-error-at-satellite, because it uses different hardware and survives either. When the fault model grows a second channel (primary vs beacon), degree must be scoped to the affected channel too. | FINDINGS F-001 |
| C-001 | §5.1 | Topology-sweep metrics must be **window-relative** — identifiable grid points divided by points where the link is *open*, not by total grid points. Dividing by total time measures duty cycle and reports ~0.04 for any downlink regardless of how separable it is. | FINDINGS F-002 |

### Still open against the plan

- **The demo does not exist.** The simulator is at 125 tests and the frontend is at zero files. That asymmetry is the only thing that matters now: a two-minute demo is judged on what is on screen, and none of it is. Everything below is subordinate to fixing that.
- **§6.7 counters are cut** (C-010). Traffic sizing is out for measurement but is not on the demo critical path.
- **The identifiability result is directionally right but statistically unsupported** — 0.778 identifiable vs 0.577 unidentifiable on the weather/pointing pair, but n=9 in the identifiable bin. A power run is out. Do not quote the number until it comes back with a CI.
- **13 tracked defects in the Bayes filter** (`BAYES_STRICT=1` to run them). Highest value: `elevation_deg` unused (fade ∝ 1/sin el would separate weather from pointing *within one link*, so it works at degree 1), and `peer_degree`/`self_degree` unused — literally both halves of the identifiability condition. None block the demo.

### Settled — do not reopen

Topology (8 sats / 2 planes / 4500 km / 5 operational stations, C-007 and C-010) · F4 premise verified over 400 contacts (C-008) · renderer is Three.js with positions baked into the trace (§6.2) · demo fault is weather, not node death (§6.8) · colour encodes epistemic state, not cause (§6.3).
