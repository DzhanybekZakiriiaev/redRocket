/* ────────────────────────────────────────────────────────────────────────
   globe.ts — textured sphere, cropped so it overflows every viewport edge,
   plus an exact lat/lon -> screen projection shared with the 2D overlay.

   MeshBasicMaterial ignores lighting, so there is no material work at all.
   The basemap is rasterised once from world-atlas TopoJSON into a
   desaturated equirectangular canvas at ~10-25% luminance, exactly as a
   Blue Marble crop would read after being pushed to monochrome.
   ──────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three'
import { feature } from 'topojson-client'
import land50 from 'world-atlas/land-50m.json'
import land110 from 'world-atlas/land-110m.json'
import type { Topology, GeometryCollection } from 'topojson-specification'

const R = 1                      // Earth radius in world units
const EARTH_KM = 6371
const ALT_SCALE = 1.25           // 631 km * 1.25 / 6371 -> render radius x1.124
const FOV = 30

/** Sphere apparent radius as a fraction of min(vw,vh).
 *
 *  Raised from 0.33 to give the constellation room, but NOT to the 0.78 that
 *  UI-SPEC §5 [F] asks for. That spec was written for a globe with nothing
 *  orbiting it: the satellites draw at up to 1.19x the sphere radius, so any
 *  CROP above ~0.42 throws them off the top and bottom of the frame entirely.
 *  0.38 is the largest value that keeps all eight assets on screen. */
const CROP = 0.38

/* ── orbital shell separation ────────────────────────────────────────────
   The constellation is physically flat: SAT01 sits at ~631 km and SAT02-08
   share a ~780-808 km shell, so at true scale every orbit collapses onto one
   ring and the satellites read as a single blob.

   So the DEVIATION from the reference shell is exaggerated, not the altitude
   itself. Ordering and relative spacing are preserved — a satellite that is
   genuinely lower still draws lower — but the shells separate enough to be
   told apart. This is a display transform only; nothing reads it back, and
   the altitude quoted in the inspector is the true value from the trace.

   Bounded by the frame: at 3.2 the outer shell lands at 1.19x the sphere
   radius, which with CROP 0.38 sits just inside the viewport edge. Raising
   either constant much further starts pushing assets out of view. */
const ALT_REF_KM = 700
const ALT_SPREAD = 3.2

export type LonLat = [number, number]
export type GlobeMode = 'wireframe' | 'mars' | 'earth'

/* ── lat/lon -> unit vector, matched to three's SphereGeometry UV layout ── */
export function llToVec(latDeg: number, lonDeg: number, r = R): THREE.Vector3 {
  const la = (latDeg * Math.PI) / 180
  const lo = (lonDeg * Math.PI) / 180
  const c = Math.cos(la)
  return new THREE.Vector3(r * c * Math.cos(lo), r * Math.sin(la), -r * c * Math.sin(lo))
}

export const satRadius = (altKm: number) => {
  const eff = Math.max(120, ALT_REF_KM + (altKm - ALT_REF_KM) * ALT_SPREAD)
  return R * (1 + (eff / EARTH_KM) * ALT_SCALE)
}

/* ── basemap raster ─────────────────────────────────────────────────────── */

/** Equirectangular basemap: clean land fills over a dark ocean, no noise. */
function buildBasemap(): HTMLCanvasElement {
  const W = 2048, H = 1024
  const cv = document.createElement('canvas')
  cv.width = W; cv.height = H
  const g = cv.getContext('2d')!

  // ocean — near-black, very slightly blue, matches --bg family
  g.fillStyle = '#0b0e13'
  g.fillRect(0, 0, W, H)

  const geo = feature(
    land50 as unknown as Topology,
    (land50 as unknown as Topology).objects.land as GeometryCollection,
  ) as GeoJSON.FeatureCollection

  const proj = (c: number[]) => [((c[0] + 180) / 360) * W, ((90 - c[1]) / 180) * H]

  const tracePolys = (coords: number[][][][]) => {
    g.beginPath()
    for (const poly of coords) {
      for (const ring of poly) {
        ring.forEach((c, i) => {
          const [x, y] = proj(c)
          if (i === 0) g.moveTo(x, y); else g.lineTo(x, y)
        })
        g.closePath()
      }
    }
  }

  const all: number[][][][] = []
  for (const f of geo.features) {
    const gm = f.geometry
    if (gm.type === 'Polygon') all.push(gm.coordinates as number[][][])
    else if (gm.type === 'MultiPolygon') for (const p of gm.coordinates) all.push(p as number[][][])
  }

  // continental shelf halo — a touch above the ocean floor value
  tracePolys(all)
  g.lineJoin = 'round'
  g.strokeStyle = 'rgba(140,150,165,0.16)'
  g.lineWidth = 9
  g.stroke()

  // landmass fill — mid-dark grey, the mid-tone the composition needs
  tracePolys(all)
  g.fillStyle = '#262a30'
  g.fill('evenodd')

  // coastal rim
  g.strokeStyle = 'rgba(210,218,228,0.22)'
  g.lineWidth = 1.5
  g.stroke()

  return cv
}

/* ── coastline polylines for the 1px vector overlay ─────────────────────── */
export function coastlines(): LonLat[][] {
  const geo = feature(
    land110 as unknown as Topology,
    (land110 as unknown as Topology).objects.land as GeometryCollection,
  ) as GeoJSON.FeatureCollection
  const out: LonLat[][] = []
  for (const f of geo.features) {
    const gm = f.geometry
    const polys: number[][][][] =
      gm.type === 'Polygon' ? [gm.coordinates as number[][][]]
      : gm.type === 'MultiPolygon' ? (gm.coordinates as number[][][][])
      : []
    for (const poly of polys) {
      for (const ring of poly) {
        if (ring.length < 3) continue
        out.push(ring.map(c => [c[0], c[1]] as LonLat))
      }
    }
  }
  return out
}

/* ── Mars basemap ───────────────────────────────────────────────────────── */

/** Equirectangular Mars texture — muted terracotta with darker maria, brighter
 *  ochre plains, and pale polar caps, held into a low-luminance band so it
 *  reads as a dark tactical planet rather than a cartoon red ball. */
export function buildMars(): HTMLCanvasElement {
  const W = 2048, H = 1024
  const cv = document.createElement('canvas')
  cv.width = W; cv.height = H
  const g = cv.getContext('2d')!

  // clean, smooth warm rust — a little deeper toward the poles so the sphere
  // reads as lit, with NO spots, blobs, or caps.
  const base = g.createLinearGradient(0, 0, 0, H)
  base.addColorStop(0, '#5c2e1d')
  base.addColorStop(0.5, '#93502f')
  base.addColorStop(1, '#5c2e1d')
  g.fillStyle = base
  g.fillRect(0, 0, W, H)

  return cv
}

/* ── the renderer ───────────────────────────────────────────────────────── */

export class Globe {
  readonly renderer: THREE.WebGLRenderer
  readonly scene = new THREE.Scene()
  readonly camera: THREE.PerspectiveCamera
  readonly group = new THREE.Group()
  private mesh: THREE.Mesh
  private _mode: GlobeMode = 'wireframe'
  private _w = 1
  private _h = 1
  /** combined projection * view * groupWorld, row-major elements */
  private m = new THREE.Matrix4()
  private camLocal = new THREE.Vector3()

  yaw = 0
  tilt = (-28 * Math.PI) / 180
  zoom = 1
  autoRotate = true
  private _baseDist = 5

  constructor(canvas: HTMLCanvasElement, opts: { mode?: GlobeMode } = {}) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false })
    this.renderer.setClearColor(0x0a0b0d, 1)
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.01, 100)

    this._mode = opts.mode ?? 'wireframe'
    this.mesh = new THREE.Mesh(
      new THREE.SphereGeometry(R, 96, 64),
      this.makeMaterial(this._mode),
    )
    this.group.add(this.mesh)
    this.scene.add(this.group)
  }

  /** Material for a mode. 'mars'/'earth' are textured; 'wireframe' is a solid
   *  near-black body the overlay graticule sits on. */
  private makeMaterial(mode: GlobeMode): THREE.MeshBasicMaterial {
    if (mode === 'wireframe') {
      return new THREE.MeshBasicMaterial({ color: 0x0e1116 })
    }
    const tex = new THREE.CanvasTexture(mode === 'mars' ? buildMars() : buildBasemap())
    tex.colorSpace = THREE.SRGBColorSpace
    tex.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy())
    tex.wrapS = THREE.RepeatWrapping
    return new THREE.MeshBasicMaterial({ map: tex })
  }

  /** Swap the globe body live (wireframe / mars / earth), disposing the old. */
  setMode(mode: GlobeMode) {
    if (mode === this._mode) return
    this._mode = mode
    const old = this.mesh.material as THREE.MeshBasicMaterial
    this.mesh.material = this.makeMaterial(mode)
    old.map?.dispose()
    old.dispose()
  }

  get mode() { return this._mode }

  /** Solve camera distance so the sphere silhouette lands on CROP * max edge. */
  private solveDistance(w: number, h: number): number {
    const target = CROP * Math.min(w, h)
    const tanHalf = Math.tan((FOV * Math.PI) / 360)
    const px = (D: number) => {
      const rc = R * Math.sqrt(Math.max(1e-9, 1 - (R * R) / (D * D)))
      const dc = D - (R * R) / D
      return (rc * (h / 2)) / (dc * tanHalf)
    }
    let lo = R * 1.02, hi = 60
    for (let i = 0; i < 60; i++) {
      const mid = (lo + hi) / 2
      if (px(mid) > target) lo = mid; else hi = mid
    }
    return (lo + hi) / 2
  }

  resize(w: number, h: number, dpr: number) {
    this._w = w; this._h = h
    this.renderer.setPixelRatio(dpr)
    this.renderer.setSize(w, h, false)
    this.camera.aspect = w / h
    this._baseDist = this.solveDistance(w, h)
    this.applyCamera()
  }

  private applyCamera() {
    this.camera.position.set(0, 0, this._baseDist / this.zoom)
    this.camera.lookAt(0, 0, 0)
    this.camera.updateProjectionMatrix()
  }

  /** Drag to spin (yaw + tilt). Stops the idle auto-rotation. */
  rotateBy(dxPx: number, dyPx: number) {
    this.autoRotate = false
    this.yaw += dxPx * 0.005
    this.tilt = Math.max(-1.3, Math.min(1.3, this.tilt + dyPx * 0.005))
  }

  /** Scroll to zoom, clamped so the globe stays in frame. */
  zoomBy(deltaY: number) {
    this.zoom = Math.max(0.6, Math.min(4, this.zoom * (deltaY > 0 ? 0.9 : 1.1)))
    this.applyCamera()
  }

  /** Advance rotation and render. dtMs is wall time. */
  render(dtMs: number) {
    // gentle idle auto-rotation; disabled the moment the user drags
    if (this.autoRotate) this.yaw += ((0.02 * Math.PI) / 180) * (dtMs / (1000 / 60))
    this.group.rotation.set(this.tilt, this.yaw, 0)
    this.group.updateMatrixWorld(true)
    this.camera.updateMatrixWorld(true)

    this.m
      .copy(this.camera.projectionMatrix)
      .multiply(this.camera.matrixWorldInverse)
      .multiply(this.group.matrixWorld)

    // camera position expressed in the group's local frame, for the horizon test
    this.camLocal
      .copy(this.camera.position)
      .applyMatrix4(new THREE.Matrix4().copy(this.group.matrixWorld).invert())

    this.renderer.render(this.scene, this.camera)
  }

  /** Project a *local-frame* point to screen px. Returns null when the sphere
   *  occludes it. */
  project(v: THREE.Vector3, out: { x: number; y: number; behind: boolean }): boolean {
    const e = this.m.elements
    const x = v.x, y = v.y, z = v.z
    const cx = e[0] * x + e[4] * y + e[8] * z + e[12]
    const cy = e[1] * x + e[5] * y + e[9] * z + e[13]
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15]
    if (cw <= 0) return false
    out.x = (cx / cw * 0.5 + 0.5) * this._w
    out.y = (-(cy / cw) * 0.5 + 0.5) * this._h
    out.behind = this.occluded(x, y, z)
    return true
  }

  /** Does the unit sphere sit between the camera and this point? */
  occluded(x: number, y: number, z: number): boolean {
    const c = this.camLocal
    const dx = x - c.x, dy = y - c.y, dz = z - c.z
    const a = dx * dx + dy * dy + dz * dz
    const b = 2 * (c.x * dx + c.y * dy + c.z * dz)
    const cc = c.x * c.x + c.y * c.y + c.z * c.z - R * R
    const disc = b * b - 4 * a * cc
    if (disc <= 0) return false
    const s = Math.sqrt(disc)
    const t = (-b - s) / (2 * a)
    return t > 1e-4 && t < 1 - 1e-4
  }

  get width() { return this._w }
  get height() { return this._h }
}
