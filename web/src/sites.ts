/* ────────────────────────────────────────────────────────────────────────
   sites.ts — surface-asset roles.

   ⚠ PRESENTATION ONLY. The simulator has no concept of an asset role: every
   entry in data/ground_stations.json is just a lat/lon with an elevation mask,
   and the trace carries `operational` and nothing else. The roles below are a
   display convention invented for the demo so the globe reads as a mixed
   surface network rather than twelve identical dots.

   Nothing downstream of the renderer may branch on these. Diagnosis, policy,
   contacts and diagnosability all remain blind to role — as they must, since
   the model never saw it either.
   ──────────────────────────────────────────────────────────────────────── */

export type SiteRole = 'DSN' | 'RELAY' | 'ROVER'

/** Role per station id. Sites absent from this map fall back to RELAY. */
export const SITE_ROLE: Record<string, SiteRole> = {
  /* Deep-space complexes — the high-gain dishes that carry the Earth link. */
  GS_SVALBARD: 'DSN',
  GS_KOUROU: 'DSN',
  GS_DONGARA: 'DSN',

  /* Surface assets — the things actually generating science data. */
  GS_WALLOPS: 'ROVER',
  GS_POKERFLAT: 'ROVER',
  GS_TROLL: 'ROVER',
  GS_SANTIAGO: 'ROVER',

  /* Relay terminals — everything else. */
  GS_PUNTA: 'RELAY',
  GS_AWARUA: 'RELAY',
  GS_KIRUNA: 'RELAY',
  GS_INUVIK: 'RELAY',
  GS_REDU: 'RELAY',
}

export const roleOf = (id: string): SiteRole => SITE_ROLE[id] ?? 'RELAY'

/** Short tag drawn beside the site label. */
export const ROLE_TAG: Record<SiteRole, string> = {
  DSN: 'DSN',
  RELAY: 'RLY',
  ROVER: 'RVR',
}
