/**
 * Feature flags — the "not finished yet" seam.
 *
 * PRE-M1 used this to gate demo scaffolding. That scaffolding is gone: the
 * simulated pipeline, the mock job store and the "Use example prompt" helper
 * were deleted when generation became real, and `demoMode`/`demoHelpers` went
 * with them.
 *
 * What remains is a narrower and still-true statement about this build.
 */
export const featureFlags = {
  /**
   * Accounts, subscriptions and billing are not connected (M3.01, M3.11).
   *
   * Generation IS connected — a request creates a real job, a real worker
   * claims it, and progress streams over SSE. Only the account layer is
   * outstanding, which is exactly what the preview badge now says. Overstating
   * this ("everything is simulated") would be as misleading as understating it.
   *
   * M3: set false once authentication and billing ship.
   */
  previewBuild: true,
} as const;

export type FeatureFlags = typeof featureFlags;
