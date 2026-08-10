/**
 * Feature flags — the demo/production seam.
 *
 * Everything that exists ONLY to support the PRE-M1 client demo is gated here,
 * so Milestone 1 removes demo scaffolding by flipping booleans rather than
 * hunting through components.
 */
export const featureFlags = {
  /**
   * PRE-M1 master switch. While true the app runs entirely on mock data and
   * surfaces the "UI Preview" disclosure so the client cannot mistake the
   * simulated pipeline for connected AI (guide §3, §4).
   *
   * M1: set false once the FastAPI backend and SSE are wired up.
   */
  demoMode: true,

  /**
   * Presenter conveniences used during the guided walkthrough — currently the
   * "Use example prompt" action beside the PROMPT label (guide §7 Step 3).
   *
   * The workspace deliberately opens EMPTY so the client sees a natural creator
   * state rather than a pre-configured one; this makes the scripted prompt one
   * click away without faking it.
   *
   * M1: set false. This is not product UI.
   */
  demoHelpers: true,
} as const;

export type FeatureFlags = typeof featureFlags;
