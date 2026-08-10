/**
 * ZolexAI brand configuration — one source of truth.
 *
 * Every screen (Landing, Dashboard, Workspace, app screens, auth) reads these
 * values. The three source design files each hardcoded "ZolexAI" and the "Z"
 * mark independently; collapsing them here is what architecture doc §40 asks for.
 */
export const brand = {
  name: "ZolexAI",
  /** The single-letter mark used in the gradient logo tile. */
  shortName: "Z",
  tagline: "The AI creative workspace",
  description:
    "Create professional AI videos, music and visual content from one powerful workspace.",
  /** Landing footer copy. */
  footerTagline:
    "The AI creative workspace for video, music and visual content.",
} as const;

export type Brand = typeof brand;
