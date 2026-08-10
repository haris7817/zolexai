"use client";

import { useSyncExternalStore } from "react";

/**
 * ZolexAI responsive modes. These match the approved design exactly:
 *
 *   mobile   < 768
 *   tablet   768–1023
 *   laptop   1024–1439
 *   desktop  >= 1440
 */
export type Breakpoint = "mobile" | "tablet" | "laptop" | "desktop";

const QUERY = {
  tablet: "(min-width: 768px)",
  laptop: "(min-width: 1024px)",
  desktop: "(min-width: 1440px)",
} as const;

/**
 * ⚠️ DO NOT use this hook to drive layout or visibility.
 *
 * The source prototype read `window.innerWidth` during render, which in Next.js
 * is an SSR hydration mismatch. Everything visual is therefore done in CSS with
 * the `tablet:` / `laptop:` / `desktop:` variants, and this hook exists only to
 * supply things CSS cannot express:
 *
 *   · the settings panel's `role` / `aria-modal` (a plain <aside> on
 *     laptop+ but a modal dialog on tablet and below)
 *   · whether Escape-to-close and body scroll lock currently apply
 *
 * Those correct silently one frame after hydration — an accessibility attribute
 * settling is invisible, whereas a layout settling is a visible flash.
 *
 * Implemented with useSyncExternalStore so React uses `getServerSnapshot`
 * during SSR and hydration, then re-renders with the real value. That is the
 * sanctioned pattern and produces no hydration warning.
 */
export function useBreakpoint(): Breakpoint {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
}

function subscribe(onChange: () => void): () => void {
  const lists = Object.values(QUERY).map((q) => window.matchMedia(q));
  lists.forEach((list) => list.addEventListener("change", onChange));
  return () =>
    lists.forEach((list) => list.removeEventListener("change", onChange));
}

function getSnapshot(): Breakpoint {
  if (window.matchMedia(QUERY.desktop).matches) return "desktop";
  if (window.matchMedia(QUERY.laptop).matches) return "laptop";
  if (window.matchMedia(QUERY.tablet).matches) return "tablet";
  return "mobile";
}

/**
 * Static prerender assumes desktop — the primary design review width. Because
 * this value never drives layout, an incorrect guess costs nothing visually.
 */
function getServerSnapshot(): Breakpoint {
  return "desktop";
}

/**
 * Tablet and mobile share "compact" behaviour: the settings panel becomes a
 * modal overlay and navigation collapses.
 */
export function useIsCompact(): boolean {
  const breakpoint = useBreakpoint();
  return breakpoint === "mobile" || breakpoint === "tablet";
}
