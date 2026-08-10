"use client";

import { useEffect } from "react";

/**
 * Calls `onEscape` when Escape is pressed, while `active` is true.
 *
 * The approved design closes both the mobile nav drawer and the settings sheet
 * on Escape; that behaviour is part of what the client is approving.
 */
export function useEscapeKey(active: boolean, onEscape: () => void): void {
  useEffect(() => {
    if (!active) return;

    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape") onEscape();
    };

    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [active, onEscape]);
}
