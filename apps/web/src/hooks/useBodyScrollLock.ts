"use client";

import { useEffect } from "react";

/**
 * Locks body scroll while an overlay (nav drawer / settings sheet) is open.
 *
 * The source prototype mutated `document.body.style.overflow` *during render*,
 * which is a side effect in the render phase — illegal in React and unsafe
 * under concurrent rendering. It belongs in an effect.
 *
 * Reference-counted so two overlays open at once (drawer over sheet) cannot
 * have the first to close release the lock for both.
 */
let lockCount = 0;
let previousOverflow = "";

export function useBodyScrollLock(active: boolean): void {
  useEffect(() => {
    if (!active) return;

    if (lockCount === 0) {
      previousOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
    }
    lockCount += 1;

    return () => {
      lockCount -= 1;
      if (lockCount === 0) {
        document.body.style.overflow = previousOverflow;
      }
    };
  }, [active]);
}
