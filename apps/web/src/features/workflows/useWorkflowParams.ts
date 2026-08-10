"use client";

import { create } from "zustand";
import type { WorkflowDefinition } from "./types";

/**
 * ===========================================================================
 * Creator parameters — one source of truth for the active workflow's settings
 * ===========================================================================
 *
 * WHY A STORE AND NOT LOCAL STATE
 * Switching workflow is a route change (/app/create/text-to-video →
 * /app/create/music). Local component state would be at the mercy of whether
 * React preserves the page component across a param change — and the whole
 * point of the behaviour below is that settings SURVIVE that switch. A store
 * makes it deliberate rather than incidental.
 *
 * THE RULE (approved design, lines 476–486)
 * On switching workflow, keep duration / aspect ratio / quality if the incoming
 * workflow supports the current value; otherwise fall back to its first
 * supported value. This is the architecture doc's named bug
 * `fix(web): preserve duration on workflow switch`, and M1.08's acceptance
 * criterion "safe fallback when switching workflows".
 *
 * Worked example the demo is verified against:
 *   Text to Video @ 15s  →  Image to Video (supports only 5s / 10s)  →  5s
 * It must not crash, and must not keep an unsupported 15s.
 */

interface CreatorParamsState {
  prompt: string;
  duration: string;
  /** null when the workflow has no aspect ratios (audio output). */
  aspect: string | null;
  /** null when the workflow exposes no quality levels. */
  quality: string | null;
  motionStrength: number;
  promptAdherence: number;
  seedLocked: boolean;
  advancedOpen: boolean;

  setPrompt: (value: string) => void;
  setDuration: (value: string) => void;
  setAspect: (value: string) => void;
  setQuality: (value: string) => void;
  setMotionStrength: (value: number) => void;
  setPromptAdherence: (value: number) => void;
  toggleSeedLocked: () => void;
  toggleAdvanced: () => void;

  /** Applies the preservation rule. Call whenever the active workflow changes. */
  syncWorkflow: (workflow: WorkflowDefinition) => void;
}

export const useCreatorParams = create<CreatorParamsState>((set) => ({
  // The workspace opens as a natural EMPTY creator state — no pre-filled
  // prompt, so nothing looks artificially pre-configured and the disabled
  // Generate button demonstrates validation immediately.
  prompt: "",
  duration: "5s",
  aspect: "16:9",
  quality: "High",
  motionStrength: 60,
  promptAdherence: 75,
  seedLocked: false,
  advancedOpen: false,

  setPrompt: (prompt) => set({ prompt }),
  setDuration: (duration) => set({ duration }),
  setAspect: (aspect) => set({ aspect }),
  setQuality: (quality) => set({ quality }),
  setMotionStrength: (motionStrength) => set({ motionStrength }),
  setPromptAdherence: (promptAdherence) => set({ promptAdherence }),
  toggleSeedLocked: () => set((s) => ({ seedLocked: !s.seedLocked })),
  toggleAdvanced: () => set((s) => ({ advancedOpen: !s.advancedOpen })),

  syncWorkflow: (workflow) =>
    set((state) => {
      const next: Partial<CreatorParamsState> = {};

      if (!workflow.supportedDurations.includes(state.duration)) {
        next.duration = workflow.supportedDurations[0];
      }

      const aspects: string[] = [...workflow.supportedAspectRatios];
      if (!state.aspect || !aspects.includes(state.aspect)) {
        next.aspect = aspects[0] ?? null;
      }

      const qualities: string[] = [...workflow.supportedQualityLevels];
      if (!state.quality || !qualities.includes(state.quality)) {
        next.quality = qualities[0] ?? null;
      }

      // Return the same object when nothing changed so subscribers don't churn.
      return Object.keys(next).length ? next : state;
    }),
}));
