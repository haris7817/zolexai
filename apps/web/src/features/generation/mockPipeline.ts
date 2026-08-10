import type { JobStatus } from "./types";

/**
 * ===========================================================================
 * SIMULATED generation pipeline — PRE-M1 ONLY
 * ===========================================================================
 *
 * ⚠️ No AI runs here. This is a timer that walks a job through the same states
 * the real backend will emit, so the client can approve the generation
 * EXPERIENCE before any GPU spend (guide §7 Step 4).
 *
 * Stage timings are lifted from the approved prototype. Total ≈ 6.6s — long
 * enough to read as real work, short enough for a 2–5 minute walkthrough.
 *
 * ── M1 REPLACEMENT (task M1.22) ──────────────────────────────────────────
 * This file is the ONLY place the simulation lives. To go real:
 *   1. POST /api/v1/generations to create the job (M1.18)
 *   2. Subscribe to GET /api/v1/generations/{id}/events via SSE (M1.21)
 *   3. Emit the same `PipelineUpdate` shape from those events
 * Nothing else in the frontend changes — the store and every component already
 * consume this interface.
 */

export interface PipelineStage {
  status: JobStatus;
  hint: string;
  progress: number;
  /** How long this stage is displayed before advancing. */
  durationMs: number;
}

export const MOCK_STAGES: PipelineStage[] = [
  {
    status: "Queued",
    hint: "Waiting for an available slot…",
    progress: 8,
    durationMs: 1200,
  },
  {
    status: "Preparing",
    hint: "Setting up your generation…",
    progress: 22,
    durationMs: 1400,
  },
  {
    status: "Generating",
    hint: "This usually takes a couple of minutes.",
    progress: 62,
    durationMs: 2600,
  },
  {
    status: "Finalizing",
    hint: "Polishing and encoding…",
    progress: 90,
    durationMs: 1400,
  },
];

export interface PipelineUpdate {
  status: JobStatus;
  hint: string;
  progress: number;
}

/**
 * Walks one job through the stages, then completes it.
 *
 * Returns a cancel function that clears the pending timer. The caller owns
 * cancellation — the store cancels on `resetDemo()` so timers can never
 * outlive the jobs they belong to.
 */
export function runMockPipeline(
  onUpdate: (update: PipelineUpdate) => void,
): () => void {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let cancelled = false;

  const advance = (index: number) => {
    if (cancelled) return;

    if (index >= MOCK_STAGES.length) {
      onUpdate({ status: "Completed", hint: "", progress: 100 });
      return;
    }

    const stage = MOCK_STAGES[index];
    onUpdate({
      status: stage.status,
      hint: stage.hint,
      progress: stage.progress,
    });
    timer = setTimeout(() => advance(index + 1), stage.durationMs);
  };

  // Kick off on the next tick so the caller can finish its state update first.
  timer = setTimeout(() => advance(0), 60);

  return () => {
    cancelled = true;
    if (timer) clearTimeout(timer);
  };
}

/** Placeholder thumbnails, cycled so consecutive jobs look distinct. */
export const MOCK_THUMBS: string[] = [
  "linear-gradient(140deg, #222C10, #121808)",
  "linear-gradient(140deg, #2C3A0B, #141A0C)",
  "linear-gradient(140deg, #33430D, #182008)",
  "linear-gradient(140deg, #28340D, #10160A)",
];
