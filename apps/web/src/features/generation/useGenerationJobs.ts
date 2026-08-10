"use client";

import { useMemo } from "react";
import { create } from "zustand";
import type { WorkflowDefinition } from "@/features/workflows/types";
import type { GenerationJob, GenerationParameters } from "./types";
import { isRunning } from "./types";
import { MOCK_THUMBS, runMockPipeline } from "./mockPipeline";

/**
 * Generation job store.
 *
 * Zustand is used here and NOWHERE else, because this is the only genuinely
 * cross-page state in the demo: the shell's active-jobs indicator and the
 * Workspace's job strip both read it, and jobs must keep running while the
 * user browses other screens (guide §7 Step 5 — "users will be able to
 * continue working while generations are running").
 *
 * The store owns its timers, so no component can leak one.
 */

interface CreateJobInput {
  workflow: WorkflowDefinition;
  prompt: string;
  parameters: GenerationParameters;
}

interface GenerationJobsState {
  jobs: GenerationJob[];
  selectedJobId: string | null;

  createJob: (input: CreateJobInput) => string;
  selectJob: (id: string) => void;
  /** Cancels every timer and clears all jobs — demo housekeeping. */
  resetDemo: () => void;
}

/** jobId → cancel function for the running simulation. */
const pipelines = new Map<string, () => void>();

let jobCounter = 0;

export const useGenerationJobs = create<GenerationJobsState>((set, get) => ({
  jobs: [],
  selectedJobId: null,

  createJob: ({ workflow, prompt, parameters }) => {
    jobCounter += 1;
    const id = `gen_${jobCounter}`;

    const job: GenerationJob = {
      id,
      workflowId: workflow.id,
      workflowName: workflow.name,
      prompt,
      parameters,
      status: "Queued",
      hint: "",
      progress: 0,
      thumb: MOCK_THUMBS[(jobCounter - 1) % MOCK_THUMBS.length],
      createdAt: Date.now(),
    };

    // Newest first — matches the approved job strip.
    set((state) => ({
      jobs: [job, ...state.jobs],
      selectedJobId: id,
    }));

    const cancel = runMockPipeline((update) => {
      set((state) => ({
        jobs: state.jobs.map((existing) =>
          existing.id === id ? { ...existing, ...update } : existing,
        ),
      }));
      if (!isRunning(update.status)) pipelines.delete(id);
    });

    pipelines.set(id, cancel);
    return id;
  },

  selectJob: (id) => {
    if (get().jobs.some((job) => job.id === id)) set({ selectedJobId: id });
  },

  resetDemo: () => {
    pipelines.forEach((cancel) => cancel());
    pipelines.clear();
    set({ jobs: [], selectedJobId: null });
  },
}));

/* ── Derived selectors ─────────────────────────────────────────────────────
   Defined as standalone functions so components subscribe narrowly and a job
   progressing does not re-render screens that only care about the count.      */

/**
 * ⚠️ Selectors passed to `useGenerationJobs` MUST return a stable reference.
 *
 * Zustand v5 reads through `useSyncExternalStore`, which compares snapshots by
 * identity. A selector like `state.jobs.filter(...)` builds a NEW array on every
 * read, so React sees the snapshot change on every render and loops until it
 * throws "Maximum update depth exceeded" (React error #185).
 *
 * So: subscribe to raw state here, and derive with `useMemo` in the component.
 */
export function useActiveJobs(): GenerationJob[] {
  const jobs = useGenerationJobs((state) => state.jobs);
  return useMemo(() => jobs.filter((job) => isRunning(job.status)), [jobs]);
}

/**
 * Safe as a plain selector: `find` returns the same object reference when the
 * job is unchanged, and a stable `null` otherwise.
 */
export function selectSelectedJob(
  state: GenerationJobsState,
): GenerationJob | null {
  const { jobs, selectedJobId } = state;
  if (!selectedJobId) return null;
  return jobs.find((job) => job.id === selectedJobId) ?? null;
}
