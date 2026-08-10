import type { WorkflowId } from "@/features/workflows/types";

/**
 * Customer-facing generation states.
 *
 * These are the friendly labels from architecture doc §16. The backend's
 * internal statuses (queued / assigned / preparing / generating /
 * post_processing / uploading / completed / failed / cancelled) map onto these
 * at the API boundary in M1 — the UI never sees the internal set.
 */
export type JobStatus =
  | "Queued"
  | "Preparing"
  | "Generating"
  | "Finalizing"
  | "Completed"
  | "Failed"
  | "Cancelled";

export const TERMINAL_STATUSES: JobStatus[] = [
  "Completed",
  "Failed",
  "Cancelled",
];

export function isRunning(status: JobStatus): boolean {
  return !TERMINAL_STATUSES.includes(status);
}

/** Parameters echoed back on a result — shaped like architecture doc §15. */
export interface GenerationParameters {
  duration: string;
  aspectRatio: string | null;
  quality: string | null;
  motionStrength: number;
  promptAdherence: number;
  seed: number | null;
}

export interface GenerationJob {
  id: string;
  workflowId: WorkflowId;
  /** Display name captured at submit time, so renaming a workflow never rewrites history. */
  workflowName: string;
  prompt: string;
  parameters: GenerationParameters;

  status: JobStatus;
  /** Supporting line beneath the status — e.g. "Polishing and encoding…". */
  hint: string;
  /** 0–100. */
  progress: number;

  /** Placeholder gradient standing in for a real thumbnail. */
  thumb: string;
  createdAt: number;
}
