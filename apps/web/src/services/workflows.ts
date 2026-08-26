import {
  workflowListSchema,
  workflowSchema,
  type Workflow,
} from "@zolexai/workflow-contracts";
import { apiFetch } from "@/lib/api/client";

/**
 * Workflow catalogue access.
 *
 * Definitions are version-controlled YAML loaded at API startup, so they change
 * only on deploy. `revalidate: 60` lets a server render reuse a cached
 * catalogue for a minute rather than hitting the API on every page.
 */

export async function fetchWorkflows(signal?: AbortSignal): Promise<Workflow[]> {
  const data = await apiFetch("/workflows", workflowListSchema, { signal, revalidate: 60 });
  return data.workflows;
}

export async function fetchWorkflow(id: string, signal?: AbortSignal): Promise<Workflow> {
  return apiFetch(`/workflows/${id}`, workflowSchema, { signal, revalidate: 60 });
}

/* ── Derived helpers ────────────────────────────────────────────────────
   Kept here rather than inside components so no screen re-derives workflow
   behaviour with its own `if (workflow.id === ...)`. A difference between
   tools belongs in the definition, never in a branch (architecture rule #6). */

export function findWorkflow(workflows: Workflow[], id: string): Workflow | undefined {
  return workflows.find((workflow) => workflow.id === id);
}

export function requiredInputs(workflow: Workflow) {
  return workflow.inputs.filter((input) => input.required);
}

export function optionalInputs(workflow: Workflow) {
  return workflow.inputs.filter((input) => !input.required);
}

export function hasAdvancedSettings(workflow: Workflow): boolean {
  const { motion_strength, prompt_adherence, seed } = workflow.settings;
  return motion_strength || prompt_adherence || seed;
}

export function showsQuality(workflow: Workflow): boolean {
  return workflow.settings.quality && workflow.supported_quality_levels.length > 0;
}

export function showsAspectRatio(workflow: Workflow): boolean {
  return workflow.supported_aspect_ratios.length > 0;
}

/**
 * Chip label for a duration value. Second presets render as served ("5s");
 * minute values get the word, because "3m" reads as metres to half the world.
 */
export function durationLabel(value: string): string {
  const minutes = /^(\d+)m$/.exec(value);
  return minutes ? `${minutes[1]} min` : value;
}

/**
 * Display label for a quality level. Wire values stay lowercase ("fast",
 * "best"); the control shows them capitalised.
 */
export function qualityLabel(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}

/**
 * The duration ladder offered at the given quality level. A level absent
 * from the per-quality map — or no quality selected yet — offers the full
 * ladder, so the toggle can only ever narrow, never strand.
 */
export function durationsForQuality(
  workflow: Workflow,
  quality: string | null | undefined,
): string[] {
  if (!quality) return workflow.supported_durations;
  return (
    workflow.supported_durations_by_quality[quality] ?? workflow.supported_durations
  );
}

/**
 * Whether the sound on/off control renders — on every quality level
 * (client round two, 27 Aug): both engines deliver an audio track.
 */
export function showsSound(workflow: Workflow): boolean {
  return workflow.settings.sound;
}

/**
 * Customer copy for an automatic-duration workflow — worded from the source
 * input's kind, so Music Video says "audio" and Video to Video says "video"
 * without either being special-cased anywhere.
 */
export function autoDurationLabel(workflow: Workflow): string {
  const source = workflow.inputs.find(
    (input) => input.required && (input.kind === "video" || input.kind === "audio"),
  );
  return source?.kind === "audio" ? "Matches your audio" : "Same as source video";
}

/** Accepted MIME types for a role, as an `<input accept>` value. */
export function acceptAttribute(workflow: Workflow, role: string): string {
  const input = workflow.inputs.find((candidate) => candidate.role === role);
  return input?.accept.join(",") ?? "";
}
