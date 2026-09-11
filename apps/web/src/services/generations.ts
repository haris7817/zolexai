import {
  generationAcceptedSchema,
  generationJobSchema,
  pageSchema,
  type GenerationAccepted,
  type GenerationJob,
  type JobStatus,
  type Page,
} from "@zolexai/workflow-contracts";
import { apiFetch } from "@/lib/api/client";

/** Exactly the body `POST /generations` accepts. */
export interface CreateGenerationInput {
  workflow_id: string;
  prompt: string;
  parameters: {
    /** Omitted for automatic-duration workflows — the file sets the length. */
    duration?: string;
    aspect_ratio?: string | null;
    quality?: string | null;
    motion_strength?: number;
    prompt_adherence?: number;
    seed?: number | null;
    lyrics?: string;
    lyrics_language?: string;
    /** Only ever sent as false — "with sound" is expressed by absence. */
    sound?: boolean;
    /** Only ever sent as "director" — Standard mode is expressed by absence. */
    prompt_mode?: string;
    dialogue_language?: string;
    /** Only ever sent as true — the deployment default is expressed by
     *  absence, so a client that never heard of it is byte-identical. */
    auto_dialogue?: boolean;
    /** Sent only alongside `auto_dialogue: true`; the API rejects it alone. */
    maximum_speakers?: number;
    /** A public YouTube/Vimeo link whose camera the treatment borrows, on
     *  workflows declaring `settings.reference_video`. Sent only when given. */
    reference_video_url?: string;
    /** Music Lyrics Workflow v2.0, on workflows declaring
     *  `settings.lyrics_workflow`. Each is sent only when it differs from
     *  the workflow's default, so an untouched panel is byte-identical. */
    rhyme_scheme?: "AABB" | "ABAB" | "AAAA";
    rhyme_mode?: "relaxed";
    point_of_view?: "first_person" | "second_person" | "third_person";
    clean_mode?: false;
    reference_audio_url?: string;
    /** The band, on workflows declaring `settings.performers`: one entry per
     *  member, paired with the `performer_{slot}` picture input. Sent only
     *  when at least one member is given. */
    performers?: PerformerInput[];
  };
  inputs?: Record<string, string>;
}

export interface PerformerInput {
  slot: number;
  role: string;
  description?: string;
}

const generationPageSchema = pageSchema(generationJobSchema);

/**
 * Submits a generation and returns as soon as the job exists.
 *
 * The API answers 202 with a job id — nothing has been generated yet. Progress
 * arrives over SSE (`useGenerationStream`). A request that waited for the
 * result would hold a connection for minutes.
 *
 * `idempotencyKey` makes a double-click or a retry safe: the repeat returns the
 * original job instead of starting a second, separately-billed one
 * (directive §24).
 */
export async function createGeneration(
  input: CreateGenerationInput,
  idempotencyKey: string,
  signal?: AbortSignal,
): Promise<GenerationAccepted> {
  return apiFetch("/generations", generationAcceptedSchema, {
    method: "POST",
    body: input,
    signal,
    headers: { "Idempotency-Key": idempotencyKey },
  });
}

export interface ListGenerationsParams {
  limit?: number;
  cursor?: string | null;
  status?: JobStatus[];
  workflowId?: string | null;
}

export async function listGenerations(
  params: ListGenerationsParams = {},
  signal?: AbortSignal,
): Promise<Page<GenerationJob>> {
  const query = new URLSearchParams();
  query.set("limit", String(params.limit ?? 24));
  if (params.cursor) query.set("cursor", params.cursor);
  if (params.workflowId) query.set("workflow_id", params.workflowId);
  // Repeated `status` keys, matching FastAPI's list-query convention.
  for (const status of params.status ?? []) query.append("status", status);

  return apiFetch(`/generations?${query.toString()}`, generationPageSchema, { signal });
}

export async function fetchGeneration(
  jobId: string,
  signal?: AbortSignal,
): Promise<GenerationJob> {
  return apiFetch(`/generations/${jobId}`, generationJobSchema, { signal });
}

export async function cancelGeneration(
  jobId: string,
  signal?: AbortSignal,
): Promise<GenerationJob> {
  return apiFetch(`/generations/${jobId}/cancel`, generationJobSchema, {
    method: "POST",
    signal,
  });
}

/** The result the UI displays — a job may also carry a thumbnail or preview. */
export function primaryOutput(job: GenerationJob) {
  return job.outputs.find((output) => output.is_primary) ?? job.outputs[0];
}

/**
 * What the PLAYER should load, which is not always what the Download button
 * gives you.
 *
 * A 4K or 8K master is 300-400 MB, and a browser asked to play one fetches a
 * large part of it before the first frame appears. When a job carries a
 * non-primary video output it is the 1080p fast-start copy the worker made
 * for exactly this (client instruction, 11 Sep 2026: "Do NOT play the actual
 * 8K master on your webpage"), so the page plays that and the customer still
 * downloads the real thing.
 *
 * Falls back to the primary output, which is every job made before previews
 * existed, every 1080p delivery, and every non-video result.
 */
export function playableOutput(job: GenerationJob) {
  const preview = job.outputs.find(
    (output) => !output.is_primary && output.kind === "video" && output.url,
  );
  return preview ?? primaryOutput(job);
}
