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
    /** Only ever sent as "director" — Standard mode is expressed by absence. */
    prompt_mode?: string;
    dialogue_language?: string;
  };
  inputs?: Record<string, string>;
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
