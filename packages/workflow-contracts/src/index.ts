/**
 * ===========================================================================
 * Shared ZolexAI API contracts
 * ===========================================================================
 *
 * One definition of every workflow id, lifecycle status and public payload
 * shape, imported by `apps/web` and by nothing else in TypeScript
 * (directive §26). Nothing may hard-code these strings anywhere else.
 *
 * ## Why Zod schemas rather than plain `type` declarations
 *
 * A TypeScript type is erased at runtime, so it asserts nothing about what the
 * API actually sent — it only makes the compiler agree with an assumption. The
 * backend is Python and evolves independently; a field renamed there would
 * produce `undefined` at runtime while the build stayed green.
 *
 * Parsing responses through these schemas turns that into a loud, located
 * failure at the boundary rather than a mystery crash three components deep.
 * The types below are *derived* from the schemas, so the two can never drift.
 *
 * ## Coupling
 *
 * This is a description of the HTTP contract, not a mirror of Python
 * implementation details. It contains no ORM shapes, no internal worker
 * protocol, and nothing from a workflow's private `execution` block. The
 * pairing is verified by a contract test that runs these schemas against real
 * API responses (`apps/web/tests/contract.spec.ts`).
 */

import { z } from "zod";

/* ── Workflows ─────────────────────────────────────────────────────────── */

export const WORKFLOW_IDS = [
  "text-to-video",
  "image-to-video",
  "video-to-video",
  "extend-video",
  "music",
  "music-video",
] as const;

export const workflowIdSchema = z.enum(WORKFLOW_IDS);
export type WorkflowId = z.infer<typeof workflowIdSchema>;

export const workflowIconSchema = z.enum([
  "sparkles",
  "image",
  "repeat",
  "extend",
  "music",
  "clapper",
]);
export type WorkflowIcon = z.infer<typeof workflowIconSchema>;

export const outputTypeSchema = z.enum(["video", "audio", "image"]);
export type OutputType = z.infer<typeof outputTypeSchema>;

/**
 * How a workflow's output duration is decided (M2 client requirements):
 * `fixed` — user picks from `supported_durations` ("5s"…); `source` — matches
 * the uploaded file automatically, the list is empty and no duration is sent;
 * `minutes` — user picks a song length ("1m"…"5m").
 *
 * Deliberately NOT `.default()`ed: the API always serves it, and a silent
 * fallback here would mask exactly the kind of drift this contract exists to
 * catch.
 */
export const durationModeSchema = z.enum(["fixed", "source", "minutes"]);
export type DurationMode = z.infer<typeof durationModeSchema>;

export const assetKindSchema = z.enum(["video", "image", "audio"]);
export type AssetKind = z.infer<typeof assetKindSchema>;

export const workflowInputSchema = z.object({
  role: z.string(),
  kind: assetKindSchema,
  required: z.boolean(),
  label: z.string(),
  drop_hint: z.string(),
  help: z.string().default(""),
  accept: z.array(z.string()).default([]),
  max_size_mb: z.number().int().positive(),
});
export type WorkflowInput = z.infer<typeof workflowInputSchema>;

export const workflowSchema = z.object({
  id: z.string(),
  version: z.string(),
  name: z.string(),
  category: z.enum(["video", "audio"]),
  output_type: outputTypeSchema,

  description: z.string(),
  short_description: z.string(),
  marketing_description: z.string(),

  prompt: z.object({
    required: z.boolean(),
    placeholder: z.string(),
    max_length: z.number().int().positive(),
  }),

  inputs: z.array(workflowInputSchema),

  duration_mode: durationModeSchema,
  supported_durations: z.array(z.string()),
  supported_aspect_ratios: z.array(z.string()),
  supported_quality_levels: z.array(z.string()),

  settings: z.object({
    quality: z.boolean(),
    motion_strength: z.boolean(),
    prompt_adherence: z.boolean(),
    seed: z.boolean(),
    // Music only: a lyrics box plus a lyric-language choice. Defaulted so
    // API responses predating the field still parse.
    lyrics: z.boolean().default(false),
    // Text to Video only: the Standard / Idea (Director) prompt-mode toggle
    // plus, in Director mode, a dialogue-language choice. Defaulted for the
    // same reason as `lyrics`.
    prompt_modes: z.boolean().default(false),
  }),

  capabilities: z.object({
    download: z.boolean(),
    extend: z.boolean(),
    reuse_settings: z.boolean(),
    variation: z.boolean(),
  }),

  ui: z.object({ icon: workflowIconSchema, thumb: z.string() }),
});
export type Workflow = z.infer<typeof workflowSchema>;

export const workflowListSchema = z.object({ workflows: z.array(workflowSchema) });

/* ── Generation lifecycle ──────────────────────────────────────────────── */

/**
 * The machine contract. The UI never renders these directly — it shows
 * `stage_label`, which the API resolves — so internal granularity can change
 * without touching a component.
 */
export const jobStatusSchema = z.enum([
  "queued",
  "assigned",
  "preparing",
  "generating",
  "post_processing",
  "uploading",
  "completed",
  "failed",
  "cancelled",
]);
export type JobStatus = z.infer<typeof jobStatusSchema>;

export const TERMINAL_STATUSES: readonly JobStatus[] = [
  "completed",
  "failed",
  "cancelled",
] as const;

export function isTerminal(status: JobStatus): boolean {
  return TERMINAL_STATUSES.includes(status);
}

export function isRunning(status: JobStatus): boolean {
  return !isTerminal(status);
}

export const generationOutputSchema = z.object({
  asset_id: z.string(),
  kind: assetKindSchema,
  is_primary: z.boolean(),
  url: z.string().nullable().default(null),

  /* What the delivered file actually is, measured from it rather than echoed
     back from the request. The preview frame is sized from these, and the
     duration shown beside a result comes from here — an extension's requested
     "5s" is not the length of the 14-second file it produced. */
  width: z.number().nullable().default(null),
  height: z.number().nullable().default(null),
  duration_seconds: z.number().nullable().default(null),
});
export type GenerationOutput = z.infer<typeof generationOutputSchema>;

export const generationJobSchema = z.object({
  id: z.string(),
  workflow_id: z.string(),
  workflow_name: z.string(),

  status: jobStatusSchema,
  stage_label: z.string(),
  progress: z.number().int().min(0).max(100),
  hint: z.string(),

  prompt: z.string(),
  parameters: z.record(z.string(), z.unknown()).default({}),

  inputs: z
    .array(z.object({ role: z.string(), asset_id: z.string(), kind: assetKindSchema }))
    .default([]),
  outputs: z.array(generationOutputSchema).default([]),

  error: z.object({ code: z.string(), message: z.string() }).nullable().default(null),

  attempt_count: z.number().int(),
  created_at: z.string(),
  updated_at: z.string(),
  started_at: z.string().nullable().default(null),
  completed_at: z.string().nullable().default(null),

  last_event_seq: z.number().int(),
  is_terminal: z.boolean(),
});
export type GenerationJob = z.infer<typeof generationJobSchema>;

export const generationAcceptedSchema = z.object({
  job_id: z.string(),
  status: jobStatusSchema,
  stage_label: z.string(),
  events_url: z.string(),
});
export type GenerationAccepted = z.infer<typeof generationAcceptedSchema>;

/** One SSE frame. `seq` is the reconnection cursor. */
export const jobEventSchema = z.object({
  seq: z.number().int(),
  event_type: z.enum(["status", "progress", "completed", "failed", "cancelled"]),
  status: jobStatusSchema,
  stage_label: z.string(),
  progress: z.number().int(),
  message: z.string(),
  payload: z.record(z.string(), z.unknown()).default({}),
  created_at: z.string(),
});
export type JobEvent = z.infer<typeof jobEventSchema>;

/* ── Media ─────────────────────────────────────────────────────────────── */

export const assetSchema = z.object({
  id: z.string(),
  kind: assetKindSchema,
  source: z.enum(["upload", "generated"]),
  status: z.enum(["pending", "ready", "failed"]),
  name: z.string(),
  content_type: z.string(),
  size_bytes: z.number().nullable().default(null),
  duration_seconds: z.number().nullable().default(null),
  width: z.number().nullable().default(null),
  height: z.number().nullable().default(null),
  created_at: z.string(),
  url: z.string().nullable().default(null),
});
export type Asset = z.infer<typeof assetSchema>;

export const uploadUrlSchema = z.object({
  asset_id: z.string(),
  upload: z.object({
    url: z.string(),
    method: z.string(),
    headers: z.record(z.string(), z.string()),
    expires_in: z.number().int(),
  }),
  confirm_url: z.string(),
});
export type UploadUrl = z.infer<typeof uploadUrlSchema>;

export const mediaCountsSchema = z.object({
  all: z.number().int(),
  video: z.number().int(),
  image: z.number().int(),
  audio: z.number().int(),
});
export type MediaCounts = z.infer<typeof mediaCountsSchema>;

/* ── Envelopes ─────────────────────────────────────────────────────────── */

/** Keyset pagination. `next_cursor` is opaque — never build one client-side. */
export function pageSchema<T extends z.ZodTypeAny>(item: T) {
  return z.object({
    items: z.array(item),
    next_cursor: z.string().nullable().default(null),
    has_more: z.boolean().default(false),
  });
}

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
  has_more: boolean;
}

export const apiErrorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.string(), z.unknown()).default({}),
    request_id: z.string().nullish(),
  }),
});
export type ApiErrorBody = z.infer<typeof apiErrorSchema>;

/** Machine-readable error codes a client may branch on. */
export const ERROR_CODES = {
  validationFailed: "validation_failed",
  notFound: "not_found",
  conflict: "conflict",
  rateLimited: "rate_limited",
  concurrencyLimitReached: "concurrency_limit_reached",
  unsupportedParameter: "unsupported_parameter",
  missingRequiredInput: "missing_required_input",
  unsupportedMediaType: "unsupported_media_type",
  fileTooLarge: "file_too_large",
  internalError: "internal_error",
} as const;

export type ErrorCode = (typeof ERROR_CODES)[keyof typeof ERROR_CODES];
