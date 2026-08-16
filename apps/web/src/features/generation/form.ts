import { z } from "zod";
import type { Workflow } from "@zolexai/workflow-contracts";
import type { CreateGenerationInput } from "@/services/generations";

/**
 * ===========================================================================
 * The generation form — validation built FROM the workflow definition
 * ===========================================================================
 *
 * The schema is generated per workflow rather than written per workflow. Music
 * rejects an aspect ratio, Image to Video demands a source image, Extend Video
 * offers four durations — all of that comes from the definition the API served,
 * so a workflow added in M2 gets correct client validation with no frontend
 * change (architecture rule #6: never branch on a workflow id in a component).
 *
 * ## Why validate on the client at all when the API already does
 *
 * Not for security — the API is the authority and re-validates everything. This
 * is for latency and clarity: the user learns the prompt is empty before a
 * round trip, and `Generate` can be disabled rather than clickable-then-
 * rejected. The two agree because both derive from the same definition.
 *
 * ## Why the value type is declared, not inferred
 *
 * `z.infer` of a schema assembled from conditionals produces a DIFFERENT type
 * per workflow — and one where the input type (`prompt?: string`) differs from
 * the output type (`prompt: string`) wherever a `.default()` appears. React
 * Hook Form requires one stable shape whose input and output match, so the
 * shape is declared below and the builder is typed to produce it. Every branch
 * therefore has to satisfy the same contract, which is the point.
 */

export interface GenerationFormValues {
  prompt: string;
  /** null when the workflow's duration is automatic (`duration_mode: source`). */
  duration: string | null;
  /** null when the workflow exposes no aspect ratios (audio output). */
  aspectRatio: string | null;
  /** null when the workflow exposes no quality levels. */
  quality: string | null;
  motionStrength: number;
  promptAdherence: number;
  seedLocked: boolean;
  /** The customer's own lyric sheet — empty when they want them written.
   *  Only meaningful on workflows whose `settings.lyrics` is true. */
  lyrics: string;
  /** null when the workflow has no lyrics support. */
  lyricsLanguage: string | null;
  /** role → asset id. null while an optional input is unfilled. */
  inputs: Record<string, string | null>;
}

/**
 * Languages offered for GENERATED lyrics. The music model sings 50+ languages
 * natively — a customer's own pasted lyrics can be in any of them and need no
 * selection at all, because the model follows the sheet's language.
 */
export const LYRIC_LANGUAGES = [
  "English",
  "Urdu",
  "Hindi",
  "Arabic",
  "Spanish",
  "French",
  "German",
  "Portuguese",
  "Italian",
  "Turkish",
  "Russian",
  "Japanese",
  "Korean",
  "Chinese",
] as const;

/** Matches the API's GenerationParameters.lyrics max_length. */
const LYRICS_MAX_LENGTH = 10_000;

/**
 * A value that is either one of `allowed`, or null when the workflow has no
 * such control. Both states are expressed in one schema so the field type stays
 * `string | null` regardless of workflow.
 */
function choiceOrNull(allowed: readonly string[], message: string) {
  return z.union([z.string(), z.null()]).superRefine((value, ctx) => {
    if (allowed.length === 0) {
      // The control is not rendered, so anything but null means stale state
      // survived a workflow switch — a bug worth surfacing, not ignoring.
      if (value !== null) {
        ctx.addIssue({ code: "custom", message: "Not available for this tool." });
      }
      return;
    }
    if (typeof value !== "string" || !allowed.includes(value)) {
      ctx.addIssue({ code: "custom", message });
    }
  });
}

/**
 * Input and output are the SAME type on purpose.
 *
 * React Hook Form's resolver requires it, and it is also a useful constraint:
 * a `.default()` or `.transform()` anywhere in the schema would make the parsed
 * value differ from what the controls hold, so the form and the submitted
 * request could silently disagree. Forbidding that at the type level means
 * every branch of the builder has to keep them identical.
 */
export type GenerationSchema = z.ZodType<GenerationFormValues, GenerationFormValues>;

export function buildGenerationSchema(workflow: Workflow): GenerationSchema {
  const requiredRoles = workflow.inputs.filter((input) => input.required);
  const labelFor = (role: string) => {
    const label = workflow.inputs.find((input) => input.role === role)?.label ?? role;
    const lower = label.toLowerCase();
    return lower.charAt(0).toUpperCase() + lower.slice(1);
  };

  const schema = z.object({
    prompt: z
      .string()
      .max(
        workflow.prompt.max_length,
        `Keep the prompt under ${workflow.prompt.max_length} characters.`,
      )
      .superRefine((value, ctx) => {
        if (workflow.prompt.required && value.trim().length === 0) {
          ctx.addIssue({ code: "custom", message: "Describe what you want to create." });
        }
      }),

    // Source-mode workflows take their length from the uploaded file, so the
    // form must hold null there — the same null-when-absent shape aspect ratio
    // and quality already use.
    duration: choiceOrNull(
      workflow.duration_mode === "source" ? [] : workflow.supported_durations,
      "Choose a duration.",
    ),

    aspectRatio: choiceOrNull(workflow.supported_aspect_ratios, "Choose an aspect ratio."),

    quality: choiceOrNull(
      workflow.settings.quality ? workflow.supported_quality_levels : [],
      "Choose a quality level.",
    ),

    motionStrength: z.number().int().min(0).max(100),
    promptAdherence: z.number().int().min(0).max(100),
    seedLocked: z.boolean(),

    // Same stale-state discipline as choiceOrNull: on a workflow without the
    // control, anything but the empty/null resting state means values survived
    // a workflow switch — surfaced, not silently submitted.
    lyrics: z
      .string()
      .max(LYRICS_MAX_LENGTH, "Those lyrics are too long for one song.")
      .superRefine((value, ctx) => {
        if (!workflow.settings.lyrics && value.trim().length > 0) {
          ctx.addIssue({ code: "custom", message: "Not available for this tool." });
        }
      }),
    lyricsLanguage: choiceOrNull(
      workflow.settings.lyrics ? LYRIC_LANGUAGES : [],
      "Choose a language.",
    ),

    // Required roles are enforced here, which is what makes Video to Video's
    // OPTIONAL reference image work with no bespoke rule (directive §14).
    inputs: z
      .record(z.string(), z.union([z.string(), z.null()]))
      .superRefine((value, ctx) => {
        for (const input of requiredRoles) {
          if (!value[input.role]) {
            ctx.addIssue({
              code: "custom",
              path: [input.role],
              message: `${labelFor(input.role)} is required.`,
            });
          }
        }
      }),
  });

  return schema as unknown as GenerationSchema;
}

/** Fresh defaults for a workflow — first supported value of each control. */
export function defaultValuesFor(workflow: Workflow): GenerationFormValues {
  return {
    prompt: "",
    duration:
      workflow.duration_mode === "source" ? null : (workflow.supported_durations[0] ?? null),
    aspectRatio: workflow.supported_aspect_ratios[0] ?? null,
    quality: workflow.settings.quality ? (workflow.supported_quality_levels[0] ?? null) : null,
    motionStrength: 60,
    promptAdherence: 75,
    seedLocked: false,
    lyrics: "",
    lyricsLanguage: workflow.settings.lyrics ? LYRIC_LANGUAGES[0] : null,
    inputs: Object.fromEntries(workflow.inputs.map((input) => [input.role, null])),
  };
}

/**
 * Seeds the workflow's required media input with an asset the user already has.
 *
 * This is what makes "Extend" on a finished result an actual hand-off rather
 * than a bare link: the source video arrives with the navigation instead of
 * asking the user to download their own generation and upload it back. The
 * role comes from the workflow definition, so nothing here names a workflow.
 *
 * Returns the values unchanged when there is no asset or the workflow takes no
 * required input — a `?source=` on Text to Video is meaningless, not an error.
 */
export function withSourceAsset(
  workflow: Workflow,
  values: GenerationFormValues,
  assetId: string | null | undefined,
): GenerationFormValues {
  const role = workflow.inputs.find((input) => input.required)?.role;
  if (!assetId || !role) return values;
  return { ...values, inputs: { ...values.inputs, [role]: assetId } };
}

/**
 * Carries settings across a workflow switch.
 *
 * The approved behaviour: keep duration / aspect / quality when the incoming
 * workflow supports the current value, otherwise fall back to its first
 * supported value. Worked example the tests cover —
 *
 *   Text to Video @ 15s → Image to Video (5s / 10s only) → 5s
 *
 * It must not crash and must not keep an unsupported 15s.
 *
 * Inputs are deliberately NOT carried across: a source video is not a valid
 * source image, and silently keeping one would submit a request the API is
 * right to reject.
 */
export function preserveValues(
  workflow: Workflow,
  previous: GenerationFormValues | undefined,
): GenerationFormValues {
  const defaults = defaultValuesFor(workflow);
  if (!previous) return defaults;

  return {
    ...defaults,
    prompt: previous.prompt,
    motionStrength: previous.motionStrength,
    promptAdherence: previous.promptAdherence,
    seedLocked: previous.seedLocked,
    // Lyrics travel only between workflows that both have the control —
    // stale lyrics on a video workflow would trip its own validation.
    lyrics: workflow.settings.lyrics ? previous.lyrics : defaults.lyrics,
    lyricsLanguage:
      workflow.settings.lyrics &&
      previous.lyricsLanguage &&
      (LYRIC_LANGUAGES as readonly string[]).includes(previous.lyricsLanguage)
        ? previous.lyricsLanguage
        : defaults.lyricsLanguage,
    duration:
      previous.duration !== null && workflow.supported_durations.includes(previous.duration)
        ? previous.duration
        : defaults.duration,
    aspectRatio:
      previous.aspectRatio && workflow.supported_aspect_ratios.includes(previous.aspectRatio)
        ? previous.aspectRatio
        : defaults.aspectRatio,
    quality:
      previous.quality &&
      workflow.settings.quality &&
      workflow.supported_quality_levels.includes(previous.quality)
        ? previous.quality
        : defaults.quality,
  };
}

/** Form values → the exact body `POST /generations` accepts. */
export function toCreateInput(
  workflow: Workflow,
  values: GenerationFormValues,
): CreateGenerationInput {
  const inputs: Record<string, string> = {};
  for (const [role, assetId] of Object.entries(values.inputs ?? {})) {
    if (assetId) inputs[role] = assetId;
  }

  return {
    workflow_id: workflow.id,
    prompt: values.prompt.trim(),
    parameters: {
      // Omitted rather than sent as null when the workflow has no such control:
      // the API rejects a parameter a workflow does not use, and rightly so.
      // Source-mode duration is the clearest case — the length comes from the
      // uploaded file, and sending one would be rejected as a contradiction.
      ...(values.duration !== null ? { duration: values.duration } : {}),
      ...(workflow.supported_aspect_ratios.length ? { aspect_ratio: values.aspectRatio } : {}),
      ...(workflow.settings.quality && workflow.supported_quality_levels.length
        ? { quality: values.quality }
        : {}),
      ...(workflow.settings.motion_strength ? { motion_strength: values.motionStrength } : {}),
      ...(workflow.settings.prompt_adherence ? { prompt_adherence: values.promptAdherence } : {}),
      ...(workflow.settings.seed && values.seedLocked ? { seed: 123456 } : {}),
      // The customer's own words are sent exactly as typed (trimmed only at
      // the ends) — the platform never rewrites a lyric sheet. The language
      // matters when we write the lyrics, so it is sent whenever the control
      // exists; the API stores it with the job either way.
      ...(workflow.settings.lyrics && values.lyrics.trim()
        ? { lyrics: values.lyrics.trim() }
        : {}),
      ...(workflow.settings.lyrics && values.lyricsLanguage
        ? { lyrics_language: values.lyricsLanguage }
        : {}),
    },
    ...(Object.keys(inputs).length ? { inputs } : {}),
  };
}

/** Prefills the form from a finished job — "Reuse settings". */
export function valuesFromJob(
  workflow: Workflow,
  prompt: string,
  parameters: Record<string, unknown>,
): GenerationFormValues {
  const defaults = defaultValuesFor(workflow);

  const pick = (value: unknown, allowed: readonly string[], fallback: string | null) =>
    typeof value === "string" && allowed.includes(value) ? value : fallback;

  return {
    ...defaults,
    prompt,
    duration: pick(parameters.duration, workflow.supported_durations, defaults.duration),
    aspectRatio: pick(
      parameters.aspect_ratio,
      workflow.supported_aspect_ratios,
      defaults.aspectRatio,
    ),
    quality: pick(
      parameters.quality,
      workflow.settings.quality ? workflow.supported_quality_levels : [],
      defaults.quality,
    ),
    motionStrength:
      typeof parameters.motion_strength === "number"
        ? parameters.motion_strength
        : defaults.motionStrength,
    promptAdherence:
      typeof parameters.prompt_adherence === "number"
        ? parameters.prompt_adherence
        : defaults.promptAdherence,
    seedLocked: parameters.seed != null,
    lyrics:
      workflow.settings.lyrics && typeof parameters.lyrics === "string"
        ? parameters.lyrics
        : defaults.lyrics,
    lyricsLanguage: pick(
      parameters.lyrics_language,
      workflow.settings.lyrics ? LYRIC_LANGUAGES : [],
      defaults.lyricsLanguage,
    ),
  };
}
