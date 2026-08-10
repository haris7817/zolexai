/**
 * Workflow contract — the public shape of a ZolexAI creation tool.
 *
 * This mirrors the PUBLIC half of the workflow registry described in
 * architecture doc §11. It deliberately contains no execution metadata:
 * no provider, no runner, no workflow file, no VRAM requirement, no model name.
 * Those live server-side only and must never reach the browser (rule #1, R-10).
 */

export type WorkflowId =
  | "text-to-video"
  | "image-to-video"
  | "video-to-video"
  | "extend-video"
  | "music"
  | "music-video";

export type WorkflowCategory = "video" | "audio";

/** Drives which renderer the result canvas uses. */
export type OutputType = "video" | "audio" | "image";

export type InputType = "prompt" | "image" | "video" | "audio";

/** Keys into the ZolexAI icon set — see components/ui/Icon.tsx. */
export type WorkflowIconName =
  | "sparkles"
  | "image"
  | "repeat"
  | "extend"
  | "music"
  | "clapper";

export type AspectRatio = "16:9" | "9:16" | "1:1" | "4:5";

export type QualityLevel = "Standard" | "High" | "Ultra";

/** Which setting controls this workflow exposes. */
export interface WorkflowSettings {
  quality: boolean;
  motionStrength: boolean;
  promptAdherence: boolean;
  seed: boolean;
}

/**
 * Which actions a completed result offers. ResultActions renders from this and
 * nothing else — a workflow that cannot be extended simply never shows Extend.
 */
export interface WorkflowCapabilities {
  download: boolean;
  extend: boolean;
  reuseSettings: boolean;
  variation: boolean;
}

/** Present only when the workflow needs an uploaded asset. */
export interface MediaRequirement {
  /** Reads inside "Drop {kind} here" — e.g. "an image". */
  kind: string;
  /** Section label — e.g. "INPUT IMAGE". */
  label: string;
}

export interface WorkflowDefinition {
  id: WorkflowId;
  name: string;
  icon: WorkflowIconName;
  category: WorkflowCategory;
  outputType: OutputType;

  /** Workspace header subtitle. */
  description: string;
  /** Compact card copy — Dashboard quick actions. */
  shortDescription: string;
  /** Long-form copy — Landing tool grid. */
  marketingDescription: string;

  inputTypes: InputType[];
  promptPlaceholder: string;
  mediaRequirements?: MediaRequirement;

  supportedDurations: string[];
  supportedAspectRatios: AspectRatio[];
  supportedQualityLevels: QualityLevel[];

  settings: WorkflowSettings;
  capabilities: WorkflowCapabilities;

  /** Placeholder gradient used for cards and mock thumbnails. */
  thumb: string;
}
