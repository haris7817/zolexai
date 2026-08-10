import type { WorkflowDefinition, WorkflowId } from "./types";

/**
 * ===========================================================================
 * THE ZolexAI workflow registry — one source of truth for all six tools.
 * ===========================================================================
 *
 * Lifted verbatim from the approved `Video Workspace.dc.html` (lines 358–417).
 *
 * Everything workflow-shaped reads from here:
 *   · sidebar + mobile drawer navigation
 *   · the Workspace settings panel (inputs, durations, ratios, quality, advanced)
 *   · the Workspace result actions (capability-driven)
 *   · Creator Dashboard quick actions
 *   · All Tools
 *   · Landing tool grid
 *
 * RULES
 *   1. Never branch on a workflow id inside a component. If a screen needs to
 *      behave differently, that difference belongs in this file as metadata.
 *   2. Never add a tool here that is not in the frozen scope (milestones §8.1).
 *      Landing's design shows a seventh card, "AI Editing Tools" — it is a
 *      design placeholder, is NOT in scope, and is deliberately absent so no
 *      surface can imply it exists. See ADR 0001 §5.
 *   3. No provider, model or infrastructure name may ever appear here.
 *      Private execution metadata stays server-side (architecture §11).
 *
 * A gradient helper matching the design's `g(a, b)`.
 */
const g = (a: string, b: string) => `linear-gradient(140deg, ${a}, ${b})`;

export const WORKFLOWS: Record<WorkflowId, WorkflowDefinition> = {
  "text-to-video": {
    id: "text-to-video",
    name: "Text to Video",
    icon: "sparkles",
    category: "video",
    outputType: "video",
    description: "Describe a scene and generate cinematic video",
    shortDescription: "From prompt to motion",
    marketingDescription:
      "Describe a scene and watch it come to life as cinematic video.",
    inputTypes: ["prompt"],
    promptPlaceholder: "Describe the video you want to create…",
    supportedDurations: ["5s", "10s", "15s", "30s"],
    supportedAspectRatios: ["16:9", "9:16", "1:1", "4:5"],
    supportedQualityLevels: ["Standard", "High", "Ultra"],
    settings: {
      quality: true,
      motionStrength: true,
      promptAdherence: true,
      seed: true,
    },
    capabilities: {
      download: true,
      extend: true,
      reuseSettings: true,
      variation: true,
    },
    thumb: g("#4A2FA0", "#2A1F55"),
  },

  "image-to-video": {
    id: "image-to-video",
    name: "Image to Video",
    icon: "image",
    category: "video",
    outputType: "video",
    description: "Animate a still image into natural motion",
    shortDescription: "Animate any image",
    marketingDescription:
      "Animate any still image into smooth, natural motion.",
    inputTypes: ["image", "prompt"],
    promptPlaceholder: "Describe how the image should move…",
    mediaRequirements: { kind: "an image", label: "INPUT IMAGE" },
    supportedDurations: ["5s", "10s"],
    supportedAspectRatios: ["16:9", "9:16", "1:1"],
    supportedQualityLevels: ["Standard", "High", "Ultra"],
    settings: {
      quality: true,
      motionStrength: true,
      promptAdherence: true,
      seed: true,
    },
    capabilities: {
      download: true,
      extend: true,
      reuseSettings: true,
      variation: true,
    },
    thumb: g("#6D3DF5", "#1F1A45"),
  },

  "video-to-video": {
    id: "video-to-video",
    name: "Video to Video",
    icon: "repeat",
    category: "video",
    outputType: "video",
    description: "Restyle and transform existing footage",
    shortDescription: "Restyle footage",
    marketingDescription:
      "Restyle and transform existing footage with a prompt.",
    inputTypes: ["video", "prompt"],
    promptPlaceholder: "Describe the transformation…",
    mediaRequirements: { kind: "a video", label: "INPUT VIDEO" },
    supportedDurations: ["5s", "10s", "15s"],
    supportedAspectRatios: ["16:9", "9:16"],
    supportedQualityLevels: ["Standard", "High", "Ultra"],
    settings: {
      quality: true,
      motionStrength: true,
      promptAdherence: true,
      seed: false,
    },
    capabilities: {
      download: true,
      extend: true,
      reuseSettings: true,
      variation: true,
    },
    thumb: g("#3B2B85", "#241C4E"),
  },

  "extend-video": {
    id: "extend-video",
    name: "Extend Video",
    icon: "extend",
    category: "video",
    outputType: "video",
    description: "Continue a generated video seamlessly",
    shortDescription: "Continue any clip",
    marketingDescription:
      "Continue any generated video seamlessly, as long as you need.",
    inputTypes: ["video", "prompt"],
    promptPlaceholder: "Describe what happens next…",
    mediaRequirements: { kind: "a video", label: "SOURCE VIDEO" },
    supportedDurations: ["5s", "10s", "30s", "60s"],
    supportedAspectRatios: ["16:9", "9:16"],
    supportedQualityLevels: ["Standard", "High", "Ultra"],
    settings: {
      quality: true,
      motionStrength: false,
      promptAdherence: true,
      seed: false,
    },
    capabilities: {
      download: true,
      extend: true,
      reuseSettings: true,
      variation: false,
    },
    thumb: g("#5636C9", "#1C1838"),
  },

  music: {
    id: "music",
    name: "Music",
    icon: "music",
    category: "audio",
    outputType: "audio",
    description: "Generate original tracks from a description",
    shortDescription: "Original tracks",
    marketingDescription:
      "Generate original tracks from a mood, style or description.",
    inputTypes: ["prompt"],
    promptPlaceholder: "Describe the track — mood, style, tempo…",
    supportedDurations: ["30s", "60s", "120s"],
    // Audio has no frame — the Workspace hides the whole aspect-ratio section.
    supportedAspectRatios: [],
    supportedQualityLevels: [],
    settings: {
      quality: false,
      motionStrength: false,
      promptAdherence: true,
      seed: true,
    },
    // Audio cannot be extended — Extend must not appear on a music result.
    capabilities: {
      download: true,
      extend: false,
      reuseSettings: true,
      variation: true,
    },
    thumb: g("#2E2260", "#171331"),
  },

  "music-video": {
    id: "music-video",
    name: "Music Video",
    icon: "clapper",
    category: "audio",
    outputType: "video",
    description: "Pair audio with visual direction",
    shortDescription: "Audio meets visuals",
    marketingDescription:
      "Pair audio with visual direction to produce full music videos.",
    inputTypes: ["audio", "prompt"],
    promptPlaceholder: "Describe the visual direction…",
    mediaRequirements: { kind: "an audio track", label: "AUDIO" },
    supportedDurations: ["15s", "30s", "60s"],
    supportedAspectRatios: ["16:9", "9:16", "1:1"],
    supportedQualityLevels: ["Standard", "High"],
    settings: {
      quality: true,
      motionStrength: true,
      promptAdherence: true,
      seed: false,
    },
    capabilities: {
      download: true,
      extend: false,
      reuseSettings: true,
      variation: true,
    },
    thumb: g("#35256E", "#211A48"),
  },
};

/** Display order — used by every tool surface so they never disagree. */
export const WORKFLOW_ORDER: WorkflowId[] = [
  "text-to-video",
  "image-to-video",
  "video-to-video",
  "extend-video",
  "music",
  "music-video",
];

export const WORKFLOW_LIST: WorkflowDefinition[] = WORKFLOW_ORDER.map(
  (id) => WORKFLOWS[id],
);

export const DEFAULT_WORKFLOW_ID: WorkflowId = "text-to-video";

export function isWorkflowId(value: string): value is WorkflowId {
  return value in WORKFLOWS;
}

/** Returns undefined for unknown ids so routes can render notFound(). */
export function getWorkflow(id: string): WorkflowDefinition | undefined {
  return isWorkflowId(id) ? WORKFLOWS[id] : undefined;
}
