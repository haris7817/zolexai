import type { WorkflowId, OutputType } from "@/features/workflows/types";

/**
 * MOCK — PRE-M1 demo only. Real history arrives at M3.08.
 *
 * Covers every status the Generations screen must present, including `Failed`,
 * so the client can approve how errors read before the backend can produce one.
 */

export type MockGenerationStatus =
  | "Completed"
  | "Generating"
  | "Queued"
  | "Failed";

export interface MockGeneration {
  id: string;
  prompt: string;
  workflowId: WorkflowId;
  workflowName: string;
  outputType: OutputType;
  status: MockGenerationStatus;
  /** Progress 0–100, only meaningful while Generating. */
  progress?: number;
  duration: string;
  aspect: string | null;
  quality: string | null;
  createdLabel: string;
  /** Sort key — smaller is more recent. */
  order: number;
  thumb: string;
  /** Present on Failed rows; friendly copy only, never a stack trace (rule #10). */
  errorMessage?: string;
}

const g = (a: string, b: string) => `linear-gradient(140deg, ${a}, ${b})`;

export const mockGenerations: MockGeneration[] = [
  {
    id: "gen_2481",
    prompt: "Slow dolly through a misty forest at first light, volumetric rays",
    workflowId: "text-to-video",
    workflowName: "Text to Video",
    outputType: "video",
    status: "Generating",
    progress: 62,
    duration: "10s",
    aspect: "16:9",
    quality: "High",
    createdLabel: "12 min ago",
    order: 1,
    thumb: g("#141A0C", "#28340D"),
  },
  {
    id: "gen_2480",
    prompt: "Vintage car commercial, film grain, golden hour reflections",
    workflowId: "text-to-video",
    workflowName: "Text to Video",
    outputType: "video",
    status: "Queued",
    duration: "15s",
    aspect: "16:9",
    quality: "Ultra",
    createdLabel: "18 min ago",
    order: 2,
    thumb: g("#26320C", "#141A08"),
  },
  {
    id: "gen_2479",
    prompt: "Portrait photo brought to life with subtle wind motion",
    workflowId: "image-to-video",
    workflowName: "Image to Video",
    outputType: "video",
    status: "Completed",
    duration: "5s",
    aspect: "9:16",
    quality: "High",
    createdLabel: "1 hour ago",
    order: 3,
    thumb: g("#23262B", "#0D0F12"),
  },
  {
    id: "gen_2478",
    prompt: "Synthwave track, 120bpm, retro analog pads and driving bass",
    workflowId: "music",
    workflowName: "Music",
    outputType: "audio",
    status: "Completed",
    duration: "60s",
    aspect: null,
    quality: null,
    createdLabel: "3 hours ago",
    order: 4,
    thumb: g("#18200A", "#10160A"),
  },
  {
    id: "gen_2477",
    prompt: "Aerial coastline at sunset, drone push-in over breaking waves",
    workflowId: "text-to-video",
    workflowName: "Text to Video",
    outputType: "video",
    status: "Completed",
    duration: "10s",
    aspect: "16:9",
    quality: "High",
    createdLabel: "5 hours ago",
    order: 5,
    thumb: g("#222C10", "#121808"),
  },
  {
    id: "gen_2476",
    prompt: "Restyle skate footage as hand-painted animation",
    workflowId: "video-to-video",
    workflowName: "Video to Video",
    outputType: "video",
    status: "Failed",
    duration: "15s",
    aspect: "16:9",
    quality: "High",
    createdLabel: "Yesterday",
    order: 6,
    thumb: g("#1C232A", "#0B0E11"),
    errorMessage:
      "Generation could not be completed. Please try again or adjust your settings.",
  },
  {
    id: "gen_2475",
    prompt: "Neon city flythrough extended to a full sixty seconds",
    workflowId: "extend-video",
    workflowName: "Extend Video",
    outputType: "video",
    status: "Completed",
    duration: "60s",
    aspect: "16:9",
    quality: "High",
    createdLabel: "Yesterday",
    order: 7,
    thumb: g("#1C232A", "#0B0E11"),
  },
  {
    id: "gen_2474",
    prompt: "Synthwave visuals cut to the analog track, chrome and grid",
    workflowId: "music-video",
    workflowName: "Music Video",
    outputType: "video",
    status: "Completed",
    duration: "30s",
    aspect: "9:16",
    quality: "High",
    createdLabel: "2 days ago",
    order: 8,
    thumb: g("#1C240C", "#182008"),
  },
  {
    id: "gen_2473",
    prompt: "Ambient score for a documentary opening, strings and texture",
    workflowId: "music",
    workflowName: "Music",
    outputType: "audio",
    status: "Completed",
    duration: "120s",
    aspect: null,
    quality: null,
    createdLabel: "3 days ago",
    order: 9,
    thumb: g("#26261F", "#101010"),
  },
  {
    id: "gen_2472",
    prompt: "Product loop, matte black bottle rotating on a lit pedestal",
    workflowId: "text-to-video",
    workflowName: "Text to Video",
    outputType: "video",
    status: "Completed",
    duration: "5s",
    aspect: "1:1",
    quality: "Ultra",
    createdLabel: "4 days ago",
    order: 10,
    thumb: g("#111708", "#26320C"),
  },
  {
    id: "gen_2471",
    prompt: "Studio portrait animated with a slow parallax push",
    workflowId: "image-to-video",
    workflowName: "Image to Video",
    outputType: "video",
    status: "Completed",
    duration: "5s",
    aspect: "4:5",
    quality: "High",
    createdLabel: "5 days ago",
    order: 11,
    thumb: g("#28340D", "#10160A"),
  },
  {
    id: "gen_2470",
    prompt: "Rain on a window, shallow depth of field, moody evening light",
    workflowId: "text-to-video",
    workflowName: "Text to Video",
    outputType: "video",
    status: "Completed",
    duration: "10s",
    aspect: "9:16",
    quality: "Standard",
    createdLabel: "1 week ago",
    order: 12,
    thumb: g("#23262B", "#0D0F12"),
  },
];

/** "Continue creating" — drafts on the Creator Dashboard. */
export interface MockDraft {
  id: string;
  title: string;
  workflowName: string;
  editedLabel: string;
  progress: string;
  thumb: string;
}

export const mockDrafts: MockDraft[] = [
  {
    id: "draft_1",
    title: "Neon city flythrough",
    workflowName: "Text to Video",
    editedLabel: "edited 2h ago",
    progress: "70%",
    thumb: g("#222C10", "#121808"),
  },
  {
    id: "draft_2",
    title: "Ocean drone shot, golden hour",
    workflowName: "Image to Video",
    editedLabel: "edited yesterday",
    progress: "40%",
    thumb: g("#2C3A0B", "#141A0C"),
  },
  {
    id: "draft_3",
    title: "Lo-fi study track",
    workflowName: "Music",
    editedLabel: "edited 3d ago",
    progress: "85%",
    thumb: g("#1C232A", "#0B0E11"),
  },
];
