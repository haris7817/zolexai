/**
 * MOCK — PRE-M1 demo only. A real media library arrives at M3.07
 * (signed uploads M3.05, signed downloads M3.06).
 */

export type MediaKind = "video" | "image" | "audio";
export type MediaSource = "generated" | "upload";

export interface MockMediaItem {
  id: string;
  name: string;
  kind: MediaKind;
  source: MediaSource;
  /** Duration for video/audio; null for images. */
  duration: string | null;
  dimensions: string | null;
  size: string;
  createdLabel: string;
  order: number;
  thumb: string;
}

const g = (a: string, b: string) => `linear-gradient(140deg, ${a}, ${b})`;

export const mockMedia: MockMediaItem[] = [
  {
    id: "asset_701",
    name: "Misty forest dolly",
    kind: "video",
    source: "generated",
    duration: "0:10",
    dimensions: "1920 × 1080",
    size: "18.4 MB",
    createdLabel: "12 min ago",
    order: 1,
    thumb: g("#141A0C", "#28340D"),
  },
  {
    id: "asset_700",
    name: "Portrait — wind motion",
    kind: "video",
    source: "generated",
    duration: "0:05",
    dimensions: "1080 × 1920",
    size: "9.1 MB",
    createdLabel: "1 hour ago",
    order: 2,
    thumb: g("#33430D", "#182008"),
  },
  {
    id: "asset_699",
    name: "Synthwave 120bpm",
    kind: "audio",
    source: "generated",
    duration: "1:00",
    dimensions: null,
    size: "2.3 MB",
    createdLabel: "3 hours ago",
    order: 3,
    thumb: g("#23262B", "#0D0F12"),
  },
  {
    id: "asset_698",
    name: "Studio portrait reference",
    kind: "image",
    source: "upload",
    duration: null,
    dimensions: "2048 × 2560",
    size: "4.7 MB",
    createdLabel: "3 hours ago",
    order: 4,
    thumb: g("#26320C", "#141A08"),
  },
  {
    id: "asset_697",
    name: "Coastline sunset aerial",
    kind: "video",
    source: "generated",
    duration: "0:10",
    dimensions: "1920 × 1080",
    size: "17.9 MB",
    createdLabel: "5 hours ago",
    order: 5,
    thumb: g("#222C10", "#121808"),
  },
  {
    id: "asset_696",
    name: "Skate footage — source",
    kind: "video",
    source: "upload",
    duration: "0:22",
    dimensions: "1920 × 1080",
    size: "41.2 MB",
    createdLabel: "Yesterday",
    order: 6,
    thumb: g("#1C232A", "#0B0E11"),
  },
  {
    id: "asset_695",
    name: "Neon city — 60s cut",
    kind: "video",
    source: "generated",
    duration: "1:00",
    dimensions: "1920 × 1080",
    size: "96.8 MB",
    createdLabel: "Yesterday",
    order: 7,
    thumb: g("#2C3A0B", "#141A08"),
  },
  {
    id: "asset_694",
    name: "Ambient documentary score",
    kind: "audio",
    source: "generated",
    duration: "2:00",
    dimensions: null,
    size: "4.6 MB",
    createdLabel: "3 days ago",
    order: 8,
    thumb: g("#18200A", "#0E1207"),
  },
  {
    id: "asset_693",
    name: "Concept art — chrome city",
    kind: "image",
    source: "upload",
    duration: null,
    dimensions: "1536 × 1536",
    size: "3.1 MB",
    createdLabel: "3 days ago",
    order: 9,
    thumb: g("#26261F", "#101010"),
  },
  {
    id: "asset_692",
    name: "Product loop — matte bottle",
    kind: "video",
    source: "generated",
    duration: "0:05",
    dimensions: "1080 × 1080",
    size: "7.4 MB",
    createdLabel: "4 days ago",
    order: 10,
    thumb: g("#111708", "#26320C"),
  },
  {
    id: "asset_691",
    name: "Reference still — rain window",
    kind: "image",
    source: "upload",
    duration: null,
    dimensions: "3024 × 4032",
    size: "6.2 MB",
    createdLabel: "1 week ago",
    order: 11,
    thumb: g("#141A0C", "#222C10"),
  },
  {
    id: "asset_690",
    name: "Music video — synthwave cut",
    kind: "video",
    source: "generated",
    duration: "0:30",
    dimensions: "1080 × 1920",
    size: "52.3 MB",
    createdLabel: "1 week ago",
    order: 12,
    thumb: g("#23262B", "#0D0F12"),
  },
];
