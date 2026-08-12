import {
  assetSchema,
  mediaCountsSchema,
  pageSchema,
  uploadUrlSchema,
  type Asset,
  type AssetKind,
  type MediaCounts,
  type Page,
} from "@zolexai/workflow-contracts";
import { apiFetch, uploadToStorage } from "@/lib/api/client";
import { z } from "zod";

const assetPageSchema = pageSchema(assetSchema);
const downloadUrlSchema = z.object({ url: z.string(), expires_in: z.number().int() });

export interface ListMediaParams {
  limit?: number;
  cursor?: string | null;
  kind?: AssetKind | null;
  source?: "upload" | "generated" | null;
}

export async function listMedia(
  params: ListMediaParams = {},
  signal?: AbortSignal,
): Promise<Page<Asset>> {
  const query = new URLSearchParams();
  query.set("limit", String(params.limit ?? 24));
  if (params.cursor) query.set("cursor", params.cursor);
  if (params.kind) query.set("kind", params.kind);
  if (params.source) query.set("source", params.source);
  return apiFetch(`/media?${query.toString()}`, assetPageSchema, { signal });
}

export async function fetchMediaCounts(signal?: AbortSignal): Promise<MediaCounts> {
  return apiFetch("/media/counts", mediaCountsSchema, { signal });
}

/** One asset the user owns — used when a screen has an id but not the asset. */
export async function fetchAsset(assetId: string, signal?: AbortSignal): Promise<Asset> {
  return apiFetch(`/assets/${assetId}`, assetSchema, { signal });
}

export async function fetchDownloadUrl(assetId: string): Promise<string> {
  const data = await apiFetch(`/assets/${assetId}/download-url`, downloadUrlSchema, {
    method: "POST",
  });
  return data.url;
}

/**
 * Uploads a file directly to object storage.
 *
 *   1. ask the API for a presigned PUT (it validates type and size first)
 *   2. PUT the file straight to storage
 *   3. confirm, so the API can verify what actually landed
 *
 * Step 2 is the important one: the bytes never pass through Next.js or FastAPI
 * (directive §13). A 500 MB video routed through the API would occupy a server
 * process for the whole transfer.
 *
 * Until step 3 succeeds the asset stays `pending` and cannot be used as a
 * generation input — so a half-finished upload can never reach a worker.
 */
export async function uploadAsset(
  file: File,
  kind: AssetKind,
  options: { signal?: AbortSignal } = {},
): Promise<Asset> {
  const ticket = await apiFetch("/assets/upload-url", uploadUrlSchema, {
    method: "POST",
    signal: options.signal,
    body: {
      filename: file.name,
      content_type: file.type || "application/octet-stream",
      kind,
      size_bytes: file.size,
    },
  });

  await uploadToStorage(
    ticket.upload.url,
    ticket.upload.method,
    ticket.upload.headers,
    file,
    options.signal,
  );

  return apiFetch(`/assets/${ticket.asset_id}/confirm`, assetSchema, {
    method: "POST",
    signal: options.signal,
    body: {},
  });
}

/** Maps a browser MIME type onto the asset kind the API expects. */
export function kindForFile(file: File): AssetKind | null {
  if (file.type.startsWith("video/")) return "video";
  if (file.type.startsWith("image/")) return "image";
  if (file.type.startsWith("audio/")) return "audio";
  return null;
}

export function formatBytes(bytes: number | null): string {
  if (!bytes) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDuration(seconds: number | null): string | null {
  if (seconds == null) return null;
  const total = Math.round(seconds);
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}
