/**
 * ===========================================================================
 * The ZolexAI API client — the ONLY place `fetch` is called
 * ===========================================================================
 *
 * No component calls the network directly (directive §25). Everything goes
 * through here, so retry policy, error shape, credentials and the base URL are
 * decided once instead of drifting across twenty call sites.
 *
 * Responses are parsed through the shared Zod contracts before they are
 * returned. A field the Python backend renamed becomes a clear error naming the
 * field, at the boundary, rather than `undefined` surfacing three components
 * deep with no clue where it came from.
 */

import { apiErrorSchema, type ErrorCode } from "@zolexai/workflow-contracts";
import type { z } from "zod";

/**
 * Base URL, from the environment — never a hard-coded host
 * (scalability rule #14). Empty means same-origin, which is what a production
 * deployment behind one domain uses.
 */
export const API_BASE = (process.env.NEXT_PUBLIC_API_URL ?? "").replace(/\/$/, "");

export const API_V1 = `${API_BASE}/api/v1`;

/** A failure the API described. `code` is stable; `message` is customer-safe. */
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details: Record<string, unknown>;
  readonly requestId?: string;

  constructor(
    status: number,
    code: string,
    message: string,
    details: Record<string, unknown> = {},
    requestId?: string,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
    this.requestId = requestId;
  }

  is(code: ErrorCode): boolean {
    return this.code === code;
  }

  /** 4xx means the request was wrong — repeating it unchanged cannot help. */
  get isRetryable(): boolean {
    return this.status >= 500 || this.status === 429;
  }

  /** Per-field problems, when the API returned them. Feeds form errors. */
  get fieldErrors(): { field: string; reason: string }[] {
    const fields = this.details.fields;
    if (!Array.isArray(fields)) return [];
    return fields.flatMap((entry) =>
      entry && typeof entry === "object" && "field" in entry
        ? [
            {
              field: String((entry as Record<string, unknown>).field),
              reason: String((entry as Record<string, unknown>).reason ?? "Invalid"),
            },
          ]
        : [],
    );
  }
}

/** Thrown when a response does not match the agreed contract. */
export class ContractError extends Error {
  constructor(path: string, issues: string) {
    super(`The server returned an unexpected shape for ${path}: ${issues}`);
    this.name = "ContractError";
  }
}

export interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  signal?: AbortSignal;
  headers?: Record<string, string>;
  /** Cache directive for server-side calls. Client fetches are always live. */
  revalidate?: number | false;
}

async function toApiError(response: Response): Promise<ApiError> {
  let code = "internal_error";
  let message = "Something went wrong. Please try again.";
  let details: Record<string, unknown> = {};
  let requestId: string | undefined;

  try {
    const parsed = apiErrorSchema.safeParse(await response.json());
    if (parsed.success) {
      code = parsed.data.error.code;
      message = parsed.data.error.message;
      details = parsed.data.error.details;
      requestId = parsed.data.error.request_id ?? undefined;
    }
  } catch {
    // A non-JSON body (a proxy error page, a dropped connection) is not worth
    // failing over — the status code still tells the user what happened.
  }

  return new ApiError(
    response.status,
    code,
    message,
    details,
    requestId ?? response.headers.get("X-Request-ID") ?? undefined,
  );
}

/**
 * Performs one request and validates the response against `schema`.
 *
 * `credentials: "include"` is set now, before authentication exists, because
 * M3 introduces HttpOnly session cookies and CORS must already be configured
 * for credentialed requests — a change that is invisible today and expensive to
 * discover later.
 */
export async function apiFetch<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  options: RequestOptions = {},
): Promise<z.infer<S>> {
  const { method = "GET", body, signal, headers = {}, revalidate } = options;

  const response = await fetch(`${API_V1}${path}`, {
    method,
    signal,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
    ...(revalidate !== undefined ? { next: { revalidate } } : {}),
  });

  if (!response.ok) {
    throw await toApiError(response);
  }

  const parsed = schema.safeParse(await response.json());
  if (!parsed.success) {
    throw new ContractError(
      path,
      parsed.error.issues
        .slice(0, 3)
        .map((issue) => `${issue.path.join(".") || "(root)"}: ${issue.message}`)
        .join("; "),
    );
  }
  return parsed.data;
}

/**
 * Uploads a file straight to object storage via a presigned URL.
 *
 * Note it does NOT go through `apiFetch`: the target is the storage provider,
 * not our API, and it must carry no credentials. The headers come from the
 * presign response and must be sent verbatim — `Content-Type` is part of the
 * signature, so altering it makes storage reject the upload.
 */
export async function uploadToStorage(
  url: string,
  method: string,
  headers: Record<string, string>,
  file: File,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(url, { method, headers, body: file, signal });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      "upload_failed",
      "The upload could not be completed. Please try again.",
    );
  }
}
