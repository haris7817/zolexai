import { QueryClient } from "@tanstack/react-query";
import { ApiError, ContractError } from "@/lib/api/client";

/**
 * Query keys, in one place.
 *
 * Invalidation is only as reliable as key agreement: a screen that spells its
 * key slightly differently silently stops refreshing when a mutation
 * invalidates. Deriving every key from this object makes that impossible.
 */
export const queryKeys = {
  workflows: {
    all: ["workflows"] as const,
    detail: (id: string) => ["workflows", id] as const,
  },
  generations: {
    all: ["generations"] as const,
    list: (filters: Record<string, unknown>) => ["generations", "list", filters] as const,
    detail: (id: string) => ["generations", "detail", id] as const,
  },
  media: {
    all: ["media"] as const,
    list: (filters: Record<string, unknown>) => ["media", "list", filters] as const,
    counts: ["media", "counts"] as const,
    detail: (id: string) => ["media", "detail", id] as const,
  },
} as const;

export function createQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Generation state arrives by SSE, not by polling, so a short stale
        // window is enough to deduplicate the burst of requests a navigation
        // causes without making screens feel behind.
        staleTime: 30_000,
        gcTime: 5 * 60_000,
        refetchOnWindowFocus: false,
        retry: (failureCount, error) => {
          // Never retry a request the server said was wrong — a 422 repeated
          // three times is three identical rejections and a slower error.
          if (error instanceof ApiError) return error.isRetryable && failureCount < 2;
          // A contract mismatch is a deployment problem; retrying cannot fix it.
          if (error instanceof ContractError) return false;
          return failureCount < 2;
        },
        retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 8000),
      },
      mutations: {
        // Mutations are never retried automatically. `createGeneration` starts
        // paid work; an automatic retry is exactly the duplicate the
        // idempotency key exists to prevent, and deciding to retry belongs to
        // the user, not the cache layer.
        retry: false,
      },
    },
  });
}
