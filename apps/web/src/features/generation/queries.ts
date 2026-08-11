"use client";

import { useMemo } from "react";
import {
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import {
  isRunning,
  type GenerationJob,
  type JobStatus,
} from "@zolexai/workflow-contracts";
import { queryKeys } from "@/lib/query";
import {
  cancelGeneration,
  createGeneration,
  fetchGeneration,
  listGenerations,
  type CreateGenerationInput,
} from "@/services/generations";

/**
 * Server-state hooks for generations.
 *
 * Every screen reads jobs from here. There is no client-side copy of job state
 * to keep in sync — the cache is the single client-side view of the server's
 * truth, and SSE writes into that same cache (`useGenerationStream`).
 */

/** How many recent jobs the workspace strip shows. */
const RECENT_LIMIT = 8;

export function useGeneration(jobId: string | null) {
  return useQuery({
    queryKey: queryKeys.generations.detail(jobId ?? ""),
    queryFn: ({ signal }) => fetchGeneration(jobId as string, signal),
    enabled: Boolean(jobId),
    // Live updates arrive over SSE, so this is never polled. It is refetched
    // deliberately when a stream reports a terminal state.
    staleTime: 60_000,
  });
}

export interface GenerationFilters {
  status?: JobStatus[];
  workflowId?: string | null;
}

/**
 * Paginated history.
 *
 * `useInfiniteQuery` with the API's opaque cursor — never a page number. The
 * cursor is a keyset position, so page 200 costs the same as page 1
 * (directive §5).
 */
export function useGenerationHistory(filters: GenerationFilters = {}) {
  return useInfiniteQuery({
    queryKey: queryKeys.generations.list(filters as Record<string, unknown>),
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      listGenerations(
        {
          limit: 24,
          cursor: pageParam,
          status: filters.status,
          workflowId: filters.workflowId,
        },
        signal,
      ),
    getNextPageParam: (lastPage) => (lastPage.has_more ? lastPage.next_cursor : undefined),
  });
}

/** The workspace job strip — the most recent jobs, whatever their state. */
export function useRecentGenerations() {
  const query = useQuery({
    queryKey: queryKeys.generations.list({ recent: RECENT_LIMIT }),
    queryFn: ({ signal }) => listGenerations({ limit: RECENT_LIMIT }, signal),
    staleTime: 15_000,
  });

  const jobs = useMemo(() => query.data?.items ?? [], [query.data]);
  const active = useMemo(() => jobs.filter((job) => isRunning(job.status)), [jobs]);

  return { ...query, jobs, active };
}

/**
 * Jobs still running, for the sidebar indicator.
 *
 * Its own request rather than a filter over the recent list, because the
 * indicator lives in the app shell on every screen and must not depend on a
 * workspace-scoped query being mounted.
 */
export function useActiveGenerations() {
  const query = useQuery({
    queryKey: queryKeys.generations.list({ active: true }),
    queryFn: ({ signal }) =>
      listGenerations(
        {
          limit: 20,
          status: ["queued", "assigned", "preparing", "generating", "post_processing", "uploading"],
        },
        signal,
      ),
    staleTime: 15_000,
    // A safety net, not the delivery mechanism: SSE covers the job being
    // watched, but a job started in another tab has no stream here. One minute
    // is slow enough to be negligible load and fast enough to stay honest.
    refetchInterval: 60_000,
  });

  return { ...query, jobs: query.data?.items ?? [] };
}

/** A key that makes a resubmission of the SAME request idempotent. */
function newIdempotencyKey(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Submits a generation.
 *
 * Each submission gets a fresh `Idempotency-Key`, so a genuine second
 * generation is never suppressed — while a double-click, a re-render or a
 * network retry of that same submission returns the original job instead of
 * starting a second, separately-billed one (directive §24).
 */
export function useCreateGeneration() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (input: CreateGenerationInput) =>
      createGeneration(input, newIdempotencyKey()),
    onSuccess: async (accepted) => {
      // Seed the detail cache so the canvas renders "Queued" immediately,
      // rather than flashing empty until the first fetch returns.
      queryClient.setQueryData<GenerationJob | undefined>(
        queryKeys.generations.detail(accepted.job_id),
        (previous) => previous,
      );
      await queryClient.invalidateQueries({ queryKey: queryKeys.generations.all });
    },
  });
}

export function useCancelGeneration() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (jobId: string) => cancelGeneration(jobId),
    onSuccess: async (job) => {
      queryClient.setQueryData(queryKeys.generations.detail(job.id), job);
      await queryClient.invalidateQueries({ queryKey: queryKeys.generations.all });
    },
  });
}
