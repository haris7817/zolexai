"use client";

import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { jobEventSchema, isTerminal, type GenerationJob } from "@zolexai/workflow-contracts";
import { API_V1 } from "@/lib/api/client";
import { queryKeys } from "@/lib/query";

/**
 * ===========================================================================
 * Live generation progress over Server-Sent Events
 * ===========================================================================
 *
 * Replaces the PRE-M1 timer simulation and, equally, replaces polling. One
 * connection per watched job delivers status, stage, progress and the terminal
 * result as the worker reports them (directive §10).
 *
 * ## Why `EventSource` rather than `fetch` + a reader
 *
 * `EventSource` reconnects on its own after a dropped connection and replays
 * the last `id:` it saw as a `Last-Event-ID` header — which is exactly the
 * cursor the API needs to resume from PostgreSQL without losing an event. That
 * behaviour is free here and would have to be hand-written otherwise.
 *
 * ## How this reaches the UI
 *
 * Events are written straight into the React Query cache for the job. Every
 * component reading `useGeneration(jobId)` re-renders — canvas, job strip,
 * sidebar indicator — with no prop threading and no second source of state.
 *
 * The stream closes when the job reaches a terminal state. It must: a browser
 * left open on a finished job would otherwise hold a connection and a Redis
 * subscriber indefinitely.
 */

export type StreamState = "idle" | "connecting" | "open" | "closed" | "error";

export function useGenerationStream(
  jobId: string | null,
  options: { enabled?: boolean } = {},
): StreamState {
  const enabled = options.enabled ?? true;
  const queryClient = useQueryClient();
  const [state, setState] = useState<StreamState>("idle");

  // Held in a ref so the cleanup below closes the exact instance it opened,
  // even if a re-render happens between open and unmount.
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId || !enabled) {
      setState("idle");
      return;
    }

    let closed = false;
    setState("connecting");

    const source = new EventSource(`${API_V1}/generations/${jobId}/events`, {
      withCredentials: true,
    });
    sourceRef.current = source;

    const close = () => {
      if (closed) return;
      closed = true;
      source.close();
    };

    source.onopen = () => setState("open");

    const handle = (raw: MessageEvent<string>) => {
      const parsed = jobEventSchema.safeParse(JSON.parse(raw.data));
      if (!parsed.success) return; // ignore a frame we cannot understand
      const event = parsed.data;

      queryClient.setQueryData<GenerationJob>(
        queryKeys.generations.detail(jobId),
        (previous) => {
          if (!previous) return previous;
          // Guard against an out-of-order frame: progress must never move
          // backwards, which on screen reads as a fault even on a healthy job.
          if (event.seq <= previous.last_event_seq) return previous;
          return {
            ...previous,
            status: event.status,
            stage_label: event.stage_label,
            progress: Math.max(previous.progress, event.progress),
            hint: event.message,
            last_event_seq: event.seq,
            is_terminal: isTerminal(event.status),
          };
        },
      );

      if (isTerminal(event.status)) {
        // The terminal frame carries no output asset URL — refetch the job for
        // the result, and refresh the lists this job now belongs to.
        queryClient.invalidateQueries({ queryKey: queryKeys.generations.detail(jobId) });
        queryClient.invalidateQueries({ queryKey: queryKeys.generations.all });
        queryClient.invalidateQueries({ queryKey: queryKeys.media.all });
        close();
        setState("closed");
      }
    };

    // The API names its events, so each type is subscribed explicitly rather
    // than relying on `onmessage` (which only receives unnamed events).
    for (const name of ["status", "progress", "completed", "failed", "cancelled"]) {
      source.addEventListener(name, handle as EventListener);
    }

    source.onerror = () => {
      if (closed) return;
      // EventSource retries on its own and resumes from Last-Event-ID, so this
      // is usually transient. Surfaced so the UI can say "reconnecting" rather
      // than appearing frozen.
      setState(source.readyState === EventSource.CLOSED ? "error" : "connecting");
    };

    return () => {
      close();
      sourceRef.current = null;
    };
  }, [jobId, enabled, queryClient]);

  return state;
}
