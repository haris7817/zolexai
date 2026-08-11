"use client";

import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import type { Workflow } from "@zolexai/workflow-contracts";
import { queryKeys } from "@/lib/query";
import { fetchWorkflows } from "@/services/workflows";

/**
 * Workflow catalogue, from the API.
 *
 * The app shell renders on the server and seeds this cache with the build-time
 * YAML catalogue (`catalog.server.ts`), so the first paint has a full sidebar
 * instead of skeletons — and the query still runs, so what the user interacts
 * with is genuinely what the API serves. The two agree by construction: both
 * read the same version-controlled files, and a parity test proves it.
 *
 * Definitions change only on deploy, hence the long stale time.
 */
export function useWorkflows() {
  const query = useQuery({
    queryKey: queryKeys.workflows.all,
    queryFn: ({ signal }) => fetchWorkflows(signal),
    staleTime: 10 * 60_000,
  });

  return { ...query, workflows: query.data ?? [] };
}

export function useWorkflow(workflowId: string | null | undefined): Workflow | undefined {
  const { workflows } = useWorkflows();
  return useMemo(
    () => workflows.find((workflow) => workflow.id === workflowId),
    [workflows, workflowId],
  );
}
