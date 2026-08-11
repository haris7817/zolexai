"use client";

import { useState, type ReactNode } from "react";
import { QueryClientProvider } from "@tanstack/react-query";
import type { Workflow } from "@zolexai/workflow-contracts";
import { createQueryClient, queryKeys } from "@/lib/query";

/**
 * Application providers.
 *
 * The QueryClient is created inside `useState` rather than at module scope.
 * A module-level client is shared by every request the Node server handles,
 * which on the server would leak one user's cached data into another's render.
 * Creating it per component instance keeps it per-request on the server and
 * stable across re-renders on the client.
 *
 * `initialWorkflows` comes from the server component that renders this — the
 * catalogue read straight from the YAML definitions at render time. Seeding it
 * here means the sidebar and tool grids paint complete on first render instead
 * of as skeletons, while the live query still runs and takes over.
 */
export function Providers({
  children,
  initialWorkflows,
}: {
  children: ReactNode;
  initialWorkflows?: Workflow[];
}) {
  const [queryClient] = useState(() => {
    const client = createQueryClient();
    if (initialWorkflows?.length) {
      client.setQueryData(queryKeys.workflows.all, initialWorkflows);
    }
    return client;
  });

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
